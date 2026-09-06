"""Exact input-routed BF16-output compiler for a declared integer input slice.

This is a research reference, NOT a HF executor or GPU-performance claim.
Construction never enumerates input assignments. Query has no weight array.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Sequence

class BudgetExceeded(RuntimeError):
    pass

class BDD:
    def __init__(self, node_cap: int = 100_000, cache_cap: int = 300_000):
        self.nodes = [(-1, 0, 0), (-1, 1, 1)]
        self.unique: dict[tuple[int,int,int],int] = {}
        self.cache: dict[tuple[str,int,int],int] = {}
        self.node_cap, self.cache_cap = node_cap, cache_cap
        self.apply_calls = 0

    def mk(self, v: int, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        key = (v,lo,hi)
        if key in self.unique:
            return self.unique[key]
        if len(self.nodes)-2 >= self.node_cap:
            raise BudgetExceeded('node_cap')
        i = len(self.nodes)
        self.nodes.append(key)
        self.unique[key] = i
        return i

    def op(self, op: str, a: int, b: int) -> int:
        self.apply_calls += 1
        if a > b:
            a,b = b,a
        if op == '&':
            if a == 0: return 0
            if a == 1: return b
            if a == b: return a
        elif op == '|':
            if a == 0: return b
            if a == 1: return 1
            if a == b: return a
        elif op == '^':
            if a == 0: return b
            if a == b: return 0
            if a == 1 and b == 1: return 0
        else:
            raise ValueError(op)
        key = (op,a,b)
        if key in self.cache:
            return self.cache[key]
        va = self.nodes[a][0] if a > 1 else 10**12
        vb = self.nodes[b][0] if b > 1 else 10**12
        v = min(va,vb)
        aa = self.nodes[a][1:] if va == v else (a,a)
        bb = self.nodes[b][1:] if vb == v else (b,b)
        lo = self.op(op,aa[0],bb[0])
        hi = self.op(op,aa[1],bb[1])
        result = self.mk(v,lo,hi)
        if len(self.cache) >= self.cache_cap:
            raise BudgetExceeded('cache_cap')
        self.cache[key] = result
        return result

    def inv(self,a: int) -> int:
        return self.op('^',a,1)

    def mux(self,c: int,a: int,b: int) -> int:
        """c ? a : b"""
        return self.op('|',self.op('&',c,a),self.op('&',self.inv(c),b))

    def add(self,a: Sequence[int],b: Sequence[int]) -> list[int]:
        if len(a) != len(b): raise ValueError('width mismatch')
        carry=0; out=[]
        for x,y in zip(a,b):
            xy=self.op('^',x,y)
            out.append(self.op('^',xy,carry))
            carry=self.op('|',self.op('&',x,y),self.op('&',xy,carry))
        return out

    def neg(self,a: Sequence[int]) -> list[int]:
        return self.add([self.inv(x) for x in a],[1]+[0]*(len(a)-1))

    def cmul(self,a: Sequence[int],c: int) -> list[int]:
        k=len(a); out=[0]*k
        for shift in range(abs(c).bit_length()):
            if (abs(c)>>shift)&1:
                out=self.add(out,[0]*shift+list(a[:k-shift]))
        return self.neg(out) if c < 0 else out

    def bf16(self,a: Sequence[int]) -> list[int]:
        """Symbolic RNE of a bounded signed integer to the 16 BF16 bits."""
        sign=a[-1]
        na=self.neg(a)
        mag=[self.mux(sign,x,y) for x,y in zip(na,a)]
        out=[0]*16; higher=0; nonzero=0
        for e in range(len(mag)-1,-1,-1):
            lead=self.op('&',mag[e],self.inv(higher))
            higher=self.op('|',higher,mag[e]); nonzero=higher
            if e<=7:
                sig=[0]*(7-e)+mag[:e+1]
                over=0
            else:
                shift=e-7
                sig=mag[shift:e+1]
                sticky=0
                for z in mag[:shift-1]: sticky=self.op('|',sticky,z)
                inc=self.op('&',mag[shift-1],self.op('|',sticky,sig[0]))
                rounded=self.add(sig+[0],[inc]+[0]*8)
                over=rounded[8]; sig=rounded[:8]
            # A carry out makes significand exactly 1.0000000 at e+1.
            frac=[self.op('&',self.inv(over),z) for z in sig[:7]]
            exp=[self.mux(over,((e+128)>>j)&1,((e+127)>>j)&1) for j in range(8)]
            for j,z in enumerate(frac+exp):
                out[j]=self.op('|',out[j],self.op('&',lead,z))
        out[15]=self.op('&',sign,nonzero)
        return out

HEADER=struct.Struct('<4sIIIIII')
NODE=struct.Struct('<III')
PAIR=struct.Struct('<HH')
U32=struct.Struct('<I')

@dataclass(frozen=True)
class Program:
    data: bytes

    def __post_init__(self):
        if len(self.data)<HEADER.size: raise ValueError('truncated header')
        magic,n,m,p,k,num_nodes,num_roots=HEADER.unpack_from(self.data)
        if magic!=b'VRC1' or not (1<=n<=256 and 1<=m<=256 and 1<=p<=8):
            raise ValueError('header/domain')
        if not (2<=k<=26) or num_roots!=16*m:
            raise ValueError('roots/width')
        nvars=n*p
        root_off=HEADER.size+PAIR.size*nvars
        node_off=root_off+U32.size*num_roots
        if node_off+NODE.size*num_nodes!=len(self.data): raise ValueError('length')
        pairs=[PAIR.unpack_from(self.data,HEADER.size+PAIR.size*j) for j in range(nvars)]
        if set(pairs)!={(a,b) for a in range(n) for b in range(p)}:
            raise ValueError('variable permutation')
        roots=[U32.unpack_from(self.data,root_off+4*j)[0] for j in range(num_roots)]
        if any(r>=num_nodes+2 for r in roots): raise ValueError('root id')
        for j in range(num_nodes):
            v,lo,hi=NODE.unpack_from(self.data,node_off+12*j)
            if v>=nvars or lo>=j+2 or hi>=j+2 or lo==hi:
                raise ValueError('invalid acyclic node')
        # Cold program is immutable. Validation is paid once at load, not free.
        for key,val in dict(n=n,m=m,p=p,k=k,num_nodes=num_nodes,num_roots=num_roots,
                            root_off=root_off,node_off=node_off).items():
            object.__setattr__(self,key,val)

    def run(self,x: Sequence[int]) -> tuple[list[int],dict]:
        if len(x)!=self.n or any(type(v)!=int or not(-2**(self.p-1)<=v<2**(self.p-1)) for v in x):
            raise ValueError('input outside registered integer slice')
        # Sparse query memo; no clearing/scanning an array as large as the graph.
        cache={0:0,1:1}; varcache={}; decoded_nodes=set(); blocks64=set(); pages=set()
        requests=0
        def touch(off: int,size: int):
            blocks64.update(range(off//64,(off+size-1)//64+1))
            pages.update(range(off//4096,(off+size-1)//4096+1))
        def ev(node: int) -> int:
            nonlocal requests
            requests+=1
            if node in cache: return cache[node]
            off=self.node_off+NODE.size*(node-2)
            v,lo,hi=NODE.unpack_from(self.data,off)
            touch(off,12); decoded_nodes.add(node)
            if v not in varcache:
                poff=HEADER.size+PAIR.size*v
                j,bit=PAIR.unpack_from(self.data,poff)
                touch(poff,4); varcache[v]=(x[j]>>bit)&1
            result=ev(hi if varcache[v] else lo)
            cache[node]=result
            return result
        y=[]
        for r in range(self.m):
            word=0
            for bit in range(16):
                off=self.root_off+4*(r*16+bit)
                root=U32.unpack_from(self.data,off)[0]
                touch(off,4); word|=ev(root)<<bit
            y.append(word)
        visited=len(decoded_nodes)
        return y,dict(visited_nodes=visited,node_payload_bytes=12*visited,
                      root_payload_bytes=4*self.num_roots,varmap_bytes=4*len(varcache),
                      model_code_bytes=12*visited+4*self.num_roots+4*len(varcache),
                      memo_lookups=requests,memo_inserts=visited,input_bit_tests=len(varcache),
                      memo_value_bytes_lower_bound=5*visited,
                      unique_64byte_blocks=len(blocks64),unique_4096byte_pages=len(pages),
                      input_bytes=2*self.n,output_bytes=2*self.m)


def compile_program(w: Sequence[Sequence[int]],p: int,order: str,
                    node_cap: int=100_000,cache_cap: int=300_000) -> tuple[Program,dict]:
    if not w or not w[0]: raise ValueError('empty weights')
    m,n=len(w),len(w[0])
    if not (1<=n<=256 and 1<=m<=256 and 1<=p<=8): raise ValueError('shape/p')
    if any(len(row)!=n for row in w): raise ValueError('ragged')
    if any(type(c)!=int or abs(c)>127 for row in w for c in row): raise ValueError('weights')
    bound=max(sum(abs(c) for c in row) for row in w)*2**(p-1)
    if bound>=2**24: raise ValueError('not in exact-FP32 slice')
    k=max(2,bound.bit_length()+1)
    if order=='input': pairs=[(j,b) for j in range(n) for b in range(p)]
    elif order=='plane': pairs=[(j,b) for b in range(p) for j in range(n)]
    else: raise ValueError('order')
    manager=BDD(node_cap,cache_cap)
    inputs=[[0]*p for _ in range(n)]
    for v,(j,b) in enumerate(pairs): inputs[j][b]=manager.mk(v,0,1)
    vectors=[a+[a[-1]]*(k-p) if k>=p else a[:k] for a in inputs]
    roots=[]
    try:
        for row in w:
            total=[0]*k
            for c,x in zip(row,vectors): total=manager.add(total,manager.cmul(x,c))
            roots.extend(manager.bf16(total))
    except BudgetExceeded as exc:
        exc.stats=dict(constructed_nodes=len(manager.nodes)-2,apply_calls=manager.apply_calls,
                       cache_entries=len(manager.cache),completed_outputs=len(roots)//16)
        raise
    live=set()
    def visit(u):
        if u<2 or u in live: return
        live.add(u)
        _,lo,hi=manager.nodes[u]; visit(lo); visit(hi)
    for root in roots: visit(root)
    remap={0:0,1:1}
    for old in sorted(live): remap[old]=len(remap)
    data=bytearray(HEADER.pack(b'VRC1',n,m,p,k,len(live),len(roots)))
    for pair in pairs: data.extend(PAIR.pack(*pair))
    for r in roots: data.extend(U32.pack(remap[r]))
    for u in sorted(live):
        v,lo,hi=manager.nodes[u]; data.extend(NODE.pack(v,remap[lo],remap[hi]))
    stats=dict(constructed_nodes=len(manager.nodes)-2,live_nodes=len(live),
               apply_calls=manager.apply_calls,cache_entries=len(manager.cache),
               original_weight_bytes=2*m*n,constructor_coefficient_visits=3*m*n,
               program_bytes=len(data),program_validation_scan_bytes=len(data),
               constructor_logical_node_bytes=12*(len(manager.nodes)-2),
               constructor_python_overhead_not_in_logical_bytes=True,input_truth_table_rows_built=0)
    return Program(bytes(data)),stats


def integer_to_bf16(value: int) -> int:
    """Independent scalar integer RNE, no compiler gate routines."""
    if value==0: return 0
    sign=0x8000 if value<0 else 0; a=abs(value); e=a.bit_length()-1
    if e<=7: return sign|((e+127)<<7)|((a<<(7-e))&127)
    shift=e-7; q,rem=divmod(a,1<<shift)
    if rem>(1<<(shift-1)) or (rem==(1<<(shift-1)) and q&1): q+=1
    if q==256: q=128; e+=1
    return sign|((e+127)<<7)|(q&127)
