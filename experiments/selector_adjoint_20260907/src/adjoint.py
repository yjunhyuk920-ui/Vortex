"""Exact extraction of a selector-linear F2 scalar program.

Coefficients contain current-input Boolean computations. Linear nodes contain
formal selectors, XORs and multiplication by selector-independent coefficients.
No product of two selector-bearing expressions is allowed. This is coefficient
extraction, NOT an STE or differentiation of native floating-point arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
from pathlib import Path

NODE = struct.Struct('<IIII')
HEAD = struct.Struct('<4sIIIII')

@dataclass
class Program:
    ninputs: int
    noutputs: int
    coeff: list[tuple[int, int, int, int]]
    linear: list[tuple[int, int, int, int]]
    root: int

    def validate(self) -> None:
        if not (0 < self.ninputs <= 100000 and 0 < self.noutputs <= 100000):
            raise ValueError('invalid dimensions')
        if not (0 <= self.root < len(self.linear)):
            raise ValueError('invalid root')
        for i, (op,a,b,reserved) in enumerate(self.coeff):
            if reserved or op not in range(4): raise ValueError('coefficient opcode')
            if op==0 and (a not in (0,1) or b): raise ValueError('constant')
            if op==1 and (a>=self.ninputs or b): raise ValueError('input')
            if op in (2,3) and (a>=i or b>=i): raise ValueError('coefficient cycle')
        for i,(op,a,b,reserved) in enumerate(self.linear):
            if reserved or op not in range(4): raise ValueError('linear opcode')
            if op==0 and (a or b): raise ValueError('linear constant must be zero')
            if op==1 and (a>=self.noutputs or b): raise ValueError('selector')
            if op==2 and (a>=i or b>=i): raise ValueError('linear cycle')
            if op==3 and (a>=len(self.coeff) or b>=i): raise ValueError('scale type/cycle')

    def to_bytes(self) -> bytes:
        self.validate()
        return HEAD.pack(b'SLA1',self.ninputs,self.noutputs,len(self.coeff),len(self.linear),self.root)+b''.join(NODE.pack(*n) for n in self.coeff+self.linear)

    @classmethod
    def from_bytes(cls, raw: bytes) -> 'Program':
        if len(raw)<HEAD.size: raise ValueError('truncated header')
        magic,ni,no,nc,nl,root=HEAD.unpack_from(raw)
        if magic!=b'SLA1' or len(raw)!=HEAD.size+NODE.size*(nc+nl):
            raise ValueError('file size/version')
        nodes=list(NODE.iter_unpack(raw[HEAD.size:]))
        obj=cls(ni,no,nodes[:nc],nodes[nc:],root); obj.validate(); return obj

    def coefficients(self, inputs: list[int], lanes: int=1) -> list[int]:
        if len(inputs)!=self.ninputs or lanes<1: raise ValueError('input shape')
        mask=(1<<lanes)-1
        if any(x<0 or x>mask for x in inputs): raise ValueError('input lane bits')
        v=[]
        for op,a,b,_ in self.coeff:
            if op==0: v.append(mask if a else 0)
            elif op==1: v.append(inputs[a])
            elif op==2: v.append(v[a]^v[b])
            else: v.append(v[a]&v[b])
        return v

    def extract(self, inputs: list[int], lanes: int=1) -> list[int]:
        """One coefficient sweep and one reverse selector sweep; all work paid."""
        cv=self.coefficients(inputs,lanes)
        bars=[0]*len(self.linear); bars[self.root]=(1<<lanes)-1
        out=[0]*self.noutputs
        # No input-dependent gate-skipping claim: scan every stored linear node.
        for i in range(len(self.linear)-1,-1,-1):
            op,a,b,_=self.linear[i]; z=bars[i]
            if op==1: out[a]^=z
            elif op==2: bars[a]^=z; bars[b]^=z
            elif op==3: bars[b]^=z&cv[a]
        return out

    def scalar(self, inputs: list[int], selectors: list[int], lanes: int=1) -> int:
        """Independent forward evaluation of the represented scalar."""
        if len(selectors)!=self.noutputs: raise ValueError('selector shape')
        mask=(1<<lanes)-1
        if any(x<0 or x>mask for x in selectors): raise ValueError('selector bits')
        cv=self.coefficients(inputs,lanes); lv=[]
        for op,a,b,_ in self.linear:
            if op==0: lv.append(0)
            elif op==1: lv.append(selectors[a])
            elif op==2: lv.append(lv[a]^lv[b])
            else: lv.append(cv[a]&lv[b])
        return lv[self.root]

    def costs(self) -> dict:
        cg=sum(n[0] in (2,3) for n in self.coeff)
        lx=sum(n[0]==2 for n in self.linear)
        ls=sum(n[0]==3 for n in self.linear)
        lb=sum(n[0]==1 for n in self.linear)
        return dict(coefficient_nodes=len(self.coeff),coefficient_bit_gates=cg,
                    selector_nodes=len(self.linear),linear_xor_nodes=lx,scale_nodes=ls,
                    selector_leaves=lb,scalar_bit_gates=cg+lx+ls,
                    extraction_bit_operations=cg+2*lx+2*ls+lb,
                    serialized_bytes=len(self.to_bytes()),
                    node_field_bytes_scanned=NODE.size*(len(self.coeff)+len(self.linear)),
                    logical_live_bit_slots=len(self.coeff)+len(self.linear)+self.noutputs,
                    note='Logical bit gates/serialized fields, not GPU traffic, word operations or latency. Python objects and lane packing extra.')

class Builder:
    def __init__(self,ninputs:int,noutputs:int):
        self.ninputs=ninputs;self.noutputs=noutputs
        self.c=[];self.l=[];self.cm={};self.lm={}
        self.zero=self._c(0,0,0);self.one=self._c(0,1,0)
        self.lzero=self._l(0,0,0)
    def _c(self,op,a,b):
        n=(op,a,b,0)
        if n not in self.cm: self.cm[n]=len(self.c);self.c.append(n)
        return self.cm[n]
    def _l(self,op,a,b):
        n=(op,a,b,0)
        if n not in self.lm: self.lm[n]=len(self.l);self.l.append(n)
        return self.lm[n]
    def input(self,i):
        if not 0<=i<self.ninputs: raise ValueError('input')
        return self._c(1,i,0)
    def xor(self,a,b):
        if a==b:return self.zero
        if a==self.zero:return b
        if b==self.zero:return a
        return self._c(2,*sorted((a,b)))
    def and_(self,a,b):
        if a==self.zero or b==self.zero:return self.zero
        if a==b:return a
        if a==self.one:return b
        if b==self.one:return a
        return self._c(3,*sorted((a,b)))
    def not_(self,a):return self.xor(a,self.one)
    def or_(self,a,b):return self.xor(self.xor(a,b),self.and_(a,b))
    def mux(self,s,t,f):return self.xor(f,self.and_(s,self.xor(t,f)))
    def add_bits(self,a,b):
        if len(a)!=len(b):raise ValueError('width')
        carry=self.zero;out=[]
        for x,y in zip(a,b):
            z=self.xor(x,y);out.append(self.xor(z,carry))
            carry=self.xor(self.and_(x,y),self.and_(carry,z))
        return out
    def basis(self,i):
        if not 0<=i<self.noutputs:raise ValueError('selector')
        return self._l(1,i,0)
    def lx(self,a,b):
        if a==b:return self.lzero
        if a==self.lzero:return b
        if b==self.lzero:return a
        return self._l(2,*sorted((a,b)))
    def scale(self,c,l):
        if c==self.zero or l==self.lzero:return self.lzero
        if c==self.one:return l
        return self._l(3,c,l)
    def wrap(self, roots:list[int]) -> Program:
        if len(roots)!=self.noutputs:raise ValueError('root count')
        root=self.lzero
        for i,c in enumerate(roots):root=self.lx(root,self.scale(c,self.basis(i)))
        return self.finish(root)
    def finish(self, root:int) -> Program:
        p=Program(self.ninputs,self.noutputs,self.c.copy(),self.l.copy(),root);p.validate();return p

def native_rounder() -> tuple[Program,list[int]]:
    """Build FP32-word -> BF16 RNE bits plus unchanged 32-bit successor word.
    NaN is explicitly canonicalized to +0x7fc0. All other raw words supported.
    No response enumeration, calls to a floating-point oracle, or model weights.
    """
    g=Builder(32,48);x=[g.input(i) for i in range(32)]
    bias=[g.not_(x[16])]*15+[x[16]]+[g.zero]*16
    rounded=g.add_bits(x,bias)[16:32]
    exponent_all=g.one
    for j in range(23,31):exponent_all=g.and_(exponent_all,x[j])
    frac_any=g.zero
    for j in range(23):frac_any=g.or_(frac_any,x[j])
    nan=g.and_(exponent_all,frac_any)
    bits=[g.mux(nan,g.one if (0x7fc0>>i)&1 else g.zero,rounded[i]) for i in range(16)]
    roots=bits+x
    return g.wrap(roots),roots

def transpose_binary_matrix(w:list[list[int]]) -> Program:
    """Phi = XOR_j x_j (XOR_i W_ij z_i). W lives in topology, not free."""
    if not w or not w[0] or any(len(row)!=len(w[0]) for row in w): raise ValueError('shape')
    if any(v not in (0,1) for row in w for v in row):raise ValueError('GF2 only')
    m,n=len(w),len(w[0]);g=Builder(n,m);sels=[g.basis(i) for i in range(m)];root=g.lzero
    for j in range(n):
        c=g.lzero
        for i in range(m):
            if w[i][j]:c=g.lx(c,sels[i])
        root=g.lx(root,g.scale(g.input(j),c))
    return g.finish(root)
