"""Exact GF(2) response factorization with original pivot rows; no approximation."""
from __future__ import annotations
import struct, hashlib
from dataclasses import dataclass
import numpy as np
MAGIC=b'VXCODE01'
HEADER=struct.Struct('<8s8I')

def factor(rows):
    """Return original-row basis indices, exact codes and echelon pivot witnesses."""
    basis={}; selected=[]; codes=[]
    for i,raw in enumerate(rows):
        v=int.from_bytes(raw,'little'); c=0
        while v:
            p=v.bit_length()-1
            if p not in basis:
                j=len(selected); selected.append(i)
                basis[p]=(v,c^(1<<j)); c=1<<j
                break
            b,code=basis[p]; v^=b; c^=code
        codes.append(c)
    return selected,codes,sorted(basis,reverse=True)

def encode(table,alpha,n=4):
    q=len(alpha); split=n//2; L=q**split; R=q**(n-split)
    a=np.asarray(table,dtype='<u2').reshape(L,R,-1); m=a.shape[-1]
    sel,codes,piv=factor([row.tobytes() for row in a]); r=len(sel); cb=(r+7)//8
    cdata=b''.join(c.to_bytes(cb,'little') for c in codes)
    atom=(a[sel].transpose(1,0,2).copy() if r else np.zeros((R,0,m),dtype='<u2'))
    body=HEADER.pack(MAGIC,n,q,m,L,R,r,cb,split)+np.asarray(alpha,dtype='<u2').tobytes()+cdata+atom.tobytes()
    blob=body+hashlib.sha256(body).digest()
    return blob,{'basis_rows':sel,'pivot_bits':piv,'rank':r,'codes':codes,'header_bytes':HEADER.size,
                 'left_code_bytes':len(cdata),'atom_bytes':atom.nbytes,'file_bytes':len(blob)}

@dataclass
class Program:
    n:int; q:int; m:int; L:int; R:int; r:int; cb:int; split:int
    alpha:np.ndarray; codes:list[int]; atoms:np.ndarray; serialized_bytes:int
    @classmethod
    def load(cls,blob):
        if len(blob)<HEADER.size+32: raise ValueError('truncated')
        if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]: raise ValueError('checksum')
        magic,n,q,m,L,R,r,cb,split=HEADER.unpack_from(blob)
        if magic!=MAGIC or not(0<n<=16 and 1<q<=256 and 0<m<=4096 and 0<split<n): raise ValueError('header')
        if L!=q**split or R!=q**(n-split) or r>L or cb!=(r+7)//8: raise ValueError('shape')
        size=HEADER.size+2*q+L*cb+2*R*r*m+32
        if len(blob)!=size: raise ValueError('size')
        off=HEADER.size; alpha=np.frombuffer(blob,dtype='<u2',count=q,offset=off).copy(); off+=2*q
        if len(set(map(int,alpha)))!=q: raise ValueError('duplicate alphabet')
        codes=[int.from_bytes(blob[off+i*cb:off+(i+1)*cb],'little') for i in range(L)]; off+=L*cb
        if any(c>>r for c in codes): raise ValueError('padding')
        atoms=np.frombuffer(blob,dtype='<u2',count=R*r*m,offset=off).reshape(R,r,m).copy()
        return cls(n,q,m,L,R,r,cb,split,alpha,codes,atoms,len(blob))
    def query(self,xbits):
        if len(xbits)!=self.n: raise ValueError('dimension')
        digits=[]; comparisons=0
        for x in xbits:
            found=None
            for i,a in enumerate(self.alpha):
                comparisons+=1
                if int(x)==int(a): found=i; break
            if found is None: raise ValueError('unsupported input bit pattern')
            digits.append(found)
        left=right=0
        for d in digits[:self.split]: left=left*self.q+d
        for d in digits[self.split:]: right=right*self.q+d
        c=self.codes[left]; ans=np.zeros(self.m,dtype=np.uint16); k=c.bit_count()
        while c:
            bit=c&-c; j=bit.bit_length()-1; ans^=self.atoms[right,j]; c^=bit
        # Logical references, not DRAM or latency; output update = read+write each XOR.
        cost={'atoms':k,'domain_comparisons':comparisons,'payload_bytes':self.cb+2*self.m*k,
              'query_reference_bytes':HEADER.size+2*self.n+2*comparisons+self.cb+2*self.m*k+4*self.m*k+2*self.m,
              'xor_word_ops':self.m*k}
        return ans,cost

def mobius(table):
    a=np.array(table,dtype=np.uint16,copy=True); N=len(a)
    if N&(N-1): raise ValueError('power-of-two domain required')
    step=1
    while step<N:
        b=a.reshape(-1,2*step,a.shape[1]); b[:,step:,:]^=b[:,:step,:]; step*=2
    return a

def pulse_query(coeff,address):
    out=np.zeros(coeff.shape[1],np.uint16); s=address; reads=0
    while True:
        out^=coeff[s]; reads+=1
        if not s: break
        s=(s-1)&address
    return out,reads

def runs(table):
    t=np.asarray(table,dtype=np.uint16)
    starts=np.r_[0,np.flatnonzero(np.any(t[1:]!=t[:-1],axis=1))+1]
    return starts.astype(np.uint32),t[starts].copy()

def run_query(starts,values,address):
    lo=0; hi=len(starts); comps=0
    while lo<hi:
        mid=(lo+hi)//2; comps+=1
        if int(starts[mid])<=address: lo=mid+1
        else: hi=mid
    if lo==0: raise ValueError('address')
    return values[lo-1].copy(),comps
