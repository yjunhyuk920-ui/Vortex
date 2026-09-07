"""Explicit small BF16/FP32 balanced-tree reference. Not a CUDA/HF implementation."""
from __future__ import annotations
import struct
import numpy as np

def bf_bits(v):
    a=np.asarray(v,dtype=np.float32)
    u=a.view(np.uint32)
    with np.errstate(over='ignore',invalid='ignore'):
        r=(u + np.uint32(0x7fff) + ((u >> 16)&1))>>16
    # Preserve NaN rather than accidentally turn a small payload into infinity.
    nan=((u&0x7f800000)==0x7f800000)&((u&0x007fffff)!=0)
    r=np.where(nan,(u>>16)|0x40,r)
    return np.asarray(r,dtype=np.uint16)

def value(b):
    a=np.asarray(b,dtype=np.uint16)
    return (a.astype(np.uint32)<<16).view(np.float32)

def linear(x,w):
    n=w.shape[1]
    if n<1 or n&(n-1): raise ValueError('power-of-two input width required')
    p=np.multiply(x[:,None,:],w[None,:,:],dtype=np.float32)
    while p.shape[-1]>1:
        p=np.add(p[...,::2],p[...,1::2],dtype=np.float32)
    return bf_bits(p[...,0])

def model(xbits,G,U,D,lut):
    x=value(xbits)
    g=linear(x,value(G)); u=linear(x,value(U))
    z=bf_bits(np.multiply(value(lut[g]),value(u),dtype=np.float32))
    return linear(value(z),value(D)),g

def f32(v): return struct.unpack('<f',struct.pack('<f',float(v)))[0]
def scalar_bf(v):
    u=struct.unpack('<I',struct.pack('<f',float(v)))[0]
    if u&0x7f800000==0x7f800000 and u&0x7fffff: return (u>>16)|0x40
    return ((u+0x7fff+((u>>16)&1))&0xffffffff)>>16

def scalar_value(b): return struct.unpack('<f',struct.pack('<I',int(b)<<16))[0]
def scalar_linear(xbits,W):
    out=[]
    for row in W:
        p=[f32(scalar_value(x)*scalar_value(w)) for x,w in zip(xbits,row)]
        while len(p)>1: p=[f32(p[j]+p[j+1]) for j in range(0,len(p),2)]
        out.append(scalar_bf(p[0]))
    return out

def scalar_model(xbits,G,U,D,lut):
    g=scalar_linear(xbits,G); u=scalar_linear(xbits,U)
    z=[scalar_bf(f32(scalar_value(lut[a])*scalar_value(b))) for a,b in zip(g,u)]
    return scalar_linear(z,D)

def alphabet(q):
    vals={2:[-1.,1.],4:[-1.,-.5,.5,1.],8:[-2.,-1.,-.5,-.25,.25,.5,1.,2.],
          16:[-4.,-3.,-2.,-1.,-.5,-.25,-0.,0.,.25,.5,1.,2.,3.,4.,6.,8.]}[q]
    return bf_bits(vals)

def inputs(alpha,n=4):
    q=len(alpha); ids=np.arange(q**n,dtype=np.uint32); cols=[]
    for i in range(n): cols.append(alpha[(ids//(q**(n-1-i)))%q])
    return np.stack(cols,axis=1)
