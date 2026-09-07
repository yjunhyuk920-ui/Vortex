"""Exact bounded-integer source + interval/residue decoder (not a HF executor).

No original matrix is retained by Program. The serialized source is a lossless
signed-bitplane representation of a residual matrix, not an oracle for residues.
Cost counters are logical fields/operations, NOT measured hardware traffic/time.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import struct
from numbers import Integral
from typing import Sequence

MAGIC = b'VXRES001'
HEADER = struct.Struct('<8sHH')
ROW = struct.Struct('<hBBI')  # base, signed bit width, reserved, L1 radius for |x|<=1


def signed_width(values: Sequence[int]) -> int:
    if not values or all(v == 0 for v in values):
        return 0
    lo, hi = min(values), max(values)
    b = 1
    while not (-(1 << (b-1)) <= lo and hi < (1 << (b-1))):
        b += 1
    return b


def lift_interval(residue: int, modulus: int, low: int, high: int) -> int:
    """The unique integer in [low,high] with the stated residue, or reject.

    Correctness relies on the source proving the true value lies in this range.
    This function CANNOT certify a falsely supplied interval or residue.
    """
    if modulus <= 0 or low > high or high-low >= modulus:
        raise ValueError('invalid/ambiguous interval')
    y = low + ((int(residue)-low) % modulus)
    if y > high:
        raise ValueError('no representative in interval')
    return y


def compile_matrix(weights: Sequence[Sequence[int]]) -> tuple[bytes, dict]:
    if any(not isinstance(v, Integral) for r in weights for v in r):
        raise ValueError("non-integer coefficients are not silently converted")
    rows = [[int(v) for v in r] for r in weights]
    if not rows or not rows[0] or len(rows)>65535 or len(rows[0])>65535:
        raise ValueError('invalid dimensions')
    m,n=len(rows),len(rows[0])
    if n & (n-1):
        raise ValueError('balanced-tree ABI requires power-of-two width')
    packed=(n+7)//8
    out=bytearray(HEADER.pack(MAGIC,m,n))
    detail=[]
    for row in rows:
        if len(row)!=n or any(abs(v)>127 for v in row):
            raise ValueError('integer BF16 reference domain exceeded')
        if not any(v>0 for v in row):
            raise ValueError('positive coefficient required by declared zero-sign ABI')
        if sum(abs(v) for v in row) >= (1<<24):
            raise ValueError('FP32 exact intermediate bound not certified')
        base=sorted(row)[(n-1)//2]
        residual=[v-base for v in row]
        radius=sum(abs(v) for v in residual)
        b=signed_width(residual)
        out.extend(ROW.pack(base,b,0,radius))
        for bit in range(b):
            plane=sum(((v>>bit)&1)<<j for j,v in enumerate(residual))
            out.extend(plane.to_bytes(packed,'little'))
        k=(2*radius).bit_length()  # 2**k > 2*radius, including k=0 for radius=0
        detail.append({'base':base,'radius':radius,'modulus_bits':k,'stored_signed_bits':b,
                       'nonzero_residual_coefficients':sum(v!=0 for v in residual),
                       'max_residual_abs':max(map(abs,residual)),
                       'residue_coefficient_lift_recovers_every_residual':all(
                           lift_interval(v%(1<<k),1<<k,-radius,radius)==v for v in residual)})
    return bytes(out), {'rows':detail,'original_bf16_bytes':2*m*n,
        'signed_bitpacked_original_payload':sum(signed_width(r)*packed for r in rows),
        'source_bytes':len(out),'source_payload':sum(d['stored_signed_bits']*packed for d in detail),
        'preparation_original_read_bytes':2*m*n,'preparation_source_write_bytes':len(out),
        'preparation_work':'sort each row O(n log n); emit every residual coefficient bit; read all W',
        'work_memory':'this Python constructor receives all W and stores emitted blob; not a streaming 405B loader'}

@dataclass(frozen=True)
class Program:
    blob: bytes

    def __post_init__(self):
        if len(self.blob)<HEADER.size:
            raise ValueError('truncated header')
        magic,m,n=HEADER.unpack_from(self.blob)
        if magic!=MAGIC or m==0 or n==0 or n&(n-1):
            raise ValueError('bad format')
        p=HEADER.size
        for _ in range(m):
            if p+ROW.size>len(self.blob):raise ValueError('truncated row')
            base,b,reserved,radius=ROW.unpack_from(self.blob,p);p+=ROW.size
            if reserved or b>16:raise ValueError('bad row')
            p+=b*((n+7)//8)
        if p!=len(self.blob):raise ValueError('trailing/truncated planes')

    def run(self,x:Sequence[int])->tuple[list[int],dict]:
        _,m,n=HEADER.unpack_from(self.blob)
        if len(x)!=n or any(not isinstance(v,int) or v not in (-1,0,1) for v in x):
            raise ValueError('input outside declared ternary integer domain')
        pos=sum(1<<j for j,v in enumerate(x) if v>0)
        neg=sum(1<<j for j,v in enumerate(x) if v<0)
        sx=sum(x); p=HEADER.size;packed=(n+7)//8
        out=[];events=[];popcounts=0;plane_ops=0
        for i in range(m):
            base,b,_,radius=ROW.unpack_from(self.blob,p);p+=ROW.size
            k=(2*radius).bit_length(); modulus=1<<k
            rem=0
            for bit in range(b):
                plane=int.from_bytes(self.blob[p:p+packed],'little');p+=packed
                selected=(plane&pos).bit_count()-(plane&neg).bit_count()
                coeff=-(1<<bit) if bit==b-1 else 1<<bit
                rem=(rem+coeff*selected)%modulus
                popcounts+=2;plane_ops+=1
            e=lift_interval(rem,modulus,-radius,radius)
            prediction=base*sx
            out.append(prediction+e)
            events.append({'row':i,'prediction':prediction,'low':-radius,'high':radius,
                           'modulus_bits':k,'residue':rem,'lifted_residual':e})
        return out,{'events':events,'serialized_fields_read':p,'input_values':n,
                    'output_values':m,'python_bigint_popcount_calls':popcounts,
                    'plane_accumulate_calls':plane_ops,
                    'popcount_word64_work_lower':popcounts*math.ceil(n/64),
                    'workspace_bitmaps_bytes_min':2*math.ceil(n/8),
                    'not_charged_in_field_bytes':'Python objects, int arithmetic, copying, allocator, caches; timings not measured'}


def bf16_bits_int(v:int)->int:
    """Independent exact integer -> BF16 RNE, including sign for nonzero values."""
    if v==0:return 0
    sign=0x8000 if v<0 else 0;a=abs(v)
    e=a.bit_length()-1
    if e<=7:
        mant=a<<(7-e)
    else:
        s=e-7; mant,rem=divmod(a,1<<s)
        half=1<<(s-1)
        if rem>half or (rem==half and mant&1):mant+=1
        if mant==256:mant=128;e+=1
    if e+127>=255:return sign|0x7f80
    return sign|((e+127)<<7)|(mant-128)


def generic_scale(width:int,residual_abs:int=1)->dict:
    radius=width*residual_abs;k=(2*radius).bit_length()
    # Packed dense residue residues, without recognizing redundant sign planes.
    return {'width':width,'radius':radius,'required_residue_bits':k,
            'naive_residue_payload_vs_bf16':k/16,
            'ternary_signed_payload_vs_bf16':2/16 if residual_abs==1 else None,
            'scope':'row has width many +/-residual_abs coefficients; interval L1 source and no special compression; not universal lower bound'}
