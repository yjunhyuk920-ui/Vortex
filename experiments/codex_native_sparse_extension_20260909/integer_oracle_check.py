"""Independent integer-dyadic RNE oracle; no host-float oracle arithmetic."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct

from scalar_producer import compile_store, project_fp32, project, cost_formula

QNAN=0x7fc00000


def parts(word):
    sign=-1 if word>>31 else 1
    field=(word>>23)&255
    fraction=word&0x7fffff
    if field==255:
        return sign, 'nan' if fraction else 'inf', 0
    return sign*((1<<23)|fraction) if field else sign*fraction, (field-150) if field else -149, 0


def round_word(integer, exponent, zero_sign=0):
    if integer==0:
        return zero_sign<<31
    sign=0x80000000 if integer<0 else 0
    value=abs(integer)
    leading=value.bit_length()-1+exponent
    quantum=max(-149,leading-23)
    shift=quantum-exponent
    if shift>0:
        rounded,remainder=divmod(value,1<<shift)
        half=1<<(shift-1)
        rounded+=int(remainder>half or (remainder==half and rounded&1))
    else:
        rounded=value<<(-shift)
    if rounded==0:
        return sign
    if rounded.bit_length()>24:
        assert rounded==(1<<24)
        rounded>>=1
        quantum+=1
    leading=rounded.bit_length()-1+quantum
    if leading>127:
        return sign|0x7f800000
    if leading < -126:
        assert quantum==-149
        return sign|rounded
    assert rounded.bit_length()==24
    return sign|((leading+127)<<23)|(rounded-(1<<23))


def multiply_bf16(left,right):
    a,e,_=parts(left<<16)
    b,f,_=parts(right<<16)
    assert isinstance(e,int) and isinstance(f,int)
    return round_word(a*b,e+f,((left^right)>>15)&1)


def add_word(left,right):
    a,e,_=parts(left)
    b,f,_=parts(right)
    if e=='nan' or f=='nan':
        return QNAN
    if e=='inf' or f=='inf':
        if e==f=='inf' and a!=b:
            return QNAN
        return left if e=='inf' else right
    common=min(e,f)
    integer=(a<<(e-common))+(b<<(f-common))
    negative_zero=int(left==right==0x80000000)
    return round_word(integer,common,negative_zero)


def full_integer(matrix,inputs):
    width=1<<(len(inputs)-1).bit_length()
    output=[]
    for row in matrix:
        leaves=[multiply_bf16(a,b) for a,b in zip(row,inputs)]+[0]*(width-len(inputs))
        while len(leaves)>1:
            leaves=[add_word(leaves[i],leaves[i+1]) for i in range(0,len(leaves),2)]
        output.append(leaves[0])
    return output


def bf16_store(word):
    if word&0x7f800000==0x7f800000 and word&0x7fffff:
        return 0x7fc0
    return ((word+0x7fff+((word>>16)&1))>>16)&0xffff


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    # Known boundaries prevent both implementations silently agreeing on a
    # reversed RNE tie, missing signed zero, overflow, or NaN rule.
    assert round_word(1,-150)==0
    assert round_word(-1,-150)==0x80000000
    assert round_word(3,-150)==2
    assert round_word((1<<25)-1,103)==0x7f800000
    assert add_word(0x7f800000,0xff800000)==QNAN
    assert add_word(0x80000000,0x80000000)==0x80000000
    assert add_word(0x3f800000,0xbf800000)==0
    source=[0,0x8000,1,0x8001,0x80,0x8080,0x3f7f,0x3f80,0xbf80,0x4000,0x7f7f,0xff7f]
    pairs=[(1,0x8001),(0x3f80,0xbf80),(0x4000,0x4000),(0x4000,0xc000),
           (0x7f7f,0xff7f),(0x3f00,0xbf00)]
    checked=0
    nan_roots=negative_zeros=underflow_stores=0
    first_failure=None
    def check(matrix,x,signs):
        nonlocal checked,nan_roots,negative_zeros,underflow_stores,first_failure
        store=compile_store(matrix,signs)
        result,stats=project_fp32(store,x)
        actual=[struct.unpack('>I',struct.pack('>f',value))[0] for value in result]
        expected=full_integer(matrix,x)
        stored,_=project(store,x)
        checked+=1
        nan_roots+=sum(word==QNAN for word in expected)
        negative_zeros+=sum(word==0x80000000 for word in expected)
        underflow_stores+=sum((word&0x7fffffff)!=0 and bf16_store(word)==0x8000 for word in expected)
        if actual!=expected or stored!=[bf16_store(word) for word in expected]:
            first_failure={'matrix':matrix,'input':x,'signs':signs,'actual':actual,'expected':expected}
            raise AssertionError('integer oracle mismatch')
        assert stats['actual_products']==len(matrix)*sum(bool(w&0x7fff) for w in x)
        assert stats['actual_additions']<=len(matrix)
    try:
        for n in [2,3,4,5,8]:
            for i,j in itertools.combinations(range(n),2):
                for signs in [[0]*n,[c%2 for c in range(n)],[1]*n]:
                    for w1,w2 in itertools.product(source,repeat=2):
                        matrix=[[0x8000]*n,[0x3f80]*n]
                        matrix[0][i],matrix[0][j]=w1,w2
                        matrix[1][i],matrix[1][j]=w2,w1
                        for x1,x2 in pairs:
                            x=[s<<15 for s in signs]
                            x[i],x[j]=x1,x2
                            check(matrix,x,signs)
        # Empty support, unpadded all-negative-zero root; padded root is +0.
        for n in [1,2,3,4,5,8]:
            check([[0xbf80]*n],[0]*n,[0]*n)
        # A nonzero negative FP32 product may round to BF16 -0 after repair.
        check([[0x8001]],[0x3f00],[0])
    finally:
        result={'cases':checked,'nan_roots':nan_roots,'fp32_negative_zero_roots':negative_zeros,
            'nonzero_fp32_to_negative_bf16_zero':underflow_stores,'first_failure':first_failure,
            'oracle':'integer significands, exact shifts/sums/products, explicit ties-to-even',
            'scope':'frozen adversarial two-support boundary grid; not exhaustive operand pairs or native HF',
            'cost_m16384_n16384_k2':cost_formula(k=2)}
        payload=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
        (args.output_dir/'summary.json').write_bytes(payload)
        (args.output_dir/'checksums.sha256').write_bytes((hashlib.sha256(payload).hexdigest()+'  summary.json\n').encode())
        print(json.dumps(result))


if __name__=='__main__':
    main()
