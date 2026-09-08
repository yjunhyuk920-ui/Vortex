"""Declared native projection witness; never a whole HF/CUDA executor."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def word(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def balanced(values):
    values=list(values)
    padded=1 << (len(values)-1).bit_length()
    values.extend([0.0]*(padded-len(values)))
    while len(values)>1:
        values=[f32(values[i]+values[i+1]) for i in range(0,len(values),2)]
    return values[0]


def delta_for(n):
    if not 2<=n<=16384:
        raise ValueError('registered width must be in [2,16384]')
    return math.ldexp(1.0,-24-(n-1).bit_length()-8-2)


def full_reference(x):
    return [balanced(f32((2.0 if i==j else 1.0)*v) for j,v in enumerate(x)) for i in range(len(x))]


def pruned_positive_tree(n,positive_leaves):
    # Domain is positive finite leaves or +0, not arbitrary signed/NaN operands.
    if any(j<0 or j>=n or not math.isfinite(v) or v<0 or (v==0 and word(v)!=0) for j,v in positive_leaves.items()):
        raise ValueError('positive finite leaf domain required')
    current={j:v for j,v in positive_leaves.items() if v!=0.0}
    for _ in range((n-1).bit_length()):
        parents={j//2 for j in current}
        current={j:f32(current.get(2*j,0.0)+current.get(2*j+1,0.0)) for j in parents}
    return current.get(0,0.0)


def sparse_reference(n,nonzero_inputs):
    return [pruned_positive_tree(n,{j:f32((2.0 if i==j else 1.0)*v) for j,v in nonzero_inputs.items()}) for i in range(n)]


def exact_ledger(n):
    exponent=-24-(n-1).bit_length()-8-2
    delta=Fraction(2)**exponent
    coefficient_bound=255*n
    # Tiny-only sums remain exactly representable FP32 integers times delta.
    assert coefficient_bound<2**24
    residual_upper=coefficient_bound*delta
    half_ulp_at_one=Fraction(1,2**24)
    assert residual_upper<half_ulp_at_one
    # All a*delta for a<=255 need <=8 significant binary digits, hence BF16.
    assert exponent>=-126
    return {'n':n,'delta_exponent':exponent,'max_tiny_integer_coefficient':coefficient_bound,
            'tiny_sum_bound':str(residual_upper),'half_ulp_at_one':str(half_ulp_at_one),
            'bound_ratio_to_half_ulp':str(residual_upper/half_ulp_at_one),
            'real_determinant':n+1,'fiber_log2_cardinality':8*(n-1),
            'reversible_side_state_information_lower_bits_only':8*(n-1),
            'native_reference_products_per_vector':n*n,
            'native_reference_padded_additions_per_vector':n*((1<<(n-1).bit_length())-1),
            'full_matrix_BF16_payload_bytes':2*n*n,
            'family_compiler_coefficient_writes':n*n,
            'is_universal_dense_executor':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('refuse to overwrite existing evidence')
    args.output_dir.mkdir(parents=True)
    before=random.getstate()
    ledgers=[exact_ledger(n) for n in (2,3,16,16384)]
    bf16_values_checked=0
    for n in (2,3,16,16384):
        for a in range(256):
            assert word(a*delta_for(n)) & 0xffff == 0
            bf16_values_checked+=1
    assert list(map(word,full_reference([1.0,0.0])))==[0x40000000,0x3f800000]
    assert list(map(word,full_reference([1.0,1.0])))==[0x40400000,0x40400000]
    count=0
    digest=hashlib.sha256()
    for a,b in itertools.product(range(256),repeat=2):
        x=[1.0,a*delta_for(3),b*delta_for(3)]
        output=list(map(word,full_reference(x)))
        assert output==[0x40000000,0x3f800000,0x3f800000]
        digest.update(struct.pack('<IIIII',a,b,*output))
        count+=1
    comparisons=[]
    for n in (2,3,16,64):
        for a in (0,1,127,255):
            x=[1.0]+[0.0]*(n-1)
            x[-1]=a*delta_for(n)
            full=list(map(word,full_reference(x)))
            sparse=list(map(word,sparse_reference(n,{j:v for j,v in enumerate(x) if v})))
            assert sparse==full==[0x40000000]+[0x3f800000]*(n-1)
            comparisons.append({'n':n,'a':a,'words_checked':n,'mismatches':0})
    large=[]
    for a in (0,255):
        n=16384
        x={0:1.0,1:a*delta_for(n)}
        output=list(map(word,sparse_reference(n,x)))
        assert output==[0x40000000]+[0x3f800000]*(n-1)
        large.append({'n':n,'nonzero_input_pattern':a,'output_words':len(output),
                      'sha256_little_endian_FP32_output_words':hashlib.sha256(struct.pack('<'+'I'*n,*output)).hexdigest(),
                      'execution':'proved positive-zero pruned tree, not a full dense/hardware run'})
    assert random.getstate()==before
    result={'kind':'SCOPED_NATIVE_FULL_RANK_NONINJECTIVITY_WITNESS',
            'abi':'BF16 operands, separate FP32 RNE products, fixed padded balanced FP32 RNE additions, FP32 output',
            'exact_cost_and_proof_ledgers':ledgers,'BF16_values_checked':bf16_values_checked,
            'exhaustive_n3_vectors':count,'exhaustive_n3_output_words':3*count,
            'exhaustive_record_sha256':digest.hexdigest(),'exhaustive_mismatches':0,
            'pruned_vs_full_controls':comparisons,'large_pruned_controls':large,
            'python_rng_state_unchanged':True,'native_HF_RNG_KV_tested':False,
            'witness_HF_activation_reachability_proved':False,'target_hardware_tested':False,
            'new_qualifying_principles':0,'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED',
            'CORE_ADMISSION':False,'O1':'OPEN','O2':'OPEN','O3':'OPEN','O4':'OPEN','O5':'OPEN','O6':'PARTIAL'}
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    (args.output_dir/'summary.json').write_bytes(raw)
    (args.output_dir/'checksums.sha256').write_bytes((hashlib.sha256(raw).hexdigest()+'  summary.json\n').encode())
    print(json.dumps({'vectors':count,'mismatches':0,'large_pruned_vectors':2,'summary_sha256':hashlib.sha256(raw).hexdigest()}))


if __name__=='__main__':
    main()
