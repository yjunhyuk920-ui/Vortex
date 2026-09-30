"""Exhaustive tiny-domain and deterministic FP32 witnesses, no model execution."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from reference import BF16,FP32,Format,pow2
from validate import EnumeratedReference
from constructor import Addend,NativeCharts,compile_addends,compile_fma_operands,key_word,word_key


def add_stats(target, stats):
    for key,value in stats.items():
        if key.startswith('max_') or key=='peak_charts':
            target[key]=max(target.get(key,0),value)
        else:
            target[key]=target.get(key,0)+value


def run():
    start=time.monotonic()
    fmt=Format(4,3);ref=EnumeratedReference(fmt)
    initial=[key_word(fmt,k) for k in range(2*fmt.inf+2)]
    alphabet=(Addend(-17),Addend(-1),Addend(Q(-1,32)),Addend(Q(-1,64)),
              Addend(0,1),Addend(0,0),Addend(Q(1,64)),Addend(Q(1,32)),Addend(1),Addend(17))
    counts=dict(tiny_sequences=0,tiny_prefix_word_checks=0,tiny_identity_checks=0,
                native_sequences=0,native_prefix_word_checks=0,fma_wrapper_checks=0,
                rejected_nan_checks=0,rejected_nonfinite_product_checks=0,mismatches=0)
    counts.update(rejected_invalid_word_checks=0,rejected_nondyadic_checks=0)
    tiny_totals={};tiny_peaks=[]
    identity=NativeCharts(fmt);identity.check_invariants()
    for word in initial:
        assert key_word(fmt,word_key(fmt,word))==word
        assert identity.query(word)==word
        counts['tiny_identity_checks']+=1
    for sequence in product(alphabet,repeat=3):
        chart=NativeCharts(fmt);chart.check_invariants()
        expected=dict((word,word) for word in initial)
        for addend in sequence:
            chart.append(addend)
            for word in initial:
                expected[word]=ref.step(expected[word],addend.value,addend.zero_sign)
                actual=chart.query(word)
                assert actual==expected[word],('tiny',word,sequence,addend,actual,expected[word])
                counts['tiny_prefix_word_checks']+=1
        counts['tiny_sequences']+=1
        add_stats(tiny_totals,chart.stats)
        tiny_peaks.append(chart.stats['peak_charts'])
    # Deterministic full-width word population, never enumerate FP32.
    native_initial={0,FP32.sign_bit,FP32.inf,FP32.sign_bit|FP32.inf,
                    1,2,FP32.sign_bit|1,FP32.sign_bit|2,
                    0x007fffff,0x807fffff,0x00800000,0x80800000,
                    FP32.maxfinite,FP32.sign_bit|FP32.maxfinite}
    for e in (-126,-125,-75,-1,0,1,64,126,127):
        base=FP32.round(pow2(e))
        for magnitude in (base-1,base,base+1):
            if 0<=magnitude<FP32.inf:
                native_initial.add(magnitude);native_initial.add(FP32.sign_bit|magnitude)
    streams=(
        (),
        (Addend(0,0),Addend(0,1),Addend(0,0),Addend(0,1)),
        (Addend(pow2(-150)),Addend(-pow2(-150)),Addend(pow2(-149))),
        (Addend(pow2(-126)),Addend(-pow2(-126)),Addend(0,1)),
        (Addend(pow2(-24)),Addend(pow2(-24)),Addend(-pow2(-23)),Addend(1)),
        (Addend(1),Addend(-1),Addend(-1),Addend(1)),
        (Addend(pow2(128)),Addend(-pow2(127)),Addend(-pow2(128))),
        (Addend(-pow2(128)),Addend(pow2(127)),Addend(pow2(128))),
        tuple(Addend(v) for v in (1,-2,4,-8,16,-32,64,-128)),
        tuple(Addend(v) for v in (-1,2,-4,8,-16,32,-64,128)),
        (Addend(pow2(-266)),Addend(-pow2(-266)),Addend(pow2(255)),Addend(-pow2(255))),
    )
    native_totals={};native_records=[]
    for sequence in streams:
        chart=NativeCharts(FP32);chart.check_invariants()
        expected=dict((word,word) for word in native_initial)
        for word in native_initial:
            assert chart.query(word)==word
            counts['native_prefix_word_checks']+=1
        for addend in sequence:
            chart.append(addend)
            for word in native_initial:
                expected[word]=FP32.add_exact(expected[word],addend.value,addend.zero_sign)
                actual=chart.query(word)
                assert actual==expected[word],('fp32',f'{word:08x}',sequence,actual,expected[word])
                counts['native_prefix_word_checks']+=1
        counts['native_sequences']+=1
        add_stats(native_totals,chart.stats)
        native_records.append(dict(addends=[dict(value=str(a.value),zero_sign=a.zero_sign) for a in sequence],
                                   stats=chart.stats,steps=chart.steps))
    # Exact-product wrapper: all products are formed and counted, never pre-rounded.
    pairs=[(BF16.round(pow2(-75)),BF16.round(pow2(-75))),
           (BF16.round(pow2(64)),BF16.round(pow2(64))),
           (BF16.sign_bit, BF16.round(1))]
    wrapper_records=[]
    for pair in pairs:
        chart=compile_fma_operands([pair])
        for word in native_initial:
            expected=FP32.fma(word,*pair,BF16)
            assert chart.query(word)==expected
            counts['fma_wrapper_checks']+=1
        assert chart.stats['coefficient_reads']==chart.stats['input_reads']==chart.stats['exact_products']==1
        wrapper_records.append(chart.stats)
    under=compile_fma_operands([pairs[0]])
    over=compile_fma_operands([pairs[1]])
    assert under.query(1)==2
    assert over.query(FP32.round(-pow2(127)))==0x7f000000
    negzero=compile_fma_operands([pairs[2]])
    assert negzero.query(FP32.sign_bit)==FP32.sign_bit
    try:
        under.query(FP32.inf|1)
    except ValueError:
        counts['rejected_nan_checks']+=1
    else:raise AssertionError('NaN was silently admitted')
    try:
        compile_fma_operands([(BF16.inf,BF16.round(1))])
    except ValueError:
        counts['rejected_nonfinite_product_checks']+=1
    else:raise AssertionError('Nonfinite product operand was silently admitted')
    for bad in (-1, 1 << 32):
        try:under.query(bad)
        except ValueError:counts['rejected_invalid_word_checks']+=1
        else:raise AssertionError('Out-of-range input word was silently admitted')
    try:compile_fma_operands([(1<<16,BF16.round(1))])
    except ValueError:counts['rejected_invalid_word_checks']+=1
    else:raise AssertionError('Out-of-range operand word was silently admitted')
    try:Addend(Q(1,3))
    except ValueError:counts['rejected_nondyadic_checks']+=1
    else:raise AssertionError('A non-dyadic was silently admitted')
    return dict(status='PASS',scope='complete scalar charts under declared RNE finite-product ABI; not HF/hardware',
                counts=counts,tiny_format=dict(precision=4,exponent_bits=3,initial_words=len(initial)),
                tiny_totals=tiny_totals,tiny_peak_chart_range=[min(tiny_peaks),max(tiny_peaks)],
                fp32_initial_words=[f'{word:08x}' for word in sorted(native_initial)],
                native_totals=native_totals,native_records=native_records,wrapper_records=wrapper_records,
                fp32_regions=len(NativeCharts(FP32).regions),
                elapsed_seconds=time.monotonic()-start,
                preregistration_sha256=hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite a validation record')
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','counts','tiny_peak_chart_range','native_totals','elapsed_seconds')},indent=2))
