"""Preregistered chart-algebra validation. No native chart producer is claimed."""
from fractions import Fraction as Q
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path
import time

from crossing import PeriodicMap
from reference import floor, pow2

ROOT = Path(__file__).resolve().parent


def independent_round(z, h):
    # Nearest of the two bracketing exact grid points; even-index tie.
    lo = floor(z/h)
    hi = lo + 1
    dl, dh = z-lo*h, hi*h-z
    chosen = lo if dl < dh or (dl == dh and lo % 2 == 0) else hi
    return chosen*h


def run():
    start = time.monotonic()
    counts = dict(itineraries=0, value_checks=0, inverse_checks=0,
                  equivariance_checks=0, wide_period_checks=0, max_levels=0)
    steps = tuple(product((Q(1,4), Q(1), Q(4)), (Q(-9,8), Q(-1,2), Q(0), Q(1,2), Q(9,8))))
    for sequence in product(steps, repeat=3):
        f = PeriodicMap.identity(Q(1))
        for h, c in sequence:
            f = f.then(h, c)
            f.check_invariants()
            counts['max_levels'] = max(counts['max_levels'], len(f.levels))
        counts['itineraries'] += 1
        for k in range(-17,18):
            expected = Q(k)
            for h, c in sequence:
                expected = independent_round(expected+c,h)
            assert f(k) == expected, ('value', sequence, k, f, expected)
            counts['value_checks'] += 1
            assert f(k+f.period) == f(k)+f.height
            counts['equivariance_checks'] += 1
        for threshold in (Q(-33,8), Q(-1,2), Q(0), Q(1,2), Q(33,8)):
            for strict in (False,True):
                k = f.inverse(threshold, strict)
                assert (f(k)>threshold if strict else f(k)>=threshold)
                assert (f(k-1)<=threshold if strict else f(k-1)<threshold)
                counts['inverse_checks'] += 1
    f = PeriodicMap.identity(pow2(-149))
    wide_steps = ((pow2(-149), pow2(-150)), (pow2(104), pow2(127)),
                  (pow2(-149), -pow2(127)), (pow2(0), Q(1,2)))
    for h,c in wide_steps:
        f = f.then(h,c)
        f.check_invariants()
    assert f.period == 1 << 254
    for k in (-(1<<255),-17,-1,0,1,17,(1<<254)-1,1<<254,1<<255):
        expected = k*pow2(-149)
        for h,c in wide_steps:
            expected = independent_round(expected+c,h)
        assert f(k) == expected
        counts['wide_period_checks'] += 1
    for threshold in (pow2(-149),Q(0),Q(1),pow2(127)):
        for strict in (False,True):
            k=f.inverse(threshold,strict)
            assert (f(k)>threshold if strict else f(k)>=threshold)
            assert (f(k-1)<=threshold if strict else f(k-1)<threshold)
            counts['wide_period_checks'] += 1
    # A concrete three-level witness, retained instead of incorrectly claiming two.
    witness = PeriodicMap.identity(Q(1)).then(Q(2),Q(0))
    assert len(witness.levels) == 3
    return dict(status='PASS', counts=counts,
                scope='fixed itinerary dyadic-map algebra only; native chart constructor DERIVED, not implemented',
                witness=dict(period=witness.period,cuts=witness.cuts,levels=list(map(str,witness.levels))),
                wide_period_bits=f.period.bit_length(),elapsed_seconds=time.monotonic()-start,
                preregistration_sha256=hashlib.sha256((ROOT/'CROSSING_PREREGISTRATION.md').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite a validation record')
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
