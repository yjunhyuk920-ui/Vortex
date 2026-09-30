"""Frozen inexpensive exact/arithmetic checks; no checkpoint or backend test."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
from pathlib import Path
import time

from reference import BF16, FP32, Format, Grid, compile_summary, compose, execute_with_escapes, identity, leaf, pow2

ROOT = Path(__file__).resolve().parent


class EnumeratedReference:
    """Independent nearest-neighbor reference for the tiny test format only."""
    def __init__(self, fmt):
        self.fmt = fmt
        self.values = [fmt.decode(word) for word in range(fmt.inf)]
        self.overflow = self.values[-1] + pow2(fmt.emax - fmt.frac_bits - 1)

    def round(self, z, zero_sign=0):
        if not z:
            return self.fmt.sign_bit if zero_sign else 0
        sign = self.fmt.sign_bit if z < 0 else 0
        z = abs(z)
        if z >= self.overflow:
            return sign | self.fmt.inf
        ix = bisect_left(self.values, z)
        if ix == len(self.values):
            return sign | (ix - 1)
        if self.values[ix] == z:
            return sign | ix
        lo = ix - 1
        dl, dh = z - self.values[lo], self.values[ix] - z
        chosen = lo if dl < dh or (dl == dh and lo % 2 == 0) else ix
        return sign | chosen

    def step(self, word, addend, zero_sign=0):
        if self.fmt.magnitude(word) == self.fmt.inf:
            return word
        a = self.fmt.decode(word)
        zs = int(not a and not addend and self.fmt.sign(word) and zero_sign)
        return self.round(a + addend, zs)


def in_guard(grid, z):
    lo, lc, hi, hc = grid.exact_guard
    t = z / grid.h
    return (t >= lo if lc else t > lo) and (t <= hi if hc else t < hi)


def exact_path(ref, grid, initial, products):
    word = initial
    valid = True
    for p in products:
        if not ref.fmt.is_finite(word):
            return word, False
        valid &= in_guard(grid, ref.fmt.decode(word) + p)
        word = ref.step(word, p)
    return word, valid


def balanced(items):
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return compose(balanced(items[:mid]), balanced(items[mid:]))


def validate():
    started = time.monotonic()
    fmt = Format(4, 3)
    ref = EnumeratedReference(fmt)
    counts = dict(single_leaf_cases=0, single_leaf_accepted=0, ordered_sequence_cases=0,
                  ordered_sequence_accepted=0, composition_cases=0, escape_cases=0,
                  native_edge_cases=0, mismatches=0)
    grids = [Grid(fmt, e, neg) for e in range(fmt.emin, fmt.emax + 1) for neg in (False, True)]
    # One exact eighth-ulp grid crosses all relevant representable and midpoint boundaries.
    for grid in grids:
        for numerator in range(-160, 161):
            p = Q(numerator, 8) * grid.h
            s = leaf(grid, p)
            for k in range(grid.state_bounds[0], grid.state_bounds[1] + 1):
                a = ref.round(k * grid.h)
                expected, valid = exact_path(ref, grid, a, [p])
                counts['single_leaf_cases'] += 1
                assert s.accepts(k) == valid, ('leaf-domain', grid, numerator, k, s, valid)
                if valid:
                    counts['single_leaf_accepted'] += 1
                    actual = ref.round(s.apply(k) * grid.h)
                    assert actual == expected, ('leaf-output', grid, numerator, k, actual, expected)
    alphabet = tuple(map(Q, (-1, 0, 1))) + (Q(-17, 4), Q(-5, 2), Q(-1, 2), Q(1, 2), Q(5, 2), Q(17, 4))
    for grid in grids:
        for ts in product(alphabet, repeat=3):
            ps = [t * grid.h for t in ts]
            leaves = [leaf(grid, p) for p in ps]
            s = compile_summary(grid, ps)
            right = compose(leaves[0], compose(leaves[1], leaves[2]))
            assert s == right == balanced(leaves), ('composition', grid, ts, s, right)
            counts['composition_cases'] += 1
            for k in range(grid.state_bounds[0], grid.state_bounds[1] + 1):
                a = ref.round(k * grid.h)
                expected, valid = exact_path(ref, grid, a, ps)
                counts['ordered_sequence_cases'] += 1
                assert s.accepts(k) == valid, ('sequence-domain', grid, ts, k, s, valid)
                if valid:
                    counts['ordered_sequence_accepted'] += 1
                    actual = ref.round(s.apply(k) * grid.h)
                    assert actual == expected, ('sequence-output', grid, ts, k, actual, expected)
    # All finite tiny initial words, both zeros, every fixed adversarial 3-addend sequence.
    escape_alphabet = (Q(-17), Q(-1), Q(-1, 32), Q(0), Q(1, 32), Q(1), Q(17))
    all_initial = list(range(fmt.inf)) + [fmt.sign_bit | k for k in range(fmt.inf)]
    total_escape_stats = None
    for ts in product(escape_alphabet, repeat=3):
        for initial in all_initial:
            expected = initial
            for p in ts:
                expected = ref.step(expected, p)
            actual, stats = execute_with_escapes(fmt, initial, ts)
            assert actual == expected, ('escape', initial, ts, actual, expected)
            counts['escape_cases'] += 1
            if total_escape_stats is None:
                total_escape_stats = {k: 0 for k in stats}
            for k, v in stats.items():
                total_escape_stats[k] += v
    # Preserve -0 only when both the accumulator and zero product are negative.
    for az in (0, fmt.sign_bit):
        for pz in (0, 1):
            expected = ref.step(az, Q(0), pz)
            actual, _ = execute_with_escapes(fmt, az, [(Q(0), pz)])
            assert actual == expected
            counts['escape_cases'] += 1
    # Independently search native finite words; exact products are never FP32-rounded early.
    under_w = BF16.round(pow2(-75))
    under_a = FP32.round(pow2(-149))
    under_exact = BF16.decode(under_w) ** 2
    under_fused = FP32.fma(under_a, under_w, under_w, BF16)
    under_separate = FP32.add_exact(under_a, FP32.decode(FP32.round(under_exact)))
    assert under_fused == 0x00000002 and under_separate == 0x00000001
    over_w = BF16.round(pow2(64))
    over_a = FP32.round(-pow2(127))
    over_fused = FP32.fma(over_a, over_w, over_w, BF16)
    over_product = FP32.round(BF16.decode(over_w) ** 2)
    assert over_fused == 0x7f000000 and over_product == 0x7f800000
    counts['native_edge_cases'] += 2
    native_boundaries = []
    for e in (FP32.emin, -1, 0, 1, FP32.emax):
        for neg in (False, True):
            grid = Grid(FP32, e, neg)
            a_int = grid.state_bounds[1] if neg else grid.state_bounds[0]
            a = FP32.round(a_int * grid.h)
            gl, lc, gh, hc = grid.exact_guard
            for endpoint, closed in ((gl, lc), (gh, hc)):
                for eps in (Q(-1, 16), Q(0), Q(1, 16)):
                    exact_sum_units = endpoint + eps
                    p = (exact_sum_units - a_int) * grid.h
                    s = leaf(grid, p)
                    expected = FP32.add_exact(a, p)
                    if s.accepts(a_int):
                        actual = FP32.round(s.apply(a_int) * grid.h)
                        assert actual == expected, ('native-boundary', e, neg, endpoint, eps)
                    assert s.accepts(a_int) == in_guard(grid, exact_sum_units * grid.h)
                    counts['native_edge_cases'] += 1
                    native_boundaries.append(dict(e=e, negative=neg, exact_sum_units=str(exact_sum_units),
                                                  accepted=s.accepts(a_int), result=f'{expected:08x}'))
    # Tie parity, negative tie floor, zero cancellation, and repeated binade exits.
    native_streams = (
        (Q(1), (pow2(-24), pow2(-24), -pow2(-23), pow2(0))),
        (Q(-1), (-pow2(-24), pow2(-24), Q(1), pow2(-149))),
        (pow2(127), (pow2(127), -pow2(127), Q(1))),
        (pow2(-126), (-pow2(-126), pow2(-149), pow2(-126))),
    )
    for start, ps in native_streams:
        initial = FP32.round(start)
        expected = initial
        for p in ps:
            expected = FP32.add_exact(expected, p)
        actual, _ = execute_with_escapes(FP32, initial, ps)
        assert actual == expected
        counts['native_edge_cases'] += 1
    return dict(status='PASS', scope='exact guarded scalar reference; not HF/target or speed evidence',
                counts=counts, reduced_format=dict(precision=fmt.precision, exponent_bits=fmt.exponent_bits),
                escape_totals=total_escape_stats,
                fused_vs_separate=dict(underflow_fused=f'{under_fused:08x}', underflow_separate=f'{under_separate:08x}',
                                       overflow_fused=f'{over_fused:08x}', overflow_separate_product=f'{over_product:08x}'),
                native_boundaries=native_boundaries,
                elapsed_seconds=time.monotonic()-started,
                preregistration_sha256=hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a validation record')
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'escape_totals', 'elapsed_seconds')}, indent=2))
