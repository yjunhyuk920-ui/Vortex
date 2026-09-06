"""Scoped algebraic screens; NOT an LLM executor or a performance benchmark.

Run: python src/analyze.py --out results
All large sizes are calculated, never allocated. No original checkpoint is loaded.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct


def balanced_table_size(n: int, blocks: int, q: int, entry_bytes: int = 4) -> int:
    """Minimum explicit-table payload for a fixed number of nonempty blocks.

    Each block stores one entry for each of its q**b input tuples. No sharing,
    compression or input-dependent table selection is included in this format.
    The minimization relaxes row and reduction-tree alignment constraints.
    """
    if n < 1 or not 1 <= blocks <= n or q < 2 or entry_bytes < 1:
        raise ValueError('invalid explicit-table parameters')
    d, r = divmod(n, blocks)
    return entry_bytes * ((blocks - r) * q**d + r * q**(d + 1))


def fixed_block_cost(n: int, b: int, q: int, hot: int = 0) -> dict:
    if b < 1 or n % b or hot < 0:
        raise ValueError('n must be divisible by positive b; hot must be >=0')
    blocks = n // b
    storage = balanced_table_size(n, blocks, q)
    reads = 4 * blocks
    # Fixed equal-b tables, independent of the CURRENT query. Each local
    # query tuple uniform. Correlation between output rows does not matter.
    expected_cold = max(Fraction(0), Fraction(reads) - Fraction(min(hot, storage), q**b))
    return {
        'N': n, 'b': b, 'Q': q, 'blocks': blocks,
        'raw_bf16_bytes': 2*n, 'entry_bytes': 4,
        'table_bytes': storage, 'query_entry_bytes_before_cache': reads,
        'query_fraction': str(Fraction(reads, 2*n)),
        'table_to_weight_ratio': str(Fraction(storage, 2*n)),
        'hot_bytes_budgeted_entirely_to_tables': hot,
        'expected_cold_lower_bound': str(expected_cold),
        'preparation_output_write_lower_bound_bytes': storage,
        'classification': 'DERIVED_FIXED_FORMAT_NOT_MEASURED_NOT_GENERAL_LOWER_BOUND'
    }


def round_dyadic(x: Fraction, precision: int) -> Fraction:
    """RNE with p significand bits, unbounded exponent. Used ONLY on in-range
    finite witnesses. Does not purport to implement overflow, signed zero, NaNs
    or subnormal IEEE semantics. Those are unnecessary for these witnesses.
    """
    if precision < 2:
        raise ValueError('precision must be >= 2')
    x = Fraction(x)
    if not x:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    two = lambda k: Fraction(2**k) if k >= 0 else Fraction(1, 2**(-k))
    if a < two(e):
        e -= 1
    step = two(e - precision + 1)
    z = a / step
    floor, rem = divmod(z.numerator, z.denominator)
    if 2*rem > z.denominator or (2*rem == z.denominator and floor % 2):
        floor += 1
    return sign * floor * step


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16_word(x: float) -> int:
    """RNE from already finite FP32, used only on in-range witnesses."""
    u = struct.unpack('<I', struct.pack('<f', f32(x)))[0]
    if (u & 0x7f800000) == 0x7f800000:
        raise ValueError('nonfinite outside witness domain')
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff


def bf16_value(x: float) -> float:
    return struct.unpack('<f', struct.pack('<I', bf16_word(x) << 16))[0]


def native_witnesses() -> dict:
    # All coefficients and inputs individually BF16-exact. A table may safely
    # return the FP32 root of a native subtree. Rounding it to BF16 is different.
    records = []
    for first_inputs in [(1,1),(1,0)]:
        a = f32(f32(256.0*first_inputs[0]) + f32(1.0*first_inputs[1]))
        b = f32(f32(-256.0*1.0) + f32(1.0*0.0))
        direct = bf16_value(f32(a+b))
        prematurely_rounded = bf16_value(f32(bf16_value(a)+bf16_value(b)))
        exact_a = round_dyadic(Fraction(256*first_inputs[0] + first_inputs[1]),24)
        exact_b = round_dyadic(Fraction(-256),24)
        independent = round_dyadic(round_dyadic(exact_a+exact_b,24),8)
        assert Fraction(direct) == independent
        records.append({'first_inputs': first_inputs, 'fp32_left': a,
                        'fp32_right': b, 'bf16_left': bf16_value(a),
                        'reference_final': direct, 'premature_final': prematurely_rounded,
                        'independent_final': str(independent)})
    return {'weights': [256,1,-256,1], 'right_inputs': [1,0],
            'native_tree': 'BF16(RN32(RN32(p0+p1)+RN32(p2+p3)))',
            'rows': records, 'purpose': 'invalid BF16-only subtree state, not all encodings'}


def interval_gate(n: int, p: int = 8) -> dict:
    if n < 1 or n >= 2**24 or p < 2:
        raise ValueError('outside exact integer FP32 witness range')
    # Unread terms independently +/-1, known prefix |s|<=k.
    # RN_p(s-R)=RN_p(s+R) => R<=2**(-p)|s| => k>=n/(1+2**(-p)).
    denom = 2**p + 1
    lower = (n*2**p + denom - 1) // denom
    # Particular all-positive evaluation order: exact attainable extrema.
    actual_first = next(k for k in range(n+1)
                        if round_dyadic(Fraction(2*k-n),p) == round_dyadic(Fraction(n),p))
    assert actual_first >= lower
    return {'n': n, 'precision': p, 'derived_terms_required_at_least': lower,
            'derived_fraction': str(Fraction(lower,n)),
            'all_positive_first_endpoint_certificate_after_terms': actual_first,
            'all_positive_fraction': str(Fraction(actual_first,n)),
            'scope': 'independent unread-sign endpoint certifier; not arbitrary compiled algorithms'}


def run(out: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root/'PREREGISTER.json'
    config = json.loads(config_path.read_text())
    p = config['parameters']
    costs = [fixed_block_cost(p['coefficient_count'],b,q,p['hot_bytes'])
             for q in p['alphabets'] for b in p['block_sizes']]
    intervals = [interval_gate(n) for n in p['interval_n']]
    witness = native_witnesses()
    results = {
        'stage': config['stage'], 'preregister_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'table_cases': costs, 'interval_cases': intervals, 'native_witness': witness,
        'A_affine_transport': {'symbolic_example': 'f(b,z)=b*c+g(z)',
                              'compact_decoder': 'sum_j w_j*x_j, followed by declared rounding when valid',
                              'all_nonzero_input_coefficient_payload_read_fraction': '1',
                              'native_affine_equivalence': 'not valid in general; no unproved lift used'},
        'decision': {'A': 'NOT_ADMITTED_original_coefficient_work',
                     'B': 'REJECT_FIXED_UNSHARED_EXPLICIT_TABLE_FORMAT',
                     'C': 'REJECT_INDEPENDENT_SIGN_INTERVAL_SHORTCUT',
                     'THREE_QUALIFYING_NEW_PRINCIPLES': False,
                     'THEORY_STATUS': 'NOT_ESTABLISHED', 'CORE_ADMISSION': False,
                     'HARDWARE_STATUS': 'NOT_TESTED', 'O1_O6': 'OPEN'},
        'not_run': ['matrix/Transformer executor','subtree tables','old BDD compiler',
                    'public checkpoint','KV/RNG continuity','GPU','405B',
                    '4B Q4 baseline','latency quantiles','TTFT','GitHub Actions','full repository suite']
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    (out/'native_witness.json').write_text(json.dumps(witness,indent=2)+'\n')
    (out/'costs.jsonl').write_text(''.join(json.dumps(c,sort_keys=True)+'\n' for c in costs))
    print(json.dumps({'cost_cases':len(costs),'interval_cases':len(intervals),
                      'b32_binary':next(c for c in costs if c['b']==32 and c['Q']==2),
                      'largest_interval':intervals[-1]},indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
