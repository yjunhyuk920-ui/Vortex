"""Exact, bounded checks for the native-coding closure audit.

No inference acceleration is implemented. The integer reference and libm
oracle are deliberately separate. Python integer work is not GPU timing.
Run from the repository root:
  python experiments/theory_closure/verify.py --output results/theory_closure
"""
from __future__ import annotations
import argparse
import base64
import ctypes
import ctypes.util
import hashlib
import itertools
import json
import random
import struct
from pathlib import Path

SIGN = 0x80000000
EXP = 0x7F800000
FRAC = 0x007FFFFF
HALF_BF16_MIN = 0x00008000  # 2**-134, exactly representable in binary32


def is_finite(u: int) -> bool:
    return u & EXP != EXP


def units(u: int) -> int:
    """Finite binary32 as an integer times 2**-149; signed zero maps to 0."""
    if not 0 <= u < 2**32 or not is_finite(u):
        raise ValueError("expected finite binary32 bits")
    e, f = (u >> 23) & 255, u & FRAC
    v = f if e == 0 else ((1 << 23) | f) << (e - 1)
    return -v if u & SIGN else v


def rne_quotient(n: int, shift: int) -> int:
    if shift == 0:
        return n
    q, r = divmod(n, 1 << shift)
    h = 1 << (shift - 1)
    return q + int(r > h or (r == h and q & 1))


def round_units_f32(n: int, negative_zero: bool = False) -> int:
    s = SIGN if n < 0 or (n == 0 and negative_zero) else 0
    n = abs(n)
    if n < 1 << 23:
        return s | n
    shift = n.bit_length() - 24
    q = rne_quotient(n, shift)
    if q == 1 << 24:
        q >>= 1
        shift += 1
    e = shift + 1
    return s | (EXP if e >= 255 else (e << 23) | (q - (1 << 23)))


def add_ref(a: int, b: int) -> int:
    """Exact integer reference, including infinities and RN-even signed zero."""
    if not is_finite(a) or not is_finite(b):
        if (a & FRAC and not is_finite(a)) or (b & FRAC and not is_finite(b)):
            return 0x7FC00000
        if not is_finite(a) and not is_finite(b) and (a ^ b) & SIGN:
            return 0x7FC00000
        return a if not is_finite(a) else b
    return round_units_f32(units(a) + units(b), a == SIGN and b == SIGN)


def bf16_ref(u: int) -> int:
    """Independent integer RN-even store, not a float32 bit-rounding trick."""
    if not is_finite(u):
        return (u >> 16) | (0x0040 if u & FRAC else 0)
    s = 0x8000 if u & SIGN else 0
    n = abs(units(u))
    shift = max(16, n.bit_length() - 8)
    q = rne_quotient(n, shift)
    if q == 256:
        q >>= 1
        shift += 1
    if q < 128:
        return s | q
    e = shift - 15
    return s | (0x7F80 if e >= 255 else (e << 7) | (q - 128))


def bf16_bits(u: int) -> int:
    if not is_finite(u):
        return (u >> 16) | (0x0040 if u & FRAC else 0)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def as_float(u: int) -> float:
    return struct.unpack('<f', struct.pack('<I', u))[0]


def as_bits(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def get_native_add():
    name = ctypes.util.find_library('m')
    if not name:
        raise RuntimeError('libm is required for the independent comparison')
    lib = ctypes.CDLL(name)
    lib.fmaf.argtypes = [ctypes.c_float] * 3
    lib.fmaf.restype = ctypes.c_float
    lib.fegetround.restype = ctypes.c_int
    if lib.fegetround() != 0:  # Linux FE_TONEAREST, environment of this artifact
        raise RuntimeError('expected Linux FE_TONEAREST')
    def native(a: int, b: int) -> int:
        return as_bits(lib.fmaf(1.0, as_float(b), as_float(a)))
    return native


def separator(a: int, b: int) -> tuple[int, ...]:
    """A distinguishing continuation, not a checkpoint-to-state generator.

For numerically different finite states: subtract a, then add the signed
half-minimum-BF16 value. Signed zeros are distinguished by an empty suffix.
"""
    if a == b or not is_finite(a) or not is_finite(b):
        raise ValueError('expected distinct finite states')
    if units(a) == units(b):
        return ()
    s = SIGN if units(b) < units(a) else 0
    return (a ^ SIGN, s | HALF_BF16_MIN)


def run(a: int, suffix: tuple[int, ...], add) -> int:
    for term in suffix:
        a = add(a, term)
    return a


def native_sum(terms: tuple[int, ...], add) -> int:
    return run(0, terms, add)


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/theory_closure'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).with_name('preregistration.json')
    config = json.loads(config_path.read_text())
    native = get_native_add()
    edge = [int(x, 16) for x in config['edge_words']]
    pairs = list(itertools.permutations(edge, 2))
    rng = random.Random(config['random_seed'])
    while len(pairs) < len(edge) * (len(edge) - 1) + config['random_pairs']:
        a, b = rng.getrandbits(32), rng.getrandbits(32)
        if a != b and is_finite(a) and is_finite(b):
            pairs.append((a, b))
    rows = []
    addition_checks = 0
    for a, b in pairs:
        suffix = separator(a, b)
        outputs = []
        for initial in (a, b):
            r, h = initial, initial
            assert bf16_ref(r) == bf16_bits(r)
            for term in suffix:
                r, h = add_ref(r, term), native(h, term)
                addition_checks += 1
                assert r == h, (a, b, term, r, h)
                assert bf16_ref(r) == bf16_bits(r)
            outputs.append(bf16_ref(r))
        assert outputs[0] != outputs[1], (a, b, suffix, outputs)
        rows.append({'a': f'{a:08x}', 'b': f'{b:08x}',
                     'suffix': [f'{u:08x}' for u in suffix],
                     'bf16_outputs': [f'{u:04x}' for u in outputs]})
    h, one, minus_h, zero = (as_bits(x) for x in (2.0**24, 1.0, -2.0**24, 0.0))
    t1, t2 = (h, one, minus_h), (h, minus_h, one)
    r1, r2 = native_sum(t1, native), native_sum(t2, native)
    assert r1 == 0 and r2 == one
    fx = native_sum((h, zero, minus_h), native)
    fy = native_sum((zero, one, zero), native)
    assert native(fx, fy) != r1
    # Same immediate BF16 output, different future BF16 output.
    a, b = 1, 2
    suffix = separator(a, b)
    assert bf16_ref(a) == bf16_ref(b) == 0
    assert bf16_ref(run(a, suffix, add_ref)) == 0
    assert bf16_ref(run(b, suffix, add_ref)) == 1
    count = 2**32 - 2**24  # finite encodings including both signed zeros
    state_bits = {str(m): (pow(count, m) - 1).bit_length() for m in (1, 8, 121, 16384)}
    # Counts refer to an unrestricted transducer state interface, not a fixed
    # checkpoint's reachable states, and not a latency/inference lower bound.
    for invalid in ((one, one), (EXP, one), (one, 0x7FC00000)):
        try:
            separator(*invalid)
            raise AssertionError('invalid input was accepted')
        except ValueError:
            pass
    dump(args.output / 'witnesses.json', rows)
    observations = b''.join(struct.pack('<HH', *(int(v, 16) for v in row['bf16_outputs'])) for row in rows)
    dump(args.output / 'observations.json', {
        'layout': 'two little-endian uint16 BF16 observations per pair in preregistered order',
        'pairs': len(rows),
        'input_reconstruction': 'edge permutations, then distinct finite pairs from random.Random(seed).getrandbits(32); see preregistration and verify.py',
        'data_base64': base64.b64encode(observations).decode('ascii'),
        'raw_sha256': hashlib.sha256(observations).hexdigest()
    })
    summary = {
        'base_commit': config['base_commit'],
        'scope': config['scope'],
        'THEORY_STATUS': 'NOT_ESTABLISHED',
        'HARDWARE_STATUS': 'NOT_TESTED',
        'CORE_ADMISSION': False,
        'checks': {'separator_pairs': len(pairs), 'fma_add_comparisons': addition_checks,
                   'mismatches': 0, 'input_rejection_checks': 3},
        'permutation_collision': {'exact_integer_sum_both': 1,
             'native_fp32_bits': [f'{r1:08x}', f'{r2:08x}'],
             'native_bf16_bits': [f'{bf16_ref(r1):04x}', f'{bf16_ref(r2):04x}']},
        'nonadditivity': {'F_x_plus_y': 0, 'F_x_plus_F_y': 1},
        'immediate_store_collision': {'a': '00000001', 'b': '00000002',
             'immediate_bf16_both': '0000', 'suffix': [f'{u:08x}' for u in suffix],
             'future_bf16': ['0000', '0001']},
        'unrestricted_state_count': count,
        'minimum_fixed_bits_no_correlated_side_information': state_bits,
        'warning': 'These checks support scoped proofs, not a cheap generator, not target impossibility, and not full O1-O6 closure.',
        'preregistration_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'witnesses_sha256': hashlib.sha256((args.output / 'witnesses.json').read_bytes()).hexdigest()
    }
    dump(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
