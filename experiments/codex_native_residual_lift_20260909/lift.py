"""Bounded CPU check of the specified two-query BF16 residual lift.

This is an O3 numerical-lifting experiment only.  It deliberately performs two
full augmented matvecs and accounts for them; it is not an acceleration core.
"""
import json
import hashlib
import os
import platform
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def bits32(x):
    return int(np.asarray([np.float32(x)], dtype=np.float32).view(np.uint32)[0])


def f32_from_bits(word):
    return np.asarray([word], dtype=np.uint32).view(np.float32)[0]


def bf16_rne(x):
    """Return the BF16 RNE-stored value as an exactly embedded float32."""
    word = bits32(x)
    top = word >> 16
    low = word & 0xFFFF
    # Finite inputs are guaranteed by the declared exponent/domain checks.
    if low > 0x8000 or (low == 0x8000 and (top & 1)):
        top += 1
    return f32_from_bits((top & 0xFFFF) << 16)


def bf16_word(x):
    return bits32(x) >> 16


def bf16_normal(exponent, fraction):
    if not (-100 <= exponent <= 100):
        raise ValueError("alpha unbiased exponent outside [-100,100]")
    if not (0 <= fraction <= 127):
        raise ValueError("BF16 fraction outside [0,127]")
    return f32_from_bits(((exponent + 127) << 23) | (fraction << 16))


def bf16_fraction(x):
    """Exact rational represented by a finite BF16 word."""
    word = bf16_word(x)
    sign = -1 if (word >> 15) else 1
    exponent = (word >> 7) & 0xFF
    fraction = word & 0x7F
    if exponent == 0:
        if fraction == 0:
            return Fraction(0)
        return sign * Fraction(fraction, 1) * Fraction(2) ** (-133)
    if exponent == 0xFF:
        raise ValueError("non-finite BF16 value")
    return sign * Fraction(128 + fraction, 1) * Fraction(2) ** (exponent - 127 - 7)


def round_even(q):
    """Nearest integer with ties to even, for a Python exact Fraction."""
    if q < 0:
        return -round_even(-q)
    lower = q.numerator // q.denominator
    rem = q.numerator % q.denominator
    twice = 2 * rem
    if twice < q.denominator:
        return lower
    if twice > q.denominator:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def require_binary(B, v, n):
    if n < 1 or n > 16384:
        raise ValueError("n must be in [1,16384]")
    if B.ndim != 2 or B.shape[0] < 1 or B.shape[1] != n:
        raise ValueError("B shape does not match n")
    if v.shape != (n,):
        raise ValueError("v shape does not match n")
    if not np.all((B == 0) | (B == 1)) or not np.all((v == 0) | (v == 1)):
        raise ValueError("B and v must be binary")


def compile_matrix(B):
    m, n = B.shape
    require_binary(B, np.zeros(n,dtype=np.uint8), n)
    b = n.bit_length()  # ceil(log2(n+1)), including n that is a power of two.
    D = n + m * b
    C = np.zeros((m, D), dtype=np.float32)
    C[:, :n] = B.astype(np.float32, copy=False)
    for i in range(m):
        for j in range(b):
            C[i, n + i * b + j] = np.float32(-(1 << j))
    return C, b, D


def round_bf16_ratio(y, alpha):
    """RNE quotient using finite integer significands, not an undefined witness."""
    yw, aw = bf16_word(y), bf16_word(alpha)
    if (yw & 0x7fff) == 0:
        return 0
    ey, ea = (yw >> 7) & 255, (aw >> 7) & 255
    if not (0 < ey < 255 and 0 < ea < 255) or aw >> 15:
        raise ValueError('ratio outside normal finite positive-alpha protocol')
    delta = ey - ea
    numerator = (128 + (yw & 127)) << max(delta, 0)
    denominator = (128 + (aw & 127)) << max(-delta, 0)
    quotient, remainder = divmod(numerator, denominator)
    quotient += (2 * remainder > denominator or
                 (2 * remainder == denominator and quotient % 2 == 1))
    return -quotient if yw >> 15 else quotient


def ordered_matvec(C, x):
    """Specified separate float32 products then padded balanced float32 tree."""
    m, D = C.shape
    P = 1 << (D - 1).bit_length()
    out = np.empty(m, dtype=np.float32)
    for i in range(m):
        leaves = [np.float32(C[i, j] * x[j]) for j in range(D)]
        leaves.extend([np.float32(0.0)] * (P - D))
        while len(leaves) > 1:
            leaves = [np.float32(leaves[k] + leaves[k + 1]) for k in range(0, len(leaves), 2)]
        out[i] = bf16_rne(leaves[0])
    return out, P


def run_case(name, B, v, alpha0, alpha1):
    require_binary(B, v, B.shape[1])
    if not (np.isfinite(alpha0) and np.isfinite(alpha1) and alpha0 > 0 and alpha1 > 0):
        raise ValueError("alphas must be positive finite BF16 values")
    if bf16_rne(alpha0) != alpha0 or bf16_rne(alpha1) != alpha1:
        raise ValueError("alphas must already be BF16")
    for alpha in (alpha0,alpha1):
        if not -100 <= ((bf16_word(alpha) >> 7) & 255) - 127 <= 100:
            raise ValueError('alpha unbiased exponent outside [-100,100]')
    m, n = B.shape
    C, b, D = compile_matrix(B)
    x0 = np.zeros(D, dtype=np.float32)
    x0[:n] = v.astype(np.float32) * alpha0
    y0, P = ordered_matvec(C, x0)
    k0_unclamped = [round_bf16_ratio(y,alpha0) for y in y0]
    k0 = [min(n, max(0, value)) for value in k0_unclamped]
    x1 = np.zeros(D, dtype=np.float32)
    x1[:n] = v.astype(np.float32) * alpha1
    for i, value in enumerate(k0):
        for j in range(b):
            if (value >> j) & 1:
                x1[n + i * b + j] = alpha1
    y1, _ = ordered_matvec(C, x1)
    correction = [round_bf16_ratio(y,alpha1) for y in y1]
    decoded = [k0[i] + correction[i] for i in range(m)]
    parity = [value % 2 for value in decoded]
    # Wide independent oracle is used only after both adaptive native calls.
    truth = [int(row.astype(np.int64) @ v.astype(np.int64)) for row in B]
    assert k0_unclamped == [round_even(bf16_fraction(y)/bf16_fraction(alpha0)) for y in y0]
    assert correction == [round_even(bf16_fraction(y)/bf16_fraction(alpha1)) for y in y1]
    if any(value < 0 or value > n for value in truth):
        raise AssertionError("true count outside [0,n]; do not hide it by decoder clamp")
    residuals = [truth[i] - k0[i] for i in range(m)]
    if decoded != truth:
        raise AssertionError("two-query decode mismatch")
    return {
        "name": name, "m": int(m), "n": int(n), "b": int(b), "D": int(D), "P": int(P),
        "alpha0_bf16_word_hex": f"0x{bf16_word(alpha0):04x}",
        "alpha1_bf16_word_hex": f"0x{bf16_word(alpha1):04x}",
        "truth_counts": truth, "k0_unclamped": k0_unclamped, "k0": k0,
        "residuals_truth_minus_k0": residuals, "correction": correction,
        "decoded_counts": decoded, "parity": parity,
        "y0_bf16_words_hex": [f"0x{bf16_word(y):04x}" for y in y0],
        "y1_bf16_words_hex": [f"0x{bf16_word(y):04x}" for y in y1],
        "k0_clamp_events": sum(a != b for a, b in zip(k0_unclamped, k0)),
        "max_abs_residual": max(abs(x) for x in residuals),
        "all_decode_exact": decoded == truth,
        "all_parity_exact": parity == [x % 2 for x in truth],
        "numeric_accounting": {
            "compiler_source_binary_reads": int(m * n), "compiler_bias_coefficient_writes": int(m * b),
            "compiler_total_coefficient_writes": int(m * D + m*n + m*b), "two_matvec_matrix_reads": int(2 * m * D),
            "two_matvec_products": int(2 * m * D), "two_matvec_padded_zero_leaves": int(2 * m * (P - D)),
            "two_matvec_reduction_additions": int(2 * m * (P - 1)),
            "two_input_vector_lanes_written": int(2 * D + 2*n + sum(x.bit_count() for x in k0)), "query2_bias_lanes_set": int(sum(x.bit_count() for x in k0)),
            "decode_integer_division_upper": int(2 * m), "decode_round_even_calls": int(2 * m),
            "actual_numpy_coefficient_payload_bytes": int(C.nbytes),
            "logical_bf16_coefficient_payload_bytes": int(2*m*D),
            "bias_bit_tests": int(m*b),
            "allocation_scope": "array payload and logical actions; not full Python object/allocator or hardware peak",
        },
    }


def collision_search(n, alpha):
    seen = {}
    collision = None
    for count in range(n + 1):
        stored = bf16_rne(np.float32(np.float32(alpha) * np.float32(count)))
        key = bf16_word(stored)
        if key in seen and seen[key] != count:
            collision = (seen[key], count, key)
            break
        seen[key] = count
    if collision is None:
        return {"found": False, "counts_examined": int(n + 1)}
    left, right, word = collision
    return {"found": True, "counts_examined": int(right + 1), "left_count": int(left),
            "right_count": int(right), "same_y0_bf16_word_hex": f"0x{word:04x}",
            "different_integer": left != right, "different_parity": (left % 2) != (right % 2)}


def main():
    rng = np.random.default_rng(20260909)
    alphas = [("lo", bf16_normal(-100, 0)), ("one", bf16_normal(0, 0)),
              ("mid", bf16_normal(0, 64)), ("hi", bf16_normal(100, 127))]
    cases = []
    # Declared controls, including the required n=8192/m>=4 and n=16384/m>=2 scales.
    B8192_all = np.ones((4, 8192), dtype=np.uint8)
    v8192_all = np.ones(8192, dtype=np.uint8)
    B8192_high = np.zeros((4, 8192), dtype=np.uint8)
    for i, count in enumerate((8191, 8190, 4097, 257)):
        B8192_high[i, :count] = 1
    B8192_zero = np.zeros((4, 8192), dtype=np.uint8)
    B8192_random = rng.integers(0, 2, size=(4, 8192), dtype=np.uint8)
    v8192_random = rng.integers(0, 2, size=8192, dtype=np.uint8)
    B16384_random = rng.integers(0, 2, size=(2, 16384), dtype=np.uint8)
    v16384_random = rng.integers(0, 2, size=16384, dtype=np.uint8)
    fixtures = [("all_ones_8192", B8192_all, v8192_all), ("high_counts_8192", B8192_high, v8192_all),
                ("zeros_8192", B8192_zero, v8192_all), ("random_8192", B8192_random, v8192_random),
                ("random_16384", B16384_random, v16384_random)]
    fixtures += [('all_ones_16384',np.ones((2,16384),dtype=np.uint8),np.ones(16384,dtype=np.uint8)),
                 ('empty_input_8192',B8192_random,np.zeros(8192,dtype=np.uint8)),
                 ('boundary_n1',np.ones((2,1),dtype=np.uint8),np.ones(1,dtype=np.uint8))]
    for fixture_name, B, v in fixtures:
        for alpha_name, alpha in alphas:
            cases.append(run_case(fixture_name + "_" + alpha_name, B, v, alpha, alpha))
        cases.append(run_case(fixture_name+'_changed_scale',B,v,alphas[-1][1],alphas[0][1]))
    for trial in range(64):
        n=(1,2,3,7,8,15,16,31,63,255,256,257,1023)[trial%13]
        m=1+trial%8
        B=rng.integers(0,2,size=(m,n),dtype=np.uint8);v=rng.integers(0,2,size=n,dtype=np.uint8)
        a0=bf16_normal(int(rng.integers(-100,101)),int(rng.integers(0,128)))
        a1=bf16_normal(int(rng.integers(-100,101)),int(rng.integers(0,128)))
        cases.append(run_case('random_'+str(trial),B,v,a0,a1))

    collision = collision_search(8192, bf16_normal(0, 0))
    if not collision["found"]:
        raise AssertionError("expected BF16 single-query collision was not found")
    # Validate the discovered collision by the required ordered full augmented program.
    left = int(collision["left_count"])
    right = int(collision["right_count"])
    v = np.ones(8192, dtype=np.uint8)
    BL = np.zeros((4, 8192), dtype=np.uint8); BR = np.zeros((4, 8192), dtype=np.uint8)
    BL[0, :left] = 1; BR[0, :right] = 1
    left_run = run_case("collision_left_ordered", BL, v, bf16_normal(0, 0), bf16_normal(0, 0))
    right_run = run_case("collision_right_ordered", BR, v, bf16_normal(0, 0), bf16_normal(0, 0))
    cases.extend([left_run,right_run])
    collision["ordered_program_same_y0"] = left_run["y0_bf16_words_hex"][0] == right_run["y0_bf16_words_hex"][0]
    collision["ordered_program_two_query_recovers_both"] = left_run["decoded_counts"][0] == left and right_run["decoded_counts"][0] == right

    if not all(case["all_decode_exact"] and case["all_parity_exact"] for case in cases):
        raise AssertionError("case failure")
    maximum_residual = max(case["max_abs_residual"] for case in cases)
    if maximum_residual > 65:
        raise AssertionError("observed residual exceeds declared 65 bound")
    result = {
        "experiment": "native_residual_lift", "scope": "bounded CPU O3 native arithmetic lifting only",
        "status": "PASS_DECLARED_FINITE_CASES_ONLY", "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "env_UTF8": os.environ.get("PYTHONUTF8"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest(),
        "provenance": "primary corrected incomplete Terra-requested source; no result/verified-model metadata received from agent",
        "model_metadata": "not exposed; no torch/HF model invoked",
        "specification": {"n_max": 16384, "alpha_unbiased_exponent_min": -100, "alpha_unbiased_exponent_max": 100,
                          "matvecs_per_query": 2, "coefficient_format": "explicit BF16 C=[B | row-local -2^j bias lanes]",
                          "execution": "separate float32 products; padded balanced float32 reduction; BF16 RNE store"},
        "cases": cases, "case_count": len(cases), "maximum_observed_abs_residual": maximum_residual,
        "collision_search": collision,
        "limits": ["Two full augmented matvecs are executed and charged.", "No >=10x claim; no subdense producer admission.",
                   "No GPU, CUDA, HF model, logits, KV, RNG, 405B, or target hardware measurement.",
                   "Finite test cases check the lifting protocol; they do not prove theorem truth or mission status."],
        "mission_status": "unchanged: THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED",
    }
    if (HERE/'results.json').exists():
        raise FileExistsError('refuse overwrite results.json')
    (HERE / "results.json").write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": result["status"], "case_count": result["case_count"],
                      "maximum_observed_abs_residual": maximum_residual, "collision": collision}, indent=2))


if __name__ == "__main__":
    main()
