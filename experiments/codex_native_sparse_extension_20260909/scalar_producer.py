"""Restricted, stdlib-only scalar reference and two-sparse column producer.

This module deliberately implements only the finite-BF16 scalar ABI described
in PREREGISTRATION.md.  It is not a tensor, CUDA, or Hugging Face producer.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional

CANONICAL_NAN = 0x7FC00000


def _u32_to_f32(word: int) -> float:
    return struct.unpack(">f", struct.pack(">I", word & 0xFFFFFFFF))[0]


def _f32_to_u32(value: float) -> int:
    if math.isnan(value):
        return CANONICAL_NAN
    try:
        return struct.unpack(">I", struct.pack(">f", value))[0]
    except OverflowError:
        return 0xFF800000 if math.copysign(1.0, value) < 0 else 0x7F800000


def _round32(value: float) -> float:
    return _u32_to_f32(_f32_to_u32(value))


def _add32(left: float, right: float) -> float:
    return _round32(left + right)


def _mul32(left: float, right: float) -> float:
    return _round32(left * right)


def _bf16_to_f32(word: int) -> float:
    return _u32_to_f32((word & 0xFFFF) << 16)


def _f32_to_bf16_word(value: float) -> int:
    """FP32 to BF16 RNE, canonicalizing every NaN at FP32 first."""
    bits = _f32_to_u32(value)
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF):
        bits = CANONICAL_NAN
    # Round the discarded low 16 bits to nearest even.
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _is_finite_bf16(word: int) -> bool:
    return isinstance(word, int) and 0 <= word <= 0xFFFF and ((word >> 7) & 0xFF) != 0xFF


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0


def _next_power_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _tree_sum(values: list[float], width: int) -> tuple[float, int]:
    work = values + [0.0] * (width - len(values))
    additions = 0
    while len(work) > 1:
        work = [_add32(work[i], work[i + 1]) for i in range(0, len(work), 2)]
        additions += len(work)
    result = work[0]
    return (_u32_to_f32(CANONICAL_NAN) if math.isnan(result) else result), additions


@dataclass(frozen=True)
class Store:
    rows: int
    cols: int
    tree_width: int
    zero_signs: tuple[int, ...]
    columns: tuple[tuple[int, ...], ...]
    positive_zero_counts: tuple[int, ...]
    compile_stats: dict


def compile_store(matrix_words: list[list[int]], zero_signs: Optional[list[int]] = None) -> Store:
    if not matrix_words or not matrix_words[0]:
        raise ValueError("matrix_words must be nonempty")
    m, n = len(matrix_words), len(matrix_words[0])
    if any(len(row) != n for row in matrix_words):
        raise ValueError("matrix_words must be rectangular")
    signs = tuple(0 for _ in range(n)) if zero_signs is None else tuple(zero_signs)
    if len(signs) != n or any(sign not in (0, 1) for sign in signs):
        raise ValueError("zero_signs must contain one 0 or 1 per input")
    # One source traversal validates, transposes into paid columns, and counts
    # the declared inactive-zero products.  It does not hide a second scan.
    width = _next_power_two(n)
    mutable_columns = [[] for _ in range(n)]
    counts = [width - n for _ in range(m)]  # +0 padding leaves
    for r, row in enumerate(matrix_words):
        for c, word in enumerate(row):
            if not _is_finite_bf16(word):
                raise ValueError("source BF16 word must be finite")
            mutable_columns[c].append(word)
            source_sign = (word >> 15) & 1
            if source_sign ^ signs[c] == 0:
                counts[r] += 1
    columns = tuple(tuple(column) for column in mutable_columns)
    return Store(m, n, width, signs, columns, tuple(counts), {
        "matrix_word_reads": m * n, "source_validation_reads": m * n,
        "source_traversals": 1,
        "column_payload_words": m * n, "row_metadata_words": m,
        "preprocess_source_copies": m * n,
    })


def _validate_input(store: Store, input_words: list[int], max_support: int) -> list[int]:
    if len(input_words) != store.cols:
        raise ValueError("input_words length does not match store")
    if max_support < 0 or max_support > 2:
        raise ValueError("max_support must be between 0 and 2")
    active = []
    for c, word in enumerate(input_words):
        if not _is_finite_bf16(word):
            raise ValueError("input BF16 word must be finite")
        if (word & 0x7FFF) == 0:
            if ((word >> 15) & 1) != store.zero_signs[c]:
                raise ValueError("inactive zero sign does not match fixed zero_signs")
        else:
            active.append(c)
    if len(active) > 2 or len(active) > max_support:
        raise ValueError("input support exceeds max_support")
    return active


def project_fp32(store: Store, input_words: list[int], max_support: int = 2) -> tuple[list[float], dict]:
    active = _validate_input(store, input_words, max_support)
    inputs = {c: _bf16_to_f32(input_words[c]) for c in active}
    out: list[float] = []
    products = additions = 0
    for r in range(store.rows):
        expected_positive = 0
        all_active_negative_zero = True
        any_active_nonzero = False
        selected: list[float] = []
        for c in active:
            source_word = store.columns[c][r]
            source = _bf16_to_f32(source_word)
            value = _mul32(source, inputs[c])
            selected.append(value)
            products += 1
            expected_positive += 1 if (((source_word >> 15) & 1) ^ store.zero_signs[c]) == 0 else 0
            if value != 0.0:
                any_active_nonzero = True
            if not _is_negative_zero(value):
                all_active_negative_zero = False
        # This is only a <=2-support proof: every omitted subtree is replaced
        # by +0, and one or two surviving leaves keep their original order.
        # Thus their balanced FP32 tree reduces to identity or one FP32 add;
        # this code makes no arbitrary-k tree-evaluation claim.
        if not selected:
            result = 0.0
        elif len(selected) == 1:
            result = selected[0]
        else:
            result = _add32(selected[0], selected[1])
            additions += 1
        if math.isnan(result):
            result = _u32_to_f32(CANONICAL_NAN)
        if result == 0.0:
            remaining_positive = store.positive_zero_counts[r] - expected_positive
            # The declared theorem rule: a negative result requires every
            # original product/padding leaf to be negative zero.
            if not any_active_nonzero and remaining_positive == 0 and all_active_negative_zero:
                result = -0.0
            else:
                result = 0.0
        out.append(result)
    return out, {
        "input_scans": store.cols, "support": len(active), "column_word_reads": store.rows * len(active),
        "metadata_reads": store.rows, "actual_products": products, "actual_additions": additions,
        "output_words": store.rows,
    }


def project(store: Store, input_words: list[int], max_support: int = 2) -> tuple[list[int], dict]:
    values, stats = project_fp32(store, input_words, max_support)
    return [_f32_to_bf16_word(value) for value in values], stats


def full_reference_fp32(matrix_words: list[list[int]], input_words: list[int]) -> list[float]:
    # Reference intentionally accepts the same finite operand domain, without a sparse admission rule.
    if not matrix_words or not matrix_words[0]:
        raise ValueError("matrix_words must be nonempty")
    n = len(matrix_words[0])
    if any(len(row) != n for row in matrix_words):
        raise ValueError("matrix_words must be rectangular")
    if len(input_words) != n or any(not _is_finite_bf16(word) for word in input_words):
        raise ValueError("input BF16 word must be finite and match matrix width")
    if any(not _is_finite_bf16(word) for row in matrix_words for word in row):
        raise ValueError("source BF16 word must be finite")
    inputs = [_bf16_to_f32(word) for word in input_words]
    width = _next_power_two(n)
    output = []
    for row in matrix_words:
        leaves = [_mul32(_bf16_to_f32(row[c]), inputs[c]) for c in range(n)]
        value, _ = _tree_sum(leaves, width)
        output.append(value)
    return output


def full_reference(matrix_words: list[list[int]], input_words: list[int]) -> list[int]:
    return [_f32_to_bf16_word(value) for value in full_reference_fp32(matrix_words, input_words)]


def cost_formula(m: int = 16384, n: int = 16384, k: int = 2) -> dict:
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    width = _next_power_two(n)
    common = {
        "compile_matrix_reads": m * n, "compile_validation_reads": m * n,
        "compile_source_traversals": 1, "compile_payload_copies": m * n,
        "original_plus_compiled_bf16_bytes": 4 * m * n,
        "row_positive_zero_count_bytes": 4 * m,
        "zero_sign_pattern_bytes": (n + 7) // 8,
        "actual_python_allocator_bytes": "paid, implementation-dependent, unclosed",
        "positive_control_index_terms": m * n,
        "allocator_kv_fallback_terms": "paid and unclosed; no free fallback",
    }
    if k > 2:
        return common | {"runtime_mode": "full_reference_fallback_paid_not_sparse_admission",
            "runtime_input_scans": n, "runtime_matrix_reads": m * n,
            "runtime_metadata_reads": 0, "runtime_products": m * n,
            "runtime_additions": m * (width - 1), "runtime_outputs": m}
    return common | {"runtime_mode": "two_sparse_column_producer", "runtime_input_scans": n,
        "runtime_column_reads": m * k, "runtime_metadata_reads": m,
        "runtime_products": m * k, "runtime_additions_upper_bound": m * max(k - 1, 0),
        "runtime_outputs": m}
