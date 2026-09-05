"""Exact E0 prerequisites, not a Transformer executor or a hardware benchmark.

New statement: four stored bits cannot answer all 2x2 binary rank-one parity
queries in two adaptive bit probes, even under arbitrary nonlinear encoding.
Five bits suffice. The proof is in the accompanying research note; this module
independently checks its finite algebra and its essential scope boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Iterable

BASE_SHA = "ff70c1ebbca943161684f33e6afa1dc169fa8f4c"


def outer_mask(left: int, right: int, rows: int, cols: int) -> int:
    if rows < 1 or cols < 1 or not 0 <= left < 1 << rows or not 0 <= right < 1 << cols:
        raise ValueError("invalid binary matrix query")
    return sum(((left >> i) & 1) * ((right >> j) & 1) << (i * cols + j)
               for i in range(rows) for j in range(cols))


def queries(rows: int, cols: int) -> list[int]:
    return [outer_mask(u, v, rows, cols)
            for u in range(1, 1 << rows) for v in range(1, 1 << cols)]


def gf2_rank(matrix: int, rows: int, cols: int) -> int:
    work = [(matrix >> (i * cols)) & ((1 << cols) - 1) for i in range(rows)]
    pivot = 0
    for bit in range(cols):
        found = next((i for i in range(pivot, rows) if work[i] >> bit & 1), None)
        if found is None:
            continue
        work[pivot], work[found] = work[found], work[pivot]
        for i in range(rows):
            if i != pivot and work[i] >> bit & 1:
                work[i] ^= work[pivot]
        pivot += 1
    return pivot


def character(mask: int, word: int) -> int:
    return 1 - 2 * ((mask & word).bit_count() & 1)


def low_kernel(width: int, distance: int) -> int:
    s = width - 2 * distance
    return s + (s * s - width) // 2  # degrees 1 and 2; NO constant


def rank_kernel(rows: int, cols: int, rank: int) -> int:
    return (1 << (rows + cols - rank)) - (1 << rows) - (1 << cols) + 1


def proof_scope(rows: int, cols: int, stored_cells: int, cell_bits: int,
                hot_advice_bits: int, all_matrices: bool, probes: int,
                fixed_decoder: bool = True) -> str:
    """Fail closed. This theorem does not reject seven-bit/word/native encoders."""
    if (rows, cols, stored_cells, cell_bits, hot_advice_bits, all_matrices, probes, fixed_decoder) == (
            2, 2, 4, 1, 0, True, 2, True):
        return "IMPOSSIBLE_IN_REGISTERED_SCOPE"
    return "NO_CONCLUSION_FROM_THIS_THEOREM"


def verify_kernel(rows: int, cols: int) -> dict:
    width = rows * cols
    if width > 8:
        raise ValueError("exhaustive integrity control capped at eight bits")
    q = queries(rows, cols)
    masks = [m for m in range(1, 1 << width) if m.bit_count() <= 2]
    size = 1 << width
    orthogonal_pairs = 0
    for a in q:
        for b in q:
            dot = sum(character(a, x) * character(b, x) for x in range(size))
            assert dot == (size if a == b else 0)
            orthogonal_pairs += 1
    for a in masks:
        for b in masks:
            assert sum(character(a, x) * character(b, x) for x in range(size)) == (
                size if a == b else 0)
            orthogonal_pairs += 1
    rank_hist = {}
    for difference in range(size):
        r = gf2_rank(difference, rows, cols)
        rank_hist[str(r)] = rank_hist.get(str(r), 0) + 1
        assert sum(character(a, difference) for a in q) == rank_kernel(rows, cols, r)
        assert sum(character(a, difference) for a in masks) == low_kernel(
            width, difference.bit_count())
    # Ordered pair check independent of the rank/distance multiplicity formula.
    for x in range(size):
        for y in range(size):
            assert sum(character(a, x) * character(a, y) for a in q) == rank_kernel(
                rows, cols, gf2_rank(x ^ y, rows, cols))
    return {"shape": [rows, cols], "matrices": size, "rank_one_queries": len(q),
            "low_degree_nonconstant_dimension": len(masks),
            "orthogonality_pairs": orthogonal_pairs,
            "ordered_pair_kernel_checks": size * size,
            "rank_histogram": rank_hist,
            "rank_kernel_values": [rank_kernel(rows, cols, r)
                                   for r in range(min(rows, cols) + 1)],
            "hamming_kernel_values": [low_kernel(width, h) for h in range(width + 1)]}


def kernel_obstruction() -> dict:
    # The 2x2 missing direction has squared value 10-9=1 at EVERY source.
    source_values = {rank_kernel(2, 2, r) for r in (1, 2)}
    code_values = {low_kernel(4, h) for h in range(1, 5)}
    products = sorted({c - u for c in code_values for u in source_values})
    compatible = sorted(set(products) & {-1, 1})
    assert products == [-3, 1, 5] and compatible == [1]
    # All z(x)z(y)=+1 makes z constant, but the complement is orthogonal to 1.
    return {"source_non_diagonal_kernel": sorted(source_values),
            "encoded_non_diagonal_kernel": sorted(code_values),
            "possible_complement_products": products,
            "unit_magnitude_compatible_products": compatible,
            "complement_dimension": 1, "complement_pointwise_square": 1,
            "forced_constant_contradicts_zero_mean": True,
            "conclusion": "MINIMUM_2X2_STORAGE_IS_FIVE_BITS_FOR_TWO_ADAPTIVE_BIT_PROBES",
            "proof_kind": "DERIVED; exact kernel checks support, not replace, the proof",
            "permutations_enumerated": 0}


def encode_five_bits(matrix: int) -> tuple[int, ...]:
    if not 0 <= matrix < 16:
        raise ValueError("2x2 binary matrix required")
    return tuple((matrix >> i) & 1 for i in range(4)) + (matrix.bit_count() & 1,)


def decode_five_bits(encoded: tuple[int, ...], mask: int) -> tuple[int, int]:
    if len(encoded) != 5 or any(x not in (0, 1) for x in encoded):
        raise ValueError("five binary cells required")
    if mask == 0:
        return 0, 0
    if mask not in queries(2, 2):
        raise ValueError("not a registered nonzero rank-one query")
    if mask == 15:
        return encoded[4], 1
    indices = [i for i in range(4) if mask >> i & 1]
    assert len(indices) <= 2
    return sum(encoded[i] for i in indices) & 1, len(indices)


def controls() -> dict:
    exact = reads = 0
    for matrix in range(16):
        code = encode_five_bits(matrix)
        for q in queries(2, 2):
            value, count = decode_five_bits(code, q)
            assert value == ((matrix & q).bit_count() & 1)
            exact += 1
            reads += count
    # Dynamic POSITIVE CONTROL over GF(2), NOT a checkpoint or native float lift.
    transitions = 0
    for matrix in range(16):
        for initial in range(4):
            state = reference = initial
            for _ in range(128):
                nxt = 0
                for row in range(2):
                    q = outer_mask(1 << row, state, 2, 2)
                    value, _ = decode_five_bits(encode_five_bits(matrix), q)
                    nxt |= value << row
                direct = 0
                for row in range(2):
                    value = sum(((matrix >> (row * 2 + col)) & 1)
                                * ((reference >> col) & 1) for col in range(2)) & 1
                    direct |= value << row
                state, reference = nxt, direct
                assert state == reference
                transitions += 1
    # A one-bit mutation MUST corrupt at least one legal answer for each cell.
    caught = []
    zero = encode_five_bits(0)
    for cell in range(5):
        corrupted = list(zero)
        corrupted[cell] ^= 1
        witnesses = [q for q in queries(2, 2) if decode_five_bits(tuple(corrupted), q)[0] != 0]
        assert witnesses
        caught.append({"cell": cell, "query": witnesses[0]})
    return {"exact_query_cases": exact, "logical_bit_reads": reads,
            "worst_query_probes": 2, "dynamic_gf2_state_transitions": transitions,
            "state_mismatches": 0, "fault_injection_witnesses": caught,
            "stored_bits": 5, "raw_bits": 4, "storage_expansion": "5/4",
            "physical_speedup": "NOT_MEASURED; both small encodings fit one byte",
            "scope": "TOY_CONSTRUCTION; no Transformer or BF16 capability"}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def native_boundary() -> dict:
    cases = 0
    for rows, cols in ((2, 2), (2, 3)):
        for matrix in range(1 << (rows * cols)):
            for left in range(1, 1 << rows):
                for right in range(1, 1 << cols):
                    integer = 0
                    ordered = f32(0.0)
                    for i in range(rows):
                        for j in range(cols):
                            term = ((matrix >> (i * cols + j)) & 1) * ((left >> i) & 1) * ((right >> j) & 1)
                            integer += term
                            ordered = f32(ordered + term)
                    assert ordered == integer
                    assert (integer & 1) == ((matrix & outer_mask(left, right, rows, cols)).bit_count() & 1)
                    cases += 1
    a, b, c = f32(2**24), f32(1), f32(-2**24)
    native_left = f32(f32(a + b) + c)
    alternate = f32(a + f32(b + c))
    assert native_left == 0.0 and alternate == 1.0
    return {"small_binary_ordered_float32_cases": cases,
            "parity_does_not_determine_native_sum": {
                "matrices": [0, 9], "left": 3, "right": 3,
                "same_parities": [0, 0], "different_sums": [0, 2]},
            "reassociation_counterexample": {"bf16_representable_terms": [2**24, 1, -(2**24)],
                                            "left_fp32": native_left, "right_fp32": alternate},
            "full_native_q4_bf16_transformer_lift": "NOT_CONSTRUCTED",
            "note": "binary subcase agreement is one-way only; parity output is not a numerical dot"}


def budget() -> dict:
    # Reproduce the earlier favorable DIAGNOSTIC, not a hardware lower bound.
    nominal = Fraction(16, 1350)
    extra = Fraction(17, 1350)
    assert nominal == Fraction(8, 675)
    return {"scope": "INHERITED_DIAGNOSTIC_ONLY_NOT_NATIVE_EXECUTOR",
            "source": "25x108 binary matrix; 50 64-bit cells; two logical word probes",
            "encoded_cold_bytes": 400, "two_word_bytes": 16,
            "favorable_q4_denominator_bytes": 1350,
            "nominal_fraction": str(nominal), "nominal_budget_slack_bytes": 0,
            "conditional_one_additional_byte_fraction": str(extra),
            "conditional_one_additional_byte_over_budget_factor": str(extra / nominal),
            "not_asserted": "one extra physical byte is NOT proved mandatory; native lifting cost unknown",
            "target_ssd_pcie_latency": "NOT_TESTED"}


def build_result() -> dict:
    result = {"schema": "vortex-global-decoder-kernel-audit-v1", "date": "2026-09-05",
              "base_sha": BASE_SHA, "evidence_ceiling": "E1",
              "new_result": kernel_obstruction(),
              "kernel_controls": [verify_kernel(2, 2), verify_kernel(2, 3)],
              "positive_and_fault_controls": controls(), "native_boundary": native_boundary(),
              "budget": budget(),
              "inherited_degree_check_2x3_six_bits": {
                  "all_rank_at_most_two": 64, "degree_at_most_four_capacity": sum(math.comb(6, j) for j in range(5)),
                  "status": "ALREADY_REJECTED_BY_PRIOR_RANK_AMPLIFIED_DEGREE_GATE_NOT_NEW"},
              "overall": {"decision": "SCOPED_2X2_MINIMUM_PROVED_NO_EXECUTOR_PROMOTED",
                          "seven_bit_2x3_problem": "UNRESOLVED",
                          "next_gate": "NON_ENTRYWISE_REDUNDANT_GLOBAL_CODE_WITH_NATIVE_SUCCESSOR_STATE_AND_PAID_COSTS",
                          "405b_execution": "NOT_TESTED", "physical_8gib": "NOT_TESTED",
                          "same_machine_4b_latency": "NOT_TESTED", "general_impossibility": "NOT_ESTABLISHED"}}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["deterministic_core_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
