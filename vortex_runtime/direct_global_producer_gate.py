"""Finite gates for the direct global GF(2)/native producer frontier.

The module implements three pieces frozen in
``experiments/direct_global_producer_20260909/PREREGISTRATION.md``:

* the information/compute screen for a reconstructive global erasure code;
* an exact FP32 balanced-reduction gadget whose mathematical sum is zero while
  its native rounded result carries an arbitrary GF(2) matrix-vector bit/count;
* a global largest-fiber route-span gate for simultaneous full-MatVec queries
  with arbitrary nonlinear 64-bit cells, plus an explicit arbitrary-GF(2)
  elimination producer used as a constructive control rather than as a claimed
  fast solution.

Nothing in this module is a target-hardware result.  The FP32 gadget models an
explicit balanced round-to-nearest-even tree; it is not asserted to be the
reduction ABI of every HF/CUDA kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from typing import Iterable, Sequence

from vortex_runtime.lower_bound_audit import llama_405b_tensor_plan
from vortex_runtime.sparse_functional_dictionary_frontier import (
    GLOBAL_ADVICE_BITS,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)


WORD_BITS = 64
FAVORABLE_Q4_LANES = 4
REGISTERED_P50_FRACTION = Fraction(8, 675)


def reconstructive_code_screen(source_bits_per_coefficient: int) -> dict[str, object]:
    """Best possible raw-read fraction for exact whole-source reconstruction.

    If ``X`` has ``N`` arbitrary source bits and the compiled advice has at most
    ``R`` bits, injectivity of ``(advice, additionally-read-source-bits)`` forces
    at least ``N-R`` source bits on some source whenever the runtime reconstructs
    all of ``X`` exactly.  This is deliberately favorable: addressing, parity
    reads, decode work and metadata are free in the lower screen.
    """

    if source_bits_per_coefficient <= 0:
        raise ValueError("source bit width must be positive")
    source_bits = REGISTERED_NON_EMBEDDING_COEFFICIENTS * source_bits_per_coefficient
    removable = min(source_bits, GLOBAL_ADVICE_BITS)
    minimum_raw = source_bits - removable
    removed_fraction = Fraction(removable, source_bits)
    return {
        "source_bits_per_coefficient": source_bits_per_coefficient,
        "source_bits": source_bits,
        "global_advice_bits": GLOBAL_ADVICE_BITS,
        "minimum_worst_case_additional_source_bits": minimum_raw,
        "maximum_source_read_removed_fraction": str(removed_fraction),
        "maximum_source_read_removed_fraction_decimal": float(removed_fraction),
        "minimum_source_read_fraction_decimal": float(Fraction(minimum_raw, source_bits)),
        "original_dense_arithmetic_fraction_retained": 1.0,
        "passes_90_percent_read_removal": removed_fraction >= Fraction(9, 10),
        "passes_90_percent_arithmetic_removal": False,
        "rejected_as_core": True,
    }


def _f32(value: float) -> float:
    """Round one Python float to IEEE binary32, ties-to-even."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def balanced_f32_sum(values: Sequence[float]) -> float:
    """Pairwise balanced FP32 sum with zero padding to a power of two."""

    if not values:
        return 0.0
    work = [_f32(value) for value in values]
    size = 1 << (len(work) - 1).bit_length()
    work.extend([0.0] * (size - len(work)))
    while len(work) > 1:
        work = [f32_add(work[index], work[index + 1]) for index in range(0, len(work), 2)]
    return work[0]


def zero_exact_sum_rounding_gadget(bit: int) -> tuple[float, ...]:
    """Eight leaves with exact sum zero and balanced FP32 result ``-bit``."""

    if bit not in (0, 1):
        raise ValueError("gadget bit must be binary")
    anchor = float(1 << 24)
    b = float(bit)
    return (anchor, b, -anchor, 0.0, -b, 0.0, 0.0, 0.0)


def rounding_gadget_audit(bit: int) -> dict[str, object]:
    leaves = zero_exact_sum_rounding_gadget(bit)
    exact_integer_sum = int(sum(int(value) for value in leaves))
    rounded = balanced_f32_sum(leaves)
    return {
        "bit": bit,
        "exact_integer_sum": exact_integer_sum,
        "balanced_fp32_result": rounded,
        "expected_result": -bit,
        "matches": rounded == float(-bit),
    }


def rounding_encoded_row(row_bits: Sequence[int], query_bits: Sequence[int]) -> dict[str, object]:
    """Encode ``row dot query`` solely in balanced FP32 rounding error.

    For every coordinate, the product bit ``row_j & query_j`` chooses one
    zero-exact-sum eight-leaf gadget.  Concatenating gadgets is already aligned
    to eight-leaf subtrees.  Pad the number of gadgets to a power of two so the
    upper balanced tree combines their integer ``-bit`` results exactly.
    """

    if len(row_bits) != len(query_bits) or not row_bits:
        raise ValueError("row/query widths must match and be nonempty")
    if any(bit not in (0, 1) for bit in row_bits) or any(bit not in (0, 1) for bit in query_bits):
        raise ValueError("row/query must be binary")
    products = [left & right for left, right in zip(row_bits, query_bits)]
    padded = 1 << (len(products) - 1).bit_length()
    products.extend([0] * (padded - len(products)))
    leaves: list[float] = []
    for bit in products:
        leaves.extend(zero_exact_sum_rounding_gadget(bit))
    rounded = balanced_f32_sum(leaves)
    popcount = sum(products)
    exact_sum = sum(int(value) for value in leaves)
    return {
        "width": len(row_bits),
        "padded_gadgets": padded,
        "exact_integer_sum": exact_sum,
        "balanced_fp32_result": rounded,
        "expected_negative_popcount": -popcount,
        "parity": popcount & 1,
        "decoded_parity_from_rounded_result": int(-rounded) & 1,
        "matches": rounded == float(-popcount),
    }


@dataclass(frozen=True)
class MatrixBlock:
    name: str
    rows: int
    columns: int


def registered_non_embedding_blocks() -> tuple[MatrixBlock, ...]:
    blocks: list[MatrixBlock] = []
    for spec in llama_405b_tensor_plan():
        if spec.name == "embedding":
            continue
        blocks.extend(
            MatrixBlock(f"{spec.name}[{index}]", spec.rows, spec.columns)
            for index in range(spec.count)
        )
    if len(blocks) != 883:
        raise AssertionError("registered non-embedding matrix count drifted")
    if sum(block.rows * block.columns for block in blocks) != REGISTERED_NON_EMBEDDING_COEFFICIENTS:
        raise AssertionError("registered non-embedding coefficient count drifted")
    return tuple(blocks)


def vector_route_minimum_span_per_query(blocks: Sequence[MatrixBlock]) -> int:
    """Span of all scalar output masks for one simultaneous full-Mv tuple."""

    if not blocks:
        raise ValueError("at least one block is required")
    return sum(block.rows for block in blocks)


def vector_route_capacity_dimension(
    blocks: Sequence[MatrixBlock], span_budget: int
) -> dict[str, object]:
    """Maximum sum of right-factor span dimensions inside one route.

    A nonempty route containing simultaneous queries for every block needs
    ``k_i>=1``.  Its component coefficient masks span exactly

        ``sum_i rows_i * k_i``.

    To maximize query-family exponent for a fixed span budget, every additional
    right-space dimension has unit benefit and costs ``rows_i``.  Greedy fill by
    increasing row count is therefore exact for the relaxed exponent
    ``sum k_i``.
    """

    if not blocks or span_budget < 0:
        raise ValueError("invalid route-capacity arguments")
    mandatory_span = sum(block.rows for block in blocks)
    if span_budget < mandatory_span:
        return {
            "feasible_nonempty_route": False,
            "mandatory_span": mandatory_span,
            "span_budget": span_budget,
            "sum_right_dimensions_upper": 0,
        }
    right_dimensions = len(blocks)
    remaining = span_budget - mandatory_span
    extras = sorted((block.rows, block.columns - 1) for block in blocks)
    allocated_extra = 0
    for row_cost, capacity in extras:
        if remaining < row_cost:
            break
        take = min(capacity, remaining // row_cost)
        allocated_extra += take
        remaining -= take * row_cost
        if take < capacity:
            break
    right_dimensions += allocated_extra
    return {
        "feasible_nonempty_route": True,
        "mandatory_span": mandatory_span,
        "span_budget": span_budget,
        "allocated_extra_right_dimensions": allocated_extra,
        "sum_right_dimensions_upper": right_dimensions,
        "unused_span_budget": remaining,
    }


def simultaneous_query_log2_lower(blocks: Sequence[MatrixBlock]) -> int:
    """Proof-safe integer lower exponent for simultaneous nonzero tuples.

    The exact count is ``prod_i (2**n_i - 1)``.  Since every registered
    ``n_i`` is at least 16,384 and there are only 883 blocks,

        ``prod_i (1 - 2**(-n_i)) > 1/2``.

    Hence the tuple count is strictly greater than ``2**(sum n_i - 1)``.
    Returning ``sum n_i - 1`` avoids floating-point rounding in the proof gate.
    """

    if not blocks:
        raise ValueError("at least one block is required")
    total_columns = sum(block.columns for block in blocks)
    minimum_columns = min(block.columns for block in blocks)
    if len(blocks) >= (1 << (minimum_columns - 2)):
        raise AssertionError("finite nonzero-tuple product witness drifted")
    return total_columns - 1


def global_vector_route_gate(
    *,
    word_bits: int = WORD_BITS,
    cells: int | None = None,
    target_fraction: Fraction = REGISTERED_P50_FRACTION,
) -> dict[str, object]:
    """Global nonlinear/adaptive route-span necessity for full-Mv tuples.

    Largest-fiber recursion is applied to the *joint output vector* rather than
    to one matrix.  On one final route all scalar answer masks have total linear
    span at most ``t*w``.  This permits globally mixed nonlinear cells and does
    not divide advice by matrix count.
    """

    if word_bits <= 0:
        raise ValueError("word width must be positive")
    blocks = registered_non_embedding_blocks()
    source_bits = REGISTERED_NON_EMBEDDING_COEFFICIENTS
    if cells is None:
        cells = math.ceil((source_bits + GLOBAL_ADVICE_BITS) / word_bits)
    if cells <= 0:
        raise ValueError("cell count must be positive")
    mandatory_span = vector_route_minimum_span_per_query(blocks)
    minimum_single_tuple_probes = math.ceil(mandatory_span / word_bits)
    query_log2_lower = simultaneous_query_log2_lower(blocks)
    # ``S**t <= 2**(ceil(log2 S)*t)`` is deliberately favorable to the
    # candidate and keeps the coverage rejection arithmetic integer/proof-safe.
    route_address_bits_upper = math.ceil(math.log2(cells))

    # Necessary route-cover condition using the favorable relaxed bound
    # |route queries| < 2^(sum k_i).  Search only the relevant finite interval.
    low = minimum_single_tuple_probes
    high = max(low, math.ceil(query_log2_lower / route_address_bits_upper) + low)

    def covers(probes: int) -> bool:
        capacity = vector_route_capacity_dimension(blocks, probes * word_bits)
        if not capacity["feasible_nonempty_route"]:
            return False
        exponent_upper = (
            probes * route_address_bits_upper
            + int(capacity["sum_right_dimensions_upper"])
        )
        return exponent_upper >= query_log2_lower

    while not covers(high):
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if covers(middle):
            high = middle
        else:
            low = middle + 1
    minimum_cover_probes = low
    capacity_at_min = vector_route_capacity_dimension(blocks, minimum_cover_probes * word_bits)
    capacity_before = vector_route_capacity_dimension(blocks, (minimum_cover_probes - 1) * word_bits)
    exponent_at_min = (
        minimum_cover_probes * route_address_bits_upper
        + int(capacity_at_min["sum_right_dimensions_upper"])
    )
    exponent_before = (
        (minimum_cover_probes - 1) * route_address_bits_upper
        + int(capacity_before["sum_right_dimensions_upper"])
    )

    q4_payload_bits = FAVORABLE_Q4_LANES * source_bits
    target_bits = target_fraction * q4_payload_bits
    target_words_floor = int(target_bits // word_bits)
    lower_payload_bits = minimum_cover_probes * word_bits
    return {
        "model": "GLOBAL_NONLINEAR_ADAPTIVE_SIMULTANEOUS_FULL_MV_ROUTE_GATE",
        "source_binary_bits": source_bits,
        "global_advice_bits": GLOBAL_ADVICE_BITS,
        "cells": cells,
        "word_bits": word_bits,
        "matrix_blocks": len(blocks),
        "sum_output_rows": mandatory_span,
        "sum_input_columns": sum(block.columns for block in blocks),
        "single_tuple_minimum_probes": minimum_single_tuple_probes,
        "simultaneous_query_log2_strictly_greater_than": query_log2_lower,
        "route_address_bits_upper_per_probe": route_address_bits_upper,
        "minimum_route_cover_probes_favorable": minimum_cover_probes,
        "route_capacity_at_minimum": capacity_at_min,
        "route_capacity_before_minimum": capacity_before,
        "coverage_log2_upper_at_minimum": exponent_at_min,
        "coverage_log2_upper_before_minimum": exponent_before,
        "previous_probe_count_is_proof_rejected": exponent_before < query_log2_lower,
        "minimum_probe_count_is_only_count_feasible": exponent_at_min >= query_log2_lower,
        "minimum_route_cover_payload_bits": lower_payload_bits,
        "registered_target_words_floor": target_words_floor,
        "registered_target_payload_bits_fraction": str(target_fraction),
        "lower_probes_over_target_words": minimum_cover_probes / target_words_floor,
        "rejects_registered_target": minimum_cover_probes > target_words_floor,
        "globally_mixed_nonlinear_cells_allowed": True,
        "fully_value_adaptive_addresses_allowed": True,
        "canonical_cell_pool_scope": "binary source cells plus 8 GiB advice; expanded cold sidecars are outside this substitution",
        "causal_reachability_of_complete_tuple_family_proved": False,
        "native_finite_word_lift_proved": False,
    }


@dataclass(frozen=True)
class EliminationOp:
    destination: int
    source: int


@dataclass(frozen=True)
class GF2EliminationProgram:
    rows: int
    columns: int
    reduced_rows: tuple[int, ...]
    operations: tuple[EliminationOp, ...]


def compile_gf2_elimination(row_masks: Sequence[int], columns: int) -> GF2EliminationProgram:
    """Finite exact arbitrary-GF(2) row-elimination producer.

    The compiler performs deterministic Gauss-Jordan row elimination while
    recording each row XOR.  Runtime first evaluates the reduced matrix and then
    reverses those same row operations on the output vector.  It is a concrete
    arbitrary-matrix producer, but the recorded row-operation stream can be
    quadratic and is therefore only a constructive control for P3.
    """

    if columns <= 0 or not row_masks:
        raise ValueError("matrix must be nonempty")
    limit = 1 << columns
    if any(row < 0 or row >= limit for row in row_masks):
        raise ValueError("row mask outside column width")
    rows = list(row_masks)
    ops: list[EliminationOp] = []
    pivot_row = 0
    for column in range(columns):
        pivot = next((r for r in range(pivot_row, len(rows)) if (rows[r] >> column) & 1), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            # A swap is three XOR row operations, preserving a single runtime
            # operation alphabet and making every cost explicit.
            rows[pivot_row] ^= rows[pivot]
            ops.append(EliminationOp(pivot_row, pivot))
            rows[pivot] ^= rows[pivot_row]
            ops.append(EliminationOp(pivot, pivot_row))
            rows[pivot_row] ^= rows[pivot]
            ops.append(EliminationOp(pivot_row, pivot))
        for row in range(len(rows)):
            if row != pivot_row and ((rows[row] >> column) & 1):
                rows[row] ^= rows[pivot_row]
                ops.append(EliminationOp(row, pivot_row))
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return GF2EliminationProgram(len(row_masks), columns, tuple(rows), tuple(ops))


def _parity(value: int) -> int:
    return value.bit_count() & 1


def execute_gf2_elimination(program: GF2EliminationProgram, query_mask: int) -> tuple[int, ...]:
    """Return the original matrix-vector result using the compiled program."""

    if query_mask < 0 or query_mask >= (1 << program.columns):
        raise ValueError("query mask outside column width")
    values = [_parity(row & query_mask) for row in program.reduced_rows]
    for op in reversed(program.operations):
        values[op.destination] ^= values[op.source]
    return tuple(values)


def direct_gf2_reference(row_masks: Sequence[int], query_mask: int) -> tuple[int, ...]:
    return tuple(_parity(row & query_mask) for row in row_masks)


def elimination_program_cost(program: GF2EliminationProgram) -> dict[str, object]:
    dense_bit_products = program.rows * program.columns
    reduced_nonzero_coefficients = sum(row.bit_count() for row in program.reduced_rows)
    runtime_xors = len(program.operations)
    runtime_total_bit_ops = reduced_nonzero_coefficients + runtime_xors
    return {
        "rows": program.rows,
        "columns": program.columns,
        "compile_row_xors": len(program.operations),
        "runtime_reduced_matrix_bit_products": reduced_nonzero_coefficients,
        "runtime_reverse_row_xors": runtime_xors,
        "runtime_total_bit_ops": runtime_total_bit_ops,
        "direct_dense_bit_products": dense_bit_products,
        "runtime_operation_fraction": runtime_total_bit_ops / dense_bit_products,
        "removes_at_least_90_percent_operations": runtime_total_bit_ops <= dense_bit_products / 10,
        "program_metadata_bits_lower": len(program.operations)
        * 2
        * math.ceil(math.log2(max(program.rows, 2))),
    }


def lower_triangular_elimination_cost(width: int) -> dict[str, object]:
    """Closed cost for this compiler on unit lower-triangular all-one rows.

    The frozen pivot order performs one row XOR for every strict-lower entry and
    reduces the matrix to identity.  This is an adversary for the *declared
    elimination producer only*; the same structured matrix has other cheap
    algorithms, so this is not a circuit lower bound.
    """

    if width <= 0:
        raise ValueError("width must be positive")
    row_xors = width * (width - 1) // 2
    runtime = width + row_xors
    dense = width * width
    index_bits = math.ceil(math.log2(max(width, 2)))
    return {
        "width": width,
        "compile_row_xors": row_xors,
        "runtime_reduced_matrix_bit_products": width,
        "runtime_reverse_row_xors": row_xors,
        "runtime_total_bit_ops": runtime,
        "direct_dense_bit_products": dense,
        "runtime_operation_fraction": runtime / dense,
        "exact_operation_fraction": str(Fraction(runtime, dense)),
        "program_metadata_bits_lower": row_xors * 2 * index_bits,
        "removes_at_least_90_percent_operations": runtime * 10 <= dense,
        "scope": "fixed Gauss-Jordan producer only; not a lower bound for this matrix family",
    }


def derive_direct_global_producer_audit() -> dict[str, object]:
    p1 = {
        "binary": reconstructive_code_screen(1),
        "q4": reconstructive_code_screen(4),
        "bf16": reconstructive_code_screen(16),
    }
    p2 = {
        "bit0": rounding_gadget_audit(0),
        "bit1": rounding_gadget_audit(1),
        "native_model": "explicit balanced IEEE-FP32 RNE tree only",
        "exact_sum_alone_determines_native_result": False,
        "rounding_witness_contains_arbitrary_binary_matvec": True,
    }
    p3 = global_vector_route_gate()

    # Two concrete arbitrary-GF(2) programs.  The adversarial triangular family
    # is used only against this elimination compiler, not as a universal circuit
    # lower bound.
    controls: list[dict[str, object]] = []
    for width in (8, 16, 32, 64):
        row_masks = tuple((1 << (row + 1)) - 1 for row in range(width))
        program = compile_gf2_elimination(row_masks, width)
        cost = elimination_program_cost(program)
        controls.append({"width": width, **cost})

    target_elimination = lower_triangular_elimination_cost(16_384)

    return {
        "classification": "E0_DIRECT_GLOBAL_PRODUCER_FRONTIER",
        "principle_P1_reconstructive_global_code": p1,
        "principle_P2_exact_sum_rounding_witness": p2,
        "principle_P3_global_nonlinear_route_gate": p3,
        "P3_explicit_arbitrary_gf2_elimination_producer_controls": controls,
        "P3_elimination_producer_registered_width_adversary": target_elimination,
        "decision": (
            "REJECT_P1_RECONSTRUCTIVE_GLOBAL_CODE;"
            "REDUCE_P2_ROUNDING_WITNESS_TO_DIRECT_MATVEC;"
            "GLOBAL_P3_ROUTE_GATE_INSUFFICIENT_TO_REJECT_TARGET;"
            "EXPLICIT_GF2_ELIMINATION_PRODUCER_FAILS_90_PERCENT_GATE;"
            "KEEP_DIRECT_GLOBAL_NATIVE_PRODUCER_OPEN"
        ),
        "claim_boundary": {
            "arbitrary_checkpoint_native_producer_constructed": False,
            "arbitrary_binary_matrix_elimination_producer_constructed": True,
            "global_nonlinear_route_lower_bound_proves_impossibility": False,
            "complete_simultaneous_tuple_causal_reachability": False,
            "expanded_cold_sidecar_global_route_bound": False,
            "native_cuda_reduction_abi_proved": False,
            "target_405b_executed": False,
            "target_hardware_executed": False,
        },
        "obligations": {
            "O1": "OPEN",
            "O2": "OPEN",
            "O3": "OPEN",
            "O4": "OPEN",
            "O5": "OPEN",
            "O6": "PARTIAL_E0_REPRODUCIBLE",
        },
    }
