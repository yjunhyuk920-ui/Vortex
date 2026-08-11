"""Finite-word E0 audit for exact bilinear cross-residual queries.

The concrete constructor in this module is a block rank-one truth table over
GF(2).  It genuinely uses bounded words and query-dependent addresses:

    T[p,q][x,y] = x.T @ W[p,q] @ y (mod 2)

and one table entry is combined per matrix block.  The binary parity contract
is deliberately more favorable than signed Q4/BF16 accumulation.  Failure in
this contract rejects this constructor, not every nonlinear cell-probe data
structure for rank-one queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import random
from typing import Sequence


P50_TARGET_FRACTION = Fraction(8, 675)
REQUESTED_UNIVERSAL_FRACTION = Fraction(1, 40)
REGISTERED_COEFFICIENTS = 403_747_897_344
REGISTERED_Q4_BYTES = 201_873_948_672
REGISTERED_SERVICE_TOKENS = 20_000_000
NATURAL_WORD_BITS = 64
MAILMAN_ALPHABET_SIZE = 16
REFERENCE_SEED = 0xF17E_0AD

REJECT_DECISION = (
    "NO_FINITE_WORD_CONSTRUCTOR_SURVIVES_E0_KEEP_GENERAL_RANK_ONE_CELL_PROBE_OPEN"
)


@dataclass(frozen=True)
class TruthTableBlock:
    row_start: int
    column_start: int
    rows: int
    columns: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class CompiledTruthTable:
    matrix_rows: int
    matrix_columns: int
    row_block_size: int
    column_block_size: int
    blocks: tuple[TruthTableBlock, ...]


def _validate_binary_vector(vector: Sequence[int], *, name: str) -> None:
    if any(value not in (0, 1) for value in vector):
        raise ValueError(f"{name} must contain only binary values")


def _validate_binary_matrix(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix rows must have equal width")
    for row in matrix:
        _validate_binary_vector(row, name="matrix")
    return len(matrix), columns


def gf2_bilinear(
    left: Sequence[int],
    matrix: Sequence[Sequence[int]],
    right: Sequence[int],
) -> int:
    rows, columns = _validate_binary_matrix(matrix)
    if len(left) != rows or len(right) != columns:
        raise ValueError("bilinear shape mismatch")
    _validate_binary_vector(left, name="left")
    _validate_binary_vector(right, name="right")
    result = 0
    for row_index, row in enumerate(matrix):
        if not left[row_index]:
            continue
        for column_index, coefficient in enumerate(row):
            result ^= coefficient & right[column_index]
    return result


def _block_value(
    matrix: Sequence[Sequence[int]],
    row_start: int,
    column_start: int,
    rows: int,
    columns: int,
    left_pattern: int,
    right_pattern: int,
) -> int:
    result = 0
    for local_row in range(rows):
        left_bit = (left_pattern >> local_row) & 1
        if not left_bit:
            continue
        for local_column in range(columns):
            result ^= (
                matrix[row_start + local_row][column_start + local_column]
                & ((right_pattern >> local_column) & 1)
            )
    return result


def compile_block_truth_table(
    matrix: Sequence[Sequence[int]],
    *,
    row_block_size: int,
    column_block_size: int,
) -> CompiledTruthTable:
    rows, columns = _validate_binary_matrix(matrix)
    if row_block_size <= 0 or column_block_size <= 0:
        raise ValueError("block sizes must be positive")
    blocks: list[TruthTableBlock] = []
    for row_start in range(0, rows, row_block_size):
        local_rows = min(row_block_size, rows - row_start)
        for column_start in range(0, columns, column_block_size):
            local_columns = min(column_block_size, columns - column_start)
            values = []
            for right_pattern in range(1 << local_columns):
                for left_pattern in range(1 << local_rows):
                    values.append(
                        _block_value(
                            matrix,
                            row_start,
                            column_start,
                            local_rows,
                            local_columns,
                            left_pattern,
                            right_pattern,
                        )
                    )
            blocks.append(
                TruthTableBlock(
                    row_start=row_start,
                    column_start=column_start,
                    rows=local_rows,
                    columns=local_columns,
                    values=tuple(values),
                )
            )
    return CompiledTruthTable(
        matrix_rows=rows,
        matrix_columns=columns,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
        blocks=tuple(blocks),
    )


def _pattern(vector: Sequence[int], start: int, width: int) -> int:
    result = 0
    for offset in range(width):
        result |= vector[start + offset] << offset
    return result


def query_block_truth_table(
    compiled: CompiledTruthTable,
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    if (
        len(left) != compiled.matrix_rows
        or len(right) != compiled.matrix_columns
    ):
        raise ValueError("compiled query shape mismatch")
    _validate_binary_vector(left, name="left")
    _validate_binary_vector(right, name="right")
    result = 0
    for block in compiled.blocks:
        left_pattern = _pattern(left, block.row_start, block.rows)
        right_pattern = _pattern(right, block.column_start, block.columns)
        index = left_pattern | (right_pattern << block.rows)
        result ^= block.values[index]
    return result


def run_reference_controls(
    *, seed: int = REFERENCE_SEED, cases: int = 16
) -> dict[str, int | bool]:
    if cases <= 0:
        raise ValueError("cases must be positive")
    generator = random.Random(seed)
    exhaustive_queries = 0
    exact_matches = 0
    basis_recoveries = 0
    for case in range(cases):
        rows = 1 + case % 3
        columns = 1 + (case // 3) % 3
        matrix = tuple(
            tuple(generator.randrange(2) for _ in range(columns))
            for _ in range(rows)
        )
        compiled = compile_block_truth_table(
            matrix,
            row_block_size=1 + case % 2,
            column_block_size=1 + (case // 2) % 2,
        )
        for left_pattern in range(1 << rows):
            left = tuple((left_pattern >> index) & 1 for index in range(rows))
            for right_pattern in range(1 << columns):
                right = tuple(
                    (right_pattern >> index) & 1
                    for index in range(columns)
                )
                exhaustive_queries += 1
                if query_block_truth_table(compiled, left, right) == (
                    gf2_bilinear(left, matrix, right)
                ):
                    exact_matches += 1
        for row in range(rows):
            left = tuple(int(index == row) for index in range(rows))
            for column in range(columns):
                right = tuple(
                    int(index == column) for index in range(columns)
                )
                if query_block_truth_table(compiled, left, right) == (
                    matrix[row][column]
                ):
                    basis_recoveries += 1
    return {
        "seed": seed,
        "cases": cases,
        "exhaustive_queries": exhaustive_queries,
        "exact_matches": exact_matches,
        "basis_recoveries": basis_recoveries,
        "all_controls_pass": exact_matches == exhaustive_queries,
    }


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def minimum_block_dimensions(
    *,
    word_bits: int,
    operations_per_block: int = 2,
    target_fraction: Fraction = P50_TARGET_FRACTION,
) -> tuple[int, int, int]:
    if word_bits <= 0 or operations_per_block <= 0:
        raise ValueError("word and operation counts must be positive")
    if not 0 < target_fraction < 1:
        raise ValueError("target fraction must lie strictly between zero and one")
    operation_area = _ceil_fraction(
        Fraction(operations_per_block, 1) / target_fraction
    )
    traffic_area = _ceil_fraction(
        Fraction(word_bits, 4) / target_fraction
    )
    required_area = max(operation_area, traffic_area)
    best: tuple[tuple[int, int, int, int, int], int, int] | None = None
    for rows in range(1, required_area + 1):
        columns = (required_area + rows - 1) // rows
        area = rows * columns
        key = (
            rows + columns,
            -area,
            abs(rows - columns),
            rows,
            columns,
        )
        if best is None or key < best[0]:
            best = (key, rows, columns)
    assert best is not None
    return best[1], best[2], required_area


def derive_block_truth_table_plan(
    *,
    word_bits: int,
    target_fraction: Fraction = P50_TARGET_FRACTION,
) -> dict[str, object]:
    rows, columns, required_area = minimum_block_dimensions(
        word_bits=word_bits,
        target_fraction=target_fraction,
    )
    area = rows * columns
    perimeter = rows + columns
    operation_fraction = Fraction(2, area)
    ideal_bit_payload_fraction = Fraction(1, 4 * area)
    physical_word_fraction = Fraction(word_bits, 4 * area)
    table_q4_storage_fraction = Fraction(1 << perimeter, 4 * area)
    build_dense_equivalents = Fraction(1 << perimeter, area)
    amortized_build_fraction = (
        build_dense_equivalents / REGISTERED_SERVICE_TOKENS
    )
    table_bits_numerator = REGISTERED_COEFFICIENTS * (1 << perimeter)
    table_bits = (
        table_bits_numerator + area - 1
    ) // area
    table_bytes = (table_bits + 7) // 8
    address_bits = max(1, (table_bytes - 1).bit_length())
    return {
        "word_bits": word_bits,
        "target_fraction": str(target_fraction),
        "target_fraction_decimal": float(target_fraction),
        "rows": rows,
        "columns": columns,
        "area": area,
        "perimeter": perimeter,
        "required_area": required_area,
        "operation_fraction": str(operation_fraction),
        "operation_fraction_decimal": float(operation_fraction),
        "ideal_bit_payload_fraction": str(ideal_bit_payload_fraction),
        "ideal_bit_payload_fraction_decimal": float(
            ideal_bit_payload_fraction
        ),
        "physical_word_traffic_fraction": str(physical_word_fraction),
        "physical_word_traffic_fraction_decimal": float(
            physical_word_fraction
        ),
        "remaining_traffic_headroom": str(
            target_fraction - physical_word_fraction
        ),
        "table_q4_storage_fraction": str(table_q4_storage_fraction),
        "table_q4_storage_fraction_decimal": float(
            table_q4_storage_fraction
        ),
        "minimum_table_bytes": table_bytes,
        "minimum_table_gib": table_bytes / (1 << 30),
        "minimum_address_bits": address_bits,
        "build_dense_equivalents": str(build_dense_equivalents),
        "build_dense_equivalents_decimal": float(build_dense_equivalents),
        "amortized_build_fraction": str(amortized_build_fraction),
        "amortized_build_fraction_decimal": float(
            amortized_build_fraction
        ),
        "passes_query_operations": operation_fraction <= target_fraction,
        "passes_physical_word_traffic": (
            physical_word_fraction <= target_fraction
        ),
        "passes_one_checkpoint_storage_grant": (
            table_q4_storage_fraction <= 1
        ),
        "passes_build_amortization": (
            amortized_build_fraction <= target_fraction
        ),
        "word_can_address_table": address_bits <= word_bits,
    }


def derive_mailman_floor() -> dict[str, object]:
    maximum_chunk_symbols = 0
    while MAILMAN_ALPHABET_SIZE ** (maximum_chunk_symbols + 1) <= (
        REGISTERED_COEFFICIENTS
    ):
        maximum_chunk_symbols += 1
    operation_floor = Fraction(1, maximum_chunk_symbols)
    return {
        "alphabet_size": MAILMAN_ALPHABET_SIZE,
        "maximum_favorable_chunk_symbols": maximum_chunk_symbols,
        "operation_fraction_floor": str(operation_floor),
        "operation_fraction_floor_decimal": float(operation_floor),
        "passes_target": operation_floor <= P50_TARGET_FRACTION,
        "class": "STATIC_FINITE_ALPHABET_FACTORIZATION_F050",
    }


def derive_larsen_williams_boundary() -> dict[str, object]:
    dimension = 16_384
    word_bits = 64
    root_dimension = isqrt(dimension)
    root_word = isqrt(word_bits)
    probes = dimension * root_dimension // root_word
    redundant_bits = dimension * root_dimension * root_word
    leading_one_probe_bytes = probes * (word_bits // 8)
    q4_matrix_bytes = dimension * dimension // 2
    leading_one_fraction = Fraction(
        leading_one_probe_bytes, q4_matrix_bytes
    )
    return {
        "dimension": dimension,
        "word_bits": word_bits,
        "favorable_leading_one_probes": probes,
        "favorable_redundant_bits": redundant_bits,
        "favorable_leading_one_probe_bytes": leading_one_probe_bytes,
        "favorable_q4_relative_fraction": str(leading_one_fraction),
        "favorable_q4_relative_fraction_decimal": float(
            leading_one_fraction
        ),
        "theorem_output": "BOOLEAN_RECTANGLE_NONEMPTY",
        "exact_parity_count_or_signed_sum": "NOT SUPPLIED",
        "query_set_construction": "FREE_COMPUTATION_IN_CELL_PROBE_MODEL",
        "large_remainder_branch": "RETURNS_ONE_WITHOUT_NUMERICAL_VALUE",
        "vortex_constructor": "REJECTED",
    }


def derive_audit() -> dict[str, object]:
    controls = run_reference_controls()
    if not controls["all_controls_pass"]:
        raise AssertionError("finite-word truth-table controls failed")
    natural_plan = derive_block_truth_table_plan(
        word_bits=NATURAL_WORD_BITS
    )
    return {
        "classification": "E0_FINITE_WORD_DISCONTINUOUS_SOURCE_AUDIT",
        "decision": REJECT_DECISION,
        "registered_target": {
            "p50_fraction": str(P50_TARGET_FRACTION),
            "p50_fraction_decimal": float(P50_TARGET_FRACTION),
            "dense_reduction_required": float(1 / P50_TARGET_FRACTION),
            "coefficient_population": REGISTERED_COEFFICIENTS,
            "q4_bytes": REGISTERED_Q4_BYTES,
            "service_tokens": REGISTERED_SERVICE_TOKENS,
        },
        "block_rank_one_truth_table": {
            "query_equation": (
                "xor_(p,q) T[p,q][r_block_p,u_block_q]"
            ),
            "finite_semantics": "GF2_EXACT_PARITY_FAVORABLE_SUBCLASS",
            "entry_bits": 1,
            "operations_per_block": 2,
            "plans": [
                derive_block_truth_table_plan(word_bits=32),
                natural_plan,
            ],
            "requested_2_5_percent_plan": derive_block_truth_table_plan(
                word_bits=64,
                target_fraction=REQUESTED_UNIVERSAL_FRACTION,
            ),
            "decision": "REJECT_STORAGE_BUILD_AND_ADDRESS_EXPLOSION",
        },
        "mailman": derive_mailman_floor(),
        "larsen_williams_boolean_probe": (
            derive_larsen_williams_boundary()
        ),
        "broadword_full_scan": {
            "coefficient_payload_fraction": "1",
            "reason": (
                "packing reduces instructions but still reads every Q4 word"
            ),
            "decision": "REJECT_AS_INFORMATION_SOURCE",
        },
        "native_rounding_transducer": {
            "finite_state_is_not_free": True,
            "exhaustive_state_answer_table": "FORBIDDEN",
            "on_the_fly_transition_schedule": (
                "reads the same coefficient stream unless a new index is supplied"
            ),
            "general_rounding_data_structure_lower_bound": "NOT PROVED",
        },
        "reference_controls": controls,
        "claim_boundary": {
            "local_rank_one_truth_tables": "REJECTED",
            "mailman_and_broadword_routes": "REJECTED",
            "known_boolean_cell_probe_shortcut_as_numerical_source": "REJECTED",
            "native_rounding_as_free_information": "REJECTED",
            "general_nonlinear_rank_one_cell_probe_structure": "OPEN",
            "finite_word_universal_impossibility": "NOT PROVED",
            "universal_2_5_percent_guarantee": "NOT ESTABLISHED",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
    }
