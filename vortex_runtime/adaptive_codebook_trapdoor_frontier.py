"""Finite E0 screens for two adaptive exact-query proposals.

``OMEGA-NEARESTPAIR`` stores exact scalars for representative rank-one queries
and repairs only the matrix coordinates on which the requested and cached
query masks differ.  The literal-table Gate below grants free routing, free
codebook storage, one-bit answers, and one ideal operation per repaired
coefficient.  Its packing argument even permits arbitrary (not necessarily
rank-one or separately coded) representative masks.

``OMEGA-TRAPSHIFT`` subtracts a fast trapdoored mask from a shifted-matrix
answer.  Its identity is exact over a field, but the trapdoor accelerates only
the mask; it does not supply the shifted-matrix answer.

Neither screen is a lower bound for arbitrary compressed bilinear oracles.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math


GIB = 1 << 30
HOT_BITS = 8 * GIB * 8
REGISTERED_DIMENSION = 16_384
P50_TARGET_FRACTION = Fraction(8, 675)
DECISION = "REJECT_NEARESTPAIR_LITERAL_TABLE_AND_SOURCE_FREE_TRAPSHIFT"


def hamming_ball_volume(length: int, radius: int) -> int:
    """Return ``sum(j=0..radius, binom(length, j))`` exactly."""

    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 <= radius <= length:
        raise ValueError("radius must lie in [0, length]")
    return sum(math.comb(length, index) for index in range(radius + 1))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class PairTableGate:
    dimension: int
    repair_budget_cells: int
    packing_removal_radius: int
    packing_ball_volume: int
    minimum_literal_table_entries: int
    literal_table_log2_bits: float
    literal_table_address_bits: int
    hot_capacity_log2_bits: float

    @property
    def exponent_deficit_bits(self) -> float:
        return self.literal_table_log2_bits - self.hot_capacity_log2_bits

    @property
    def fits_hot_capacity(self) -> bool:
        return self.exponent_deficit_bits <= 0.0


def nearest_pair_literal_table_gate(
    *,
    dimension: int = REGISTERED_DIMENSION,
    target_fraction: Fraction = P50_TARGET_FRACTION,
    hot_bits: int = HOT_BITS,
) -> PairTableGate:
    """Apply a packing lower bound to a direct cached-scalar table.

    Restrict queries to ``r tensor 1``.  Two such query masks have matrix
    Hamming distance ``n * dist(r, r')``.  A greedy binary code with minimum
    distance greater than ``2R/n`` has at least

        2**n / Vol(n, floor(2R/n))

    members.  No radius-``R`` ball around *any* representative mask can cover
    two members, by the triangle inequality.  Thus a direct table that repairs
    at most ``R`` raw coefficients for every query needs at least that many
    cached scalar entries.  This grants joint routing and arbitrary centers;
    separate left/right codebooks are a restricted special case.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not Fraction(0) < target_fraction < Fraction(1):
        raise ValueError("target_fraction must lie strictly in (0, 1)")
    if hot_bits <= 0:
        raise ValueError("hot_bits must be positive")

    cells = dimension * dimension
    repair_budget = (target_fraction.numerator * cells) // target_fraction.denominator
    removal_radius = (2 * repair_budget) // dimension
    if removal_radius >= dimension:
        raise ValueError("repair budget is outside the packing proof range")
    volume = hamming_ball_volume(dimension, removal_radius)
    space_size = 1 << dimension
    entries = _ceil_div(space_size, volume)
    return PairTableGate(
        dimension=dimension,
        repair_budget_cells=repair_budget,
        packing_removal_radius=removal_radius,
        packing_ball_volume=volume,
        minimum_literal_table_entries=entries,
        literal_table_log2_bits=math.log2(entries),
        literal_table_address_bits=(entries - 1).bit_length(),
        hot_capacity_log2_bits=math.log2(hot_bits),
    )


def nearest_pair_scalar_reference(
    matrix: list[list[int]],
    left: list[int],
    right: list[int],
    left_codeword: list[int],
    right_codeword: list[int],
) -> tuple[int, int]:
    """Return the exact GF(2) answer and exact difference-repair read count."""

    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if (
        rows == 0
        or columns == 0
        or any(len(row) != columns for row in matrix)
        or len(left) != rows
        or len(left_codeword) != rows
        or len(right) != columns
        or len(right_codeword) != columns
    ):
        raise ValueError("matrix and query vectors must be nonempty and aligned")
    arrays = (left, right, left_codeword, right_codeword)
    if any(value not in (0, 1) for array in arrays for value in array):
        raise ValueError("reference vectors must be binary")
    if any(value not in (0, 1) for row in matrix for value in row):
        raise ValueError("reference matrix must be binary")

    base = 0
    for row_index, row_value in enumerate(left_codeword):
        if not row_value:
            continue
        for column_index, column_value in enumerate(right_codeword):
            if column_value:
                base ^= matrix[row_index][column_index]

    answer = base
    reads = 0
    for row_index in range(rows):
        for column_index in range(columns):
            requested = left[row_index] & right[column_index]
            cached = (
                left_codeword[row_index] & right_codeword[column_index]
            )
            if requested != cached:
                reads += 1
                answer ^= matrix[row_index][column_index]
    return answer, reads


def trapshift_reference(
    matrix: list[list[int]],
    mask: list[list[int]],
    vector: list[int],
    prime: int,
) -> tuple[list[int], list[int], list[int]]:
    """Evaluate ``(W+R)x - Rx = Wx`` over ``F_prime``."""

    if prime <= 1:
        raise ValueError("prime must be greater than one")
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if (
        rows == 0
        or columns == 0
        or len(mask) != rows
        or any(len(row) != columns for row in matrix)
        or any(len(row) != columns for row in mask)
        or len(vector) != columns
    ):
        raise ValueError("matrix, mask, and vector must be aligned")

    def matvec(value: list[list[int]]) -> list[int]:
        return [
            sum(cell * coordinate for cell, coordinate in zip(row, vector))
            % prime
            for row in value
        ]

    shifted = [
        [(a + b) % prime for a, b in zip(row, mask_row)]
        for row, mask_row in zip(matrix, mask)
    ]
    shifted_product = matvec(shifted)
    mask_product = matvec(mask)
    recovered = [
        (shifted_value - mask_value) % prime
        for shifted_value, mask_value in zip(shifted_product, mask_product)
    ]
    return recovered, shifted_product, mask_product


def derive_audit() -> dict[str, object]:
    pair = nearest_pair_literal_table_gate()
    return {
        "name": "adaptive_codebook_trapdoor_frontier",
        "decision": DECISION,
        "old_2_5_percent_assumption_used": False,
        "omega_nearestpair": {
            "model": "JOINT_REPRESENTATIVE_LITERAL_SCALAR_TABLE_EXACT_DIFFERENCE_REPAIR",
            "dimension": pair.dimension,
            "target_fraction": str(P50_TARGET_FRACTION),
            "repair_budget_cells": pair.repair_budget_cells,
            "packing_subfamily": "r tensor all_ones",
            "packing_removal_radius": pair.packing_removal_radius,
            "minimum_literal_table_log2_bits": pair.literal_table_log2_bits,
            "literal_table_address_bits": pair.literal_table_address_bits,
            "hot_capacity_log2_bits": pair.hot_capacity_log2_bits,
            "storage_exponent_deficit_bits": pair.exponent_deficit_bits,
            "fits_hot_capacity": pair.fits_hot_capacity,
            "favorable_grants": [
                "all 8 GiB assigned to one binary square",
                "arbitrary joint representative masks",
                "free codebook storage and nearest-codeword routing",
                "one-bit exact cached scalar per representative",
                "free table lookup and address computation",
                "one ideal operation per repaired coefficient",
                "all native numerical and system costs omitted",
            ],
            "decision": "REJECT_OMEGA_NEARESTPAIR_LITERAL_TABLE_BY_FINITE_COVERING_STORAGE_GATE",
        },
        "omega_trapshift": {
            "identity": "W*x = (W+R)*x - R*x over an associative field",
            "trapdoored_mask_product": "FAST",
            "shifted_arbitrary_matrix_product": "MISSING_SOURCE",
            "remaining_dense_fraction_if_computed_directly": 1.0,
            "sampled_trapdoor_matrix_is_not_an_arbitrary_checkpoint": True,
            "average_case_solver_supplied_by_trapdoor_paper": False,
            "native_bf16_fp32_order_preserved": False,
            "decision": "REJECT_OMEGA_TRAPSHIFT_AS_SOURCE_FREE_MASKING_IDENTITY",
        },
        "claim_boundary": {
            "general_compressed_bilinear_oracle_rejected": False,
            "joint_literal_representatives_granted": True,
            "joint_nonseparable_compressed_codebook_rejected": False,
            "linear_image_variant": "CONTAINED_BY_F055",
            "causal_transformer_query_reachability": "NOT TESTED",
            "trapdoored_generated_checkpoint_fast": "SUPPORTED_ONLY_FOR_SAMPLED_FAMILY",
            "target_achieved": False,
            "model_or_hardware_execution": False,
        },
    }
