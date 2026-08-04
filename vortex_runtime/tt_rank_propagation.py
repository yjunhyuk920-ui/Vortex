"""Strengthen TT/MPO bond lower bounds with frozen full-matrix rank evidence."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from vortex_runtime.tt_kronecker_reuse import (
    ReusedTensorTrainPlan,
    _account,
)


def is_full_matrix_or_transpose_cut(factors: Sequence[int]) -> bool:
    """Return true when a Kronecker cut is exactly W or W-transpose."""
    m1, m2, n1, n2 = (int(value) for value in factors)
    return (m1 == 1 and n2 == 1) or (m2 == 1 and n1 == 1)


def propagate_adjacent_bond_lower_bounds(
    physical_dimensions: Sequence[int], bond_lower_bounds: Sequence[int]
) -> tuple[int, ...]:
    """Apply exact adjacent TT-rank inequalities to a fixed point.

    For physical mode size ``d_k`` and adjacent TT ranks ``R_{k-1}, R_k``:

    ``R_k <= d_k * R_{k-1}`` and ``R_{k-1} <= d_k * R_k``.

    Therefore a lower bound on either side yields a lower bound on the other.
    """
    physical = tuple(int(value) for value in physical_dimensions)
    bonds = [int(value) for value in bond_lower_bounds]
    if len(physical) != len(bonds) + 1:
        raise ValueError("physical/bond length mismatch")
    changed = True
    while changed:
        changed = False
        ranks = [1, *bonds, 1]
        for index, dimension in enumerate(physical):
            right_lower = math.ceil(ranks[index] / dimension)
            if index < len(bonds) and bonds[index] < right_lower:
                bonds[index] = right_lower
                changed = True
            left_lower = math.ceil(ranks[index + 1] / dimension)
            if index > 0 and bonds[index - 1] < left_lower:
                bonds[index - 1] = left_lower
                changed = True
    return tuple(bonds)


def strengthen_with_full_matrix_rank(
    plan: ReusedTensorTrainPlan,
    *,
    matrix_rank_lower_bound: int,
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool,
) -> ReusedTensorTrainPlan:
    """Inject frozen EXP-058 rank and propagate all implied bond bounds."""
    bonds = list(plan.bond_rank_lower_bounds)
    records = [dict(record) for record in plan.cut_records]
    for record in records:
        if is_full_matrix_or_transpose_cut(record["factors"]):
            index = int(record["cut"]) - 1
            if matrix_rank_lower_bound > bonds[index]:
                bonds[index] = int(matrix_rank_lower_bound)
                record["rank_lower_bound"] = int(matrix_rank_lower_bound)
                record["source"] = (
                    "EXP-058 frozen full integer/rational matrix-rank certificate; "
                    "cut is W or W-transpose up to permutations"
                )

    propagated = propagate_adjacent_bond_lower_bounds(
        plan.mode_plan.physical_dimensions, bonds
    )
    for index, rank in enumerate(propagated):
        if rank > bonds[index]:
            records[index]["rank_lower_bound"] = int(rank)
            records[index]["source"] = (
                records[index]["source"]
                + "; strengthened by adjacent TT-rank inequalities"
            )
            records[index]["propagated_from_adjacent_bond"] = True

    rows, columns = plan.matrix_shape
    accounting = _account(
        rows=rows,
        columns=columns,
        physical=plan.mode_plan.physical_dimensions,
        bonds=propagated,
        bits_per_core=bits_per_core,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    return replace(
        plan,
        bond_rank_lower_bounds=propagated,
        cut_records=tuple(records),
        core_scalar_lower_bound=accounting["core_scalars"],
        baseline_operations=accounting["baseline_operations"],
        lower_bound_operations=accounting["lower_bound_operations"],
        baseline_storage_bytes=accounting["baseline_storage_bytes"],
        lower_bound_storage_bytes=accounting["lower_bound_storage_bytes"],
        baseline_query_bytes=accounting["baseline_query_bytes"],
        lower_bound_query_bytes=accounting["lower_bound_query_bytes"],
    )
