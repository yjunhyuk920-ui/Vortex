"""Reuse frozen EXP-065 Kronecker ranks as exact TT/MPO bond lower bounds.

For interleaved row/column MPO modes, every internal TT cut is exactly the
Kronecker rearrangement with factors equal to the row/column prefix and suffix
products.  EXP-065 already measured every nontrivial ordered factorization of
the same deterministic Q4 matrices, so recomputing those modular ranks would
only duplicate frozen evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vortex_runtime.kronecker_rank import rearrange_kronecker
from vortex_runtime.tensor_train_rank import (
    TensorTrainModePlan,
    TensorTrainRankError,
    interleave_matrix,
    unfold_interleaved_tensor,
)


KroneckerKey = tuple[str, str, tuple[int, int, int, int]]


def tt_cut_factors(
    mode_pairs: Sequence[tuple[int, int]], cut: int
) -> tuple[int, int, int, int]:
    """Return the EXP-065 Kronecker factors represented by one TT cut."""
    pairs = tuple((int(row), int(column)) for row, column in mode_pairs)
    if cut <= 0 or cut >= len(pairs):
        raise TensorTrainRankError("cut must be internal")
    m1 = math.prod(pair[0] for pair in pairs[:cut])
    m2 = math.prod(pair[0] for pair in pairs[cut:])
    n1 = math.prod(pair[1] for pair in pairs[:cut])
    n2 = math.prod(pair[1] for pair in pairs[cut:])
    return int(m1), int(m2), int(n1), int(n2)


def tt_cut_unfolding(
    matrix: Any, mode_pairs: Sequence[tuple[int, int]], cut: int
) -> np.ndarray:
    """Materialize the TT prefix/suffix unfolding for a validation control."""
    tensor = interleave_matrix(matrix, mode_pairs)
    return unfold_interleaved_tensor(tensor, cut)


def validate_cut_equivalence(
    matrix: Any, mode_pairs: Sequence[tuple[int, int]], cut: int
) -> bool:
    """Check byte-exact equality with the EXP-065 rearrangement convention."""
    factors = tt_cut_factors(mode_pairs, cut)
    return np.array_equal(
        tt_cut_unfolding(matrix, mode_pairs, cut),
        rearrange_kronecker(
            matrix,
            m1=factors[0],
            m2=factors[1],
            n1=factors[2],
            n2=factors[3],
        ),
    )


def index_exp065_plan_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[KroneckerKey, Mapping[str, Any]]:
    """Index authoritative EXP-065 plan rows and reject duplicate conflicts."""
    indexed: dict[KroneckerKey, Mapping[str, Any]] = {}
    for row in rows:
        factors = tuple(int(value) for value in row["factors"])
        if len(factors) != 4:
            raise TensorTrainRankError("EXP-065 factor tuple must have length four")
        key: KroneckerKey = (
            str(row["model_id"]),
            str(row["tensor_name"]),
            factors,  # type: ignore[arg-type]
        )
        previous = indexed.get(key)
        if previous is not None and (
            int(previous["rank_lower_bound"]) != int(row["rank_lower_bound"])
            or tuple(previous["prime_ranks"]) != tuple(row["prime_ranks"])
        ):
            raise TensorTrainRankError("conflicting frozen EXP-065 plan rows")
        indexed[key] = row
    return indexed


def _account(
    *,
    rows: int,
    columns: int,
    physical: Sequence[int],
    bonds: Sequence[int],
    bits_per_core: int,
    activation_bytes: int,
    has_bias: bool,
) -> dict[str, int]:
    ranks = (1,) + tuple(int(value) for value in bonds) + (1,)
    core_scalars = sum(
        ranks[index] * int(physical[index]) * ranks[index + 1]
        for index in range(len(physical))
    )
    output_terms = rows + (rows if has_bias else 0)
    baseline_operations = rows * columns + output_terms
    lower_bound_operations = core_scalars + output_terms
    baseline_storage_bytes = (
        math.ceil(rows * columns * bits_per_core / 8)
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    metadata_bytes = 16 + len(physical) * 12 + len(bonds) * 4
    lower_bound_storage_bytes = (
        math.ceil(core_scalars * bits_per_core / 8)
        + rows * 4
        + (rows * 4 if has_bias else 0)
        + metadata_bytes
    )
    baseline_query_bytes = (
        math.ceil(rows * columns * bits_per_core / 8)
        + rows * columns * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    lower_bound_query_bytes = (
        math.ceil(core_scalars * bits_per_core / 8)
        + columns * activation_bytes
        + sum(bonds) * activation_bytes * 2
        + rows * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
        + metadata_bytes
    )
    return {
        "core_scalars": core_scalars,
        "baseline_operations": baseline_operations,
        "lower_bound_operations": lower_bound_operations,
        "baseline_storage_bytes": baseline_storage_bytes,
        "lower_bound_storage_bytes": lower_bound_storage_bytes,
        "baseline_query_bytes": baseline_query_bytes,
        "lower_bound_query_bytes": lower_bound_query_bytes,
    }


@dataclass(frozen=True)
class ReusedTensorTrainPlan:
    model_id: str
    tensor_name: str
    matrix_shape: tuple[int, int]
    mode_plan: TensorTrainModePlan
    bond_rank_lower_bounds: tuple[int, ...]
    cut_records: tuple[dict[str, Any], ...]
    mapped_cut_count: int
    unit_boundary_cut_count: int
    missing_nontrivial_cut_count: int
    source_witness_mismatches: int
    core_scalar_lower_bound: int
    baseline_operations: int
    lower_bound_operations: int
    baseline_storage_bytes: int
    lower_bound_storage_bytes: int
    baseline_query_bytes: int
    lower_bound_query_bytes: int

    @property
    def operation_fraction(self) -> float:
        return self.lower_bound_operations / self.baseline_operations

    @property
    def storage_fraction(self) -> float:
        return self.lower_bound_storage_bytes / self.baseline_storage_bytes

    @property
    def query_byte_fraction(self) -> float:
        return self.lower_bound_query_bytes / self.baseline_query_bytes

    @property
    def maximum_bond_rank_lower_bound(self) -> int:
        return max(self.bond_rank_lower_bounds, default=1)

    @property
    def meets_joint_p90_screen(self) -> bool:
        return self.operation_fraction <= 0.25 and self.storage_fraction <= 0.25

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tensor_name": self.tensor_name,
            "matrix_shape": list(self.matrix_shape),
            **self.mode_plan.as_dict(),
            "bond_rank_lower_bounds": list(self.bond_rank_lower_bounds),
            "maximum_bond_rank_lower_bound": self.maximum_bond_rank_lower_bound,
            "cut_records": list(self.cut_records),
            "mapped_cut_count": self.mapped_cut_count,
            "unit_boundary_cut_count": self.unit_boundary_cut_count,
            "missing_nontrivial_cut_count": self.missing_nontrivial_cut_count,
            "source_witness_mismatches": self.source_witness_mismatches,
            "core_scalar_lower_bound": self.core_scalar_lower_bound,
            "baseline_operations": self.baseline_operations,
            "lower_bound_operations": self.lower_bound_operations,
            "operation_fraction": self.operation_fraction,
            "baseline_storage_bytes": self.baseline_storage_bytes,
            "lower_bound_storage_bytes": self.lower_bound_storage_bytes,
            "storage_fraction": self.storage_fraction,
            "baseline_query_bytes": self.baseline_query_bytes,
            "lower_bound_query_bytes": self.lower_bound_query_bytes,
            "query_byte_fraction": self.query_byte_fraction,
            "meets_joint_p90_screen": self.meets_joint_p90_screen,
        }


def derive_reused_tt_plan(
    *,
    model_id: str,
    tensor_name: str,
    rows: int,
    columns: int,
    has_bias: bool,
    mode_plan: TensorTrainModePlan,
    exp065_index: Mapping[KroneckerKey, Mapping[str, Any]],
    bits_per_core: int = 4,
    activation_bytes: int = 4,
) -> ReusedTensorTrainPlan:
    """Derive conservative TT bond lower bounds from frozen EXP-065 rows."""
    bonds: list[int] = []
    cut_records: list[dict[str, Any]] = []
    mapped = 0
    unit_boundaries = 0
    missing = 0
    witness_mismatches = 0
    for cut in range(1, len(mode_plan.mode_pairs)):
        factors = tt_cut_factors(mode_plan.mode_pairs, cut)
        nontrivial = all(value > 1 for value in factors)
        source_row = exp065_index.get((model_id, tensor_name, factors))
        if source_row is not None:
            rank = int(source_row["rank_lower_bound"])
            mapped += 1
            witness_mismatches += int(source_row.get("witness_mismatches", 0))
            record = {
                "cut": cut,
                "factors": list(factors),
                "rank_lower_bound": rank,
                "prime_ranks": list(source_row["prime_ranks"]),
                "full_rearrangement_rank_proven": bool(
                    source_row["full_rearrangement_rank_proven"]
                ),
                "source": "EXP-065 frozen validated plan row",
            }
        elif nontrivial:
            # This should never occur because EXP-065 covered every ordered
            # nontrivial factorization.  Rank one is retained only to avoid an
            # unsound favorable overstatement before the correctness Gate fails.
            rank = 1
            missing += 1
            record = {
                "cut": cut,
                "factors": list(factors),
                "rank_lower_bound": rank,
                "source": "missing nontrivial EXP-065 row; fail-closed rank-one placeholder",
            }
        else:
            # EXP-065 deliberately excluded unit factors.  One is a universally
            # valid lower bound and deliberately favors the TT/MPO candidate.
            rank = 1
            unit_boundaries += 1
            record = {
                "cut": cut,
                "factors": list(factors),
                "rank_lower_bound": rank,
                "source": "unmeasured unit-boundary cut; universal favorable lower bound",
            }
        bonds.append(rank)
        cut_records.append(record)

    accounting = _account(
        rows=rows,
        columns=columns,
        physical=mode_plan.physical_dimensions,
        bonds=bonds,
        bits_per_core=bits_per_core,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    return ReusedTensorTrainPlan(
        model_id=model_id,
        tensor_name=tensor_name,
        matrix_shape=(rows, columns),
        mode_plan=mode_plan,
        bond_rank_lower_bounds=tuple(bonds),
        cut_records=tuple(cut_records),
        mapped_cut_count=mapped,
        unit_boundary_cut_count=unit_boundaries,
        missing_nontrivial_cut_count=missing,
        source_witness_mismatches=witness_mismatches,
        core_scalar_lower_bound=accounting["core_scalars"],
        baseline_operations=accounting["baseline_operations"],
        lower_bound_operations=accounting["lower_bound_operations"],
        baseline_storage_bytes=accounting["baseline_storage_bytes"],
        lower_bound_storage_bytes=accounting["lower_bound_storage_bytes"],
        baseline_query_bytes=accounting["baseline_query_bytes"],
        lower_bound_query_bytes=accounting["lower_bound_query_bytes"],
    )


def select_favorable_reused_plan(
    plans: Iterable[ReusedTensorTrainPlan],
) -> ReusedTensorTrainPlan:
    population = tuple(plans)
    if not population:
        raise TensorTrainRankError("reused TT plan population is empty")
    return min(
        population,
        key=lambda plan: (
            plan.operation_fraction,
            plan.storage_fraction,
            plan.query_byte_fraction,
            len(plan.mode_plan.mode_pairs),
            plan.mode_plan.mode_pairs,
        ),
    )
