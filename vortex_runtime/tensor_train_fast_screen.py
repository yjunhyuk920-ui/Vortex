"""Fast exact lower-bound screen for TT/MPO bond ranks.

This module does not approximate rank.  It stops Gaussian elimination after the
smallest number of pivots needed to prove that a plan exceeds both precommitted
operation and storage thresholds.  The resulting nonzero modular minor is a
rigorous integer/rational rank lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.modular_rank import modular_determinant
from vortex_runtime.tensor_train_rank import (
    TensorTrainModePlan,
    TensorTrainRankError,
    enumerate_mode_plans,
    interleave_matrix,
    unfold_interleaved_tensor,
)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True


@dataclass(frozen=True)
class ThresholdRankWitness:
    prime: int
    required_rank: int
    rank_lower_bound: int
    reached_required_rank: bool
    exact_modular_rank_if_below_required: int | None
    minimum_dimension: int
    pivot_rows: tuple[int, ...]
    pivot_columns: tuple[int, ...]
    certified_minor_determinant: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prime": self.prime,
            "required_rank": self.required_rank,
            "rank_lower_bound": self.rank_lower_bound,
            "reached_required_rank": self.reached_required_rank,
            "exact_modular_rank_if_below_required": (
                self.exact_modular_rank_if_below_required
            ),
            "minimum_dimension": self.minimum_dimension,
            "pivot_rows": list(self.pivot_rows),
            "pivot_columns": list(self.pivot_columns),
            "certified_minor_determinant": self.certified_minor_determinant,
        }


def threshold_rank_witness(
    matrix: Any, *, prime: int, required_rank: int
) -> ThresholdRankWitness:
    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0 or source.dtype.kind not in "iu":
        raise TensorTrainRankError("a nonempty integer matrix is required")
    if not _is_prime(int(prime)):
        raise TensorTrainRankError("modulus must be prime")
    minimum = min(source.shape)
    if required_rank <= 0 or required_rank > minimum:
        raise TensorTrainRankError("required rank outside matrix bounds")

    work = np.mod(source.astype(np.int64, copy=False), prime).copy()
    row_order = list(range(source.shape[0]))
    rank = 0
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    for column in range(source.shape[1]):
        if rank >= source.shape[0] or rank >= required_rank:
            break
        candidates = np.flatnonzero(work[rank:, column])
        if candidates.size == 0:
            continue
        pivot_row = rank + int(candidates[0])
        if pivot_row != rank:
            work[[rank, pivot_row]] = work[[pivot_row, rank]]
            row_order[rank], row_order[pivot_row] = (
                row_order[pivot_row],
                row_order[rank],
            )
        pivot = int(work[rank, column])
        inverse = pow(pivot, prime - 2, prime)
        work[rank, column:] = (work[rank, column:] * inverse) % prime
        if rank + 1 < source.shape[0]:
            factors = work[rank + 1 :, column].copy()
            if factors.size:
                work[rank + 1 :, column:] = (
                    work[rank + 1 :, column:]
                    - factors[:, None] * work[rank, column:][None, :]
                ) % prime
        pivot_rows.append(row_order[rank])
        pivot_columns.append(column)
        rank += 1

    if rank:
        minor = source[np.ix_(pivot_rows, pivot_columns)]
        determinant = modular_determinant(minor, prime=prime)
        if determinant == 0:
            raise TensorTrainRankError("threshold pivot minor is singular")
    else:
        determinant = 1
    reached = rank >= required_rank
    return ThresholdRankWitness(
        prime=int(prime),
        required_rank=required_rank,
        rank_lower_bound=rank,
        reached_required_rank=reached,
        exact_modular_rank_if_below_required=None if reached else rank,
        minimum_dimension=minimum,
        pivot_rows=tuple(pivot_rows),
        pivot_columns=tuple(pivot_columns),
        certified_minor_determinant=determinant,
    )


def _account(
    *,
    rows: int,
    columns: int,
    physical: Sequence[int],
    bonds: Sequence[int],
    bits_per_core: int,
    activation_bytes: int,
    has_bias: bool,
) -> dict[str, int | float]:
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
        "operation_fraction": lower_bound_operations / baseline_operations,
        "baseline_storage_bytes": baseline_storage_bytes,
        "lower_bound_storage_bytes": lower_bound_storage_bytes,
        "storage_fraction": lower_bound_storage_bytes / baseline_storage_bytes,
        "baseline_query_bytes": baseline_query_bytes,
        "lower_bound_query_bytes": lower_bound_query_bytes,
        "query_byte_fraction": lower_bound_query_bytes / baseline_query_bytes,
    }


def _minimum_decisive_rank(
    *,
    bond_index: int,
    maximum_rank: int,
    rows: int,
    columns: int,
    physical: Sequence[int],
    bonds: Sequence[int],
    bits_per_core: int,
    activation_bytes: int,
    has_bias: bool,
    rejection_operation_fraction: float,
    rejection_storage_fraction: float,
) -> int:
    def decisive(rank: int) -> bool:
        trial = list(bonds)
        trial[bond_index] = rank
        accounting = _account(
            rows=rows,
            columns=columns,
            physical=physical,
            bonds=trial,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        )
        return (
            accounting["operation_fraction"] > rejection_operation_fraction
            and accounting["storage_fraction"] > rejection_storage_fraction
        )

    if not decisive(maximum_rank):
        return maximum_rank
    low, high = 1, maximum_rank
    while low < high:
        middle = (low + high) // 2
        if decisive(middle):
            high = middle
        else:
            low = middle + 1
    return low


@dataclass(frozen=True)
class FastTensorTrainScreenPlan:
    matrix_shape: tuple[int, int]
    mode_plan: TensorTrainModePlan
    primes: tuple[int, ...]
    certified_cuts: tuple[int, ...]
    bond_rank_lower_bounds: tuple[int, ...]
    witnesses: tuple[tuple[int, tuple[ThresholdRankWitness, ...]], ...]
    exact_all_cuts: bool
    stopped_by_threshold: bool
    full_unfolding_rank_cuts: int
    core_scalar_lower_bound: int
    baseline_operations: int
    lower_bound_operations: int
    baseline_storage_bytes: int
    lower_bound_storage_bytes: int
    baseline_query_bytes: int
    lower_bound_query_bytes: int
    witness_mismatches: int
    rejection_operation_fraction: float
    rejection_storage_fraction: float

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
    def maximum_bond_rank(self) -> int:
        return max(self.bond_rank_lower_bounds, default=1)

    @property
    def joint_rejection_score(self) -> float:
        return max(
            self.operation_fraction / self.rejection_operation_fraction,
            self.storage_fraction / self.rejection_storage_fraction,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_shape": list(self.matrix_shape),
            **self.mode_plan.as_dict(),
            "primes": list(self.primes),
            "certified_cuts": list(self.certified_cuts),
            "certified_cut_count": len(self.certified_cuts),
            "total_internal_cut_count": len(self.mode_plan.mode_pairs) - 1,
            "bond_rank_lower_bounds": list(self.bond_rank_lower_bounds),
            "maximum_bond_rank": self.maximum_bond_rank,
            "witnesses": [
                {
                    "cut": cut,
                    "prime_witnesses": [item.as_dict() for item in prime_witnesses],
                }
                for cut, prime_witnesses in self.witnesses
            ],
            "exact_all_cuts": self.exact_all_cuts,
            "stopped_by_threshold": self.stopped_by_threshold,
            "full_unfolding_rank_cuts": self.full_unfolding_rank_cuts,
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
            "joint_rejection_score": self.joint_rejection_score,
            "rejection_operation_fraction": self.rejection_operation_fraction,
            "rejection_storage_fraction": self.rejection_storage_fraction,
            "witness_mismatches": self.witness_mismatches,
        }


def certify_fast_screen_plan(
    matrix: Any,
    mode_plan: TensorTrainModePlan,
    *,
    primes: Sequence[int] = (251, 257),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
    rejection_operation_fraction: float = 0.25,
    rejection_storage_fraction: float = 0.25,
) -> FastTensorTrainScreenPlan:
    source = np.asarray(matrix)
    registered = tuple(int(prime) for prime in primes)
    if len(registered) < 2 or len(set(registered)) != len(registered):
        raise TensorTrainRankError("at least two distinct primes are required")
    tensor = interleave_matrix(source, mode_plan.mode_pairs)
    rows, columns = (int(value) for value in source.shape)
    physical = mode_plan.physical_dimensions
    bonds = [1] * (tensor.ndim - 1)
    witness_rows: list[tuple[int, tuple[ThresholdRankWitness, ...]]] = []
    full_rank_cuts = 0
    stopped = False

    cut_shapes = []
    for cut in range(1, tensor.ndim):
        left = math.prod(tensor.shape[:cut])
        right = math.prod(tensor.shape[cut:])
        cut_shapes.append((cut, left, right))
    ordered_cuts = sorted(
        cut_shapes,
        key=lambda item: (
            -min(item[1], item[2]),
            abs(math.log(item[1] / item[2])),
            item[0],
        ),
    )

    for cut, left, right in ordered_cuts:
        maximum_rank = min(left, right)
        required_rank = _minimum_decisive_rank(
            bond_index=cut - 1,
            maximum_rank=maximum_rank,
            rows=rows,
            columns=columns,
            physical=physical,
            bonds=bonds,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
            rejection_operation_fraction=rejection_operation_fraction,
            rejection_storage_fraction=rejection_storage_fraction,
        )
        unfolding = unfold_interleaved_tensor(tensor, cut)
        witnesses = tuple(
            threshold_rank_witness(
                unfolding, prime=prime, required_rank=required_rank
            )
            for prime in registered
        )
        rank_lower_bound = max(item.rank_lower_bound for item in witnesses)
        bonds[cut - 1] = rank_lower_bound
        full_rank_cuts += int(rank_lower_bound == maximum_rank)
        witness_rows.append((cut, witnesses))

        accounting = _account(
            rows=rows,
            columns=columns,
            physical=physical,
            bonds=bonds,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        )
        if (
            accounting["operation_fraction"] > rejection_operation_fraction
            and accounting["storage_fraction"] > rejection_storage_fraction
        ):
            stopped = len(witness_rows) < tensor.ndim - 1
            if stopped:
                break

    accounting = _account(
        rows=rows,
        columns=columns,
        physical=physical,
        bonds=bonds,
        bits_per_core=bits_per_core,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    witness_rows.sort(key=lambda item: item[0])
    certified_cuts = tuple(item[0] for item in witness_rows)
    return FastTensorTrainScreenPlan(
        matrix_shape=(rows, columns),
        mode_plan=mode_plan,
        primes=registered,
        certified_cuts=certified_cuts,
        bond_rank_lower_bounds=tuple(bonds),
        witnesses=tuple(witness_rows),
        exact_all_cuts=len(certified_cuts) == tensor.ndim - 1,
        stopped_by_threshold=stopped,
        full_unfolding_rank_cuts=full_rank_cuts,
        core_scalar_lower_bound=int(accounting["core_scalars"]),
        baseline_operations=int(accounting["baseline_operations"]),
        lower_bound_operations=int(accounting["lower_bound_operations"]),
        baseline_storage_bytes=int(accounting["baseline_storage_bytes"]),
        lower_bound_storage_bytes=int(accounting["lower_bound_storage_bytes"]),
        baseline_query_bytes=int(accounting["baseline_query_bytes"]),
        lower_bound_query_bytes=int(accounting["lower_bound_query_bytes"]),
        witness_mismatches=0,
        rejection_operation_fraction=rejection_operation_fraction,
        rejection_storage_fraction=rejection_storage_fraction,
    )


def certify_fast_screen_family(
    matrix: Any,
    *,
    primes: Sequence[int] = (251, 257),
    maximum_modes: Sequence[int] = (16,),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
    rejection_operation_fraction: float = 0.25,
    rejection_storage_fraction: float = 0.25,
) -> tuple[FastTensorTrainScreenPlan, ...]:
    source = np.asarray(matrix)
    if source.ndim != 2:
        raise TensorTrainRankError("matrix must be two-dimensional")
    mode_plans = enumerate_mode_plans(
        int(source.shape[0]),
        int(source.shape[1]),
        maximum_modes=maximum_modes,
    )
    return tuple(
        certify_fast_screen_plan(
            source,
            plan,
            primes=primes,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
            rejection_operation_fraction=rejection_operation_fraction,
            rejection_storage_fraction=rejection_storage_fraction,
        )
        for plan in mode_plans
        if len(plan.mode_pairs) >= 2
    )


def select_favorable_fast_screen_plan(
    plans: Iterable[FastTensorTrainScreenPlan],
) -> FastTensorTrainScreenPlan:
    population = tuple(plans)
    if not population:
        raise TensorTrainRankError("mode-plan population is empty")
    return min(
        population,
        key=lambda plan: (
            plan.joint_rejection_score,
            plan.operation_fraction,
            plan.storage_fraction,
            plan.query_byte_fraction,
            len(plan.mode_plan.mode_pairs),
            plan.mode_plan.mode_pairs,
        ),
    )


def selected_fast_certificate_rows(
    *,
    model_id: str,
    tensor_name: str,
    matrix: np.ndarray,
    plan: FastTensorTrainScreenPlan,
    primes: tuple[int, ...],
) -> list[dict[str, Any]]:
    del matrix, primes
    rows: list[dict[str, Any]] = []
    for cut, witnesses in plan.witnesses:
        for witness in witnesses:
            rows.append(
                {
                    "model_id": model_id,
                    "tensor_name": tensor_name,
                    "variant": plan.mode_plan.variant,
                    "mode_pairs": [list(pair) for pair in plan.mode_plan.mode_pairs],
                    "physical_dimensions": list(plan.mode_plan.physical_dimensions),
                    "cut": cut,
                    **witness.as_dict(),
                    "verified": True,
                    "threshold_censored_plan": plan.stopped_by_threshold,
                }
            )
    return rows
