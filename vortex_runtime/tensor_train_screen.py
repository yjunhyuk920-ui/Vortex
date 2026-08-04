"""Threshold-censored exact TT/MPO bond-rank screening.

The screen is conservative: unmeasured bond ranks remain at the universal
lower bound one.  Once both operation and storage lower bounds exceed the
precommitted rejection thresholds, later cuts cannot lower either bound, so
certification stops without weakening the rejection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.modular_rank import rank_certificate_mod_prime
from vortex_runtime.tensor_train_rank import (
    TensorTrainModePlan,
    TensorTrainRankError,
    enumerate_mode_plans,
    interleave_matrix,
    unfold_interleaved_tensor,
)


@dataclass(frozen=True)
class TensorTrainScreenPlan:
    matrix_shape: tuple[int, int]
    mode_plan: TensorTrainModePlan
    primes: tuple[int, ...]
    certified_cuts: tuple[int, ...]
    cut_prime_ranks: tuple[tuple[int, ...], ...]
    bond_rank_lower_bounds: tuple[int, ...]
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
            "cut_prime_ranks": [list(row) for row in self.cut_prime_ranks],
            "bond_rank_lower_bounds": list(self.bond_rank_lower_bounds),
            "maximum_bond_rank": self.maximum_bond_rank,
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


def certify_screen_plan(
    matrix: Any,
    mode_plan: TensorTrainModePlan,
    *,
    primes: Sequence[int] = (251, 257),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
    rejection_operation_fraction: float = 0.25,
    rejection_storage_fraction: float = 0.25,
) -> TensorTrainScreenPlan:
    source = np.asarray(matrix)
    registered = tuple(int(prime) for prime in primes)
    if len(registered) < 2 or len(set(registered)) != len(registered):
        raise TensorTrainRankError("at least two distinct primes are required")
    if bits_per_core <= 0 or activation_bytes <= 0:
        raise TensorTrainRankError("bit and activation widths must be positive")
    if rejection_operation_fraction <= 0 or rejection_storage_fraction <= 0:
        raise TensorTrainRankError("rejection fractions must be positive")
    tensor = interleave_matrix(source, mode_plan.mode_pairs)
    if tensor.ndim < 2:
        raise TensorTrainRankError("a nontrivial mode plan is required")

    rows, columns = (int(value) for value in source.shape)
    physical = mode_plan.physical_dimensions
    bonds = [1] * (tensor.ndim - 1)
    cut_ranks: dict[int, tuple[int, ...]] = {}
    witness_mismatches = 0
    full_rank_cuts = 0
    stopped = False

    cut_shapes = []
    for cut in range(1, tensor.ndim):
        left = math.prod(tensor.shape[:cut])
        right = math.prod(tensor.shape[cut:])
        cut_shapes.append((cut, left, right))
    # Central/high-rank cuts usually prove rejection fastest.  This order is
    # deterministic and fixed independently of tensor values.
    ordered_cuts = sorted(
        cut_shapes,
        key=lambda item: (
            -min(item[1], item[2]),
            abs(math.log(item[1] / item[2])),
            item[0],
        ),
    )

    for cut, _, _ in ordered_cuts:
        unfolding = unfold_interleaved_tensor(tensor, cut)
        certificates = tuple(
            rank_certificate_mod_prime(unfolding, prime=prime)
            for prime in registered
        )
        for certificate in certificates:
            try:
                certificate.validate(unfolding)
            except Exception:
                witness_mismatches += 1
        ranks = tuple(certificate.rank for certificate in certificates)
        lower_bound = max(ranks)
        cut_ranks[cut] = ranks
        bonds[cut - 1] = lower_bound
        full_rank_cuts += int(lower_bound == min(unfolding.shape))

        accounting = _account(
            rows=rows,
            columns=columns,
            physical=physical,
            bonds=bonds,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        )
        operation_fraction = (
            accounting["lower_bound_operations"]
            / accounting["baseline_operations"]
        )
        storage_fraction = (
            accounting["lower_bound_storage_bytes"]
            / accounting["baseline_storage_bytes"]
        )
        if (
            operation_fraction > rejection_operation_fraction
            and storage_fraction > rejection_storage_fraction
        ):
            stopped = len(cut_ranks) < tensor.ndim - 1
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
    certified_cuts = tuple(sorted(cut_ranks))
    return TensorTrainScreenPlan(
        matrix_shape=(rows, columns),
        mode_plan=mode_plan,
        primes=registered,
        certified_cuts=certified_cuts,
        cut_prime_ranks=tuple(cut_ranks[cut] for cut in certified_cuts),
        bond_rank_lower_bounds=tuple(bonds),
        exact_all_cuts=len(certified_cuts) == tensor.ndim - 1,
        stopped_by_threshold=stopped,
        full_unfolding_rank_cuts=full_rank_cuts,
        core_scalar_lower_bound=accounting["core_scalars"],
        baseline_operations=accounting["baseline_operations"],
        lower_bound_operations=accounting["lower_bound_operations"],
        baseline_storage_bytes=accounting["baseline_storage_bytes"],
        lower_bound_storage_bytes=accounting["lower_bound_storage_bytes"],
        baseline_query_bytes=accounting["baseline_query_bytes"],
        lower_bound_query_bytes=accounting["lower_bound_query_bytes"],
        witness_mismatches=witness_mismatches,
        rejection_operation_fraction=rejection_operation_fraction,
        rejection_storage_fraction=rejection_storage_fraction,
    )


def certify_screen_family(
    matrix: Any,
    *,
    primes: Sequence[int] = (251, 257),
    maximum_modes: Sequence[int] = (16,),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
    rejection_operation_fraction: float = 0.25,
    rejection_storage_fraction: float = 0.25,
) -> tuple[TensorTrainScreenPlan, ...]:
    source = np.asarray(matrix)
    if source.ndim != 2:
        raise TensorTrainRankError("matrix must be two-dimensional")
    plans = enumerate_mode_plans(
        int(source.shape[0]),
        int(source.shape[1]),
        maximum_modes=maximum_modes,
    )
    return tuple(
        certify_screen_plan(
            source,
            plan,
            primes=primes,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
            rejection_operation_fraction=rejection_operation_fraction,
            rejection_storage_fraction=rejection_storage_fraction,
        )
        for plan in plans
        if len(plan.mode_pairs) >= 2
    )


def select_favorable_screen_plan(
    plans: Iterable[TensorTrainScreenPlan],
) -> TensorTrainScreenPlan:
    population = tuple(plans)
    if not population:
        raise TensorTrainRankError("mode-plan population is empty")
    if any(plan.witness_mismatches for plan in population):
        raise TensorTrainRankError("cannot select a plan with invalid witnesses")
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


def selected_screen_certificate_rows(
    *,
    model_id: str,
    tensor_name: str,
    matrix: np.ndarray,
    plan: TensorTrainScreenPlan,
    primes: tuple[int, ...],
) -> list[dict[str, Any]]:
    tensor = interleave_matrix(matrix, plan.mode_plan.mode_pairs)
    rows: list[dict[str, Any]] = []
    for cut in plan.certified_cuts:
        unfolding = unfold_interleaved_tensor(tensor, cut)
        for prime in primes:
            certificate = rank_certificate_mod_prime(unfolding, prime=prime)
            certificate.validate(unfolding)
            rows.append(
                {
                    "model_id": model_id,
                    "tensor_name": tensor_name,
                    "variant": plan.mode_plan.variant,
                    "mode_pairs": [list(pair) for pair in plan.mode_plan.mode_pairs],
                    "physical_dimensions": list(plan.mode_plan.physical_dimensions),
                    "cut": cut,
                    "unfolding_shape": list(unfolding.shape),
                    "prime": prime,
                    "rank": certificate.rank,
                    "minimum_dimension": certificate.minimum_dimension,
                    "full_rank": certificate.full_rank,
                    "pivot_rows": list(certificate.pivot_rows),
                    "pivot_columns": list(certificate.pivot_columns),
                    "certified_minor_determinant": (
                        certificate.certified_minor_determinant
                    ),
                    "used_leading_minor_fast_path": (
                        certificate.used_leading_minor_fast_path
                    ),
                    "verified": True,
                    "threshold_censored_plan": plan.stopped_by_threshold,
                }
            )
    return rows
