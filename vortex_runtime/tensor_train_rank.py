"""Exact Tensor-Train / MPO unfolding-rank certification and accounting."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.modular_rank import rank_certificate_mod_prime


class TensorTrainRankError(ValueError):
    """Raised when a TT/MPO certification request is malformed."""


def prime_factors(size: int) -> tuple[int, ...]:
    """Return deterministic prime factors with multiplicity."""
    if size <= 0:
        raise TensorTrainRankError("size must be positive")
    remaining = int(size)
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def _pack_factors(factors: Sequence[int], maximum_mode: int) -> tuple[int, ...]:
    if maximum_mode < 2:
        raise TensorTrainRankError("maximum_mode must be at least two")
    modes: list[int] = []
    current = 1
    for factor in factors:
        factor = int(factor)
        if current > 1 and current * factor > maximum_mode:
            modes.append(current)
            current = factor
        else:
            current *= factor
    if current > 1 or not modes:
        modes.append(current)
    return tuple(modes)


def radix_schedules(
    size: int, *, maximum_modes: Sequence[int] = (2, 4, 8, 16)
) -> tuple[tuple[int, ...], ...]:
    """Return a bounded preregisterable family of exact radix schedules.

    The family intentionally avoids an unbounded rescue search.  It contains
    prime-grained and greedily coarsened schedules in both directions.
    """
    factors = prime_factors(size)
    if not factors:
        return ((1,),)
    candidates: set[tuple[int, ...]] = {factors, tuple(reversed(factors))}
    for maximum in maximum_modes:
        candidates.add(_pack_factors(factors, int(maximum)))
        candidates.add(
            tuple(reversed(_pack_factors(tuple(reversed(factors)), int(maximum))))
        )
    valid = {
        tuple(int(value) for value in schedule)
        for schedule in candidates
        if math.prod(schedule) == size and all(value > 0 for value in schedule)
    }
    return tuple(sorted(valid, key=lambda item: (len(item), item)))


def _paired_modes(
    row_schedule: Sequence[int],
    column_schedule: Sequence[int],
    *,
    reverse_columns: bool = False,
) -> tuple[tuple[int, int], ...]:
    columns = tuple(reversed(column_schedule)) if reverse_columns else tuple(column_schedule)
    length = max(len(row_schedule), len(columns))
    rows = tuple(row_schedule) + (1,) * (length - len(row_schedule))
    columns = columns + (1,) * (length - len(columns))
    return tuple((int(rows[index]), int(columns[index])) for index in range(length))


def _singleton_interleave(
    row_schedule: Sequence[int], column_schedule: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    modes: list[tuple[int, int]] = []
    for row, column in itertools.zip_longest(
        row_schedule, column_schedule, fillvalue=None
    ):
        if row is not None:
            modes.append((int(row), 1))
        if column is not None:
            modes.append((1, int(column)))
    return tuple(modes)


@dataclass(frozen=True)
class TensorTrainModePlan:
    variant: str
    row_schedule: tuple[int, ...]
    column_schedule: tuple[int, ...]
    mode_pairs: tuple[tuple[int, int], ...]

    @property
    def physical_dimensions(self) -> tuple[int, ...]:
        return tuple(row * column for row, column in self.mode_pairs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "row_schedule": list(self.row_schedule),
            "column_schedule": list(self.column_schedule),
            "mode_pairs": [list(pair) for pair in self.mode_pairs],
            "physical_dimensions": list(self.physical_dimensions),
        }


def enumerate_mode_plans(
    rows: int,
    columns: int,
    *,
    maximum_modes: Sequence[int] = (2, 4, 8, 16),
) -> tuple[TensorTrainModePlan, ...]:
    """Enumerate the bounded mode family committed before model observation."""
    row_schedules = radix_schedules(rows, maximum_modes=maximum_modes)
    column_schedules = radix_schedules(columns, maximum_modes=maximum_modes)
    by_modes: dict[tuple[tuple[int, int], ...], TensorTrainModePlan] = {}
    for row_schedule in row_schedules:
        for column_schedule in column_schedules:
            variants = (
                (
                    "paired_forward",
                    _paired_modes(row_schedule, column_schedule),
                ),
                (
                    "paired_reverse_columns",
                    _paired_modes(
                        row_schedule, column_schedule, reverse_columns=True
                    ),
                ),
                (
                    "row_then_column",
                    tuple((int(value), 1) for value in row_schedule)
                    + tuple((1, int(value)) for value in column_schedule),
                ),
                (
                    "column_then_row",
                    tuple((1, int(value)) for value in column_schedule)
                    + tuple((int(value), 1) for value in row_schedule),
                ),
                (
                    "alternating_singletons",
                    _singleton_interleave(row_schedule, column_schedule),
                ),
            )
            for variant, mode_pairs in variants:
                cleaned = tuple(pair for pair in mode_pairs if pair != (1, 1))
                if not cleaned:
                    continue
                if math.prod(pair[0] for pair in cleaned) != rows:
                    raise TensorTrainRankError("row mode product mismatch")
                if math.prod(pair[1] for pair in cleaned) != columns:
                    raise TensorTrainRankError("column mode product mismatch")
                by_modes.setdefault(
                    cleaned,
                    TensorTrainModePlan(
                        variant=variant,
                        row_schedule=tuple(int(value) for value in row_schedule),
                        column_schedule=tuple(
                            int(value) for value in column_schedule
                        ),
                        mode_pairs=cleaned,
                    ),
                )
    return tuple(
        sorted(
            by_modes.values(),
            key=lambda plan: (
                len(plan.mode_pairs),
                plan.physical_dimensions,
                plan.mode_pairs,
                plan.variant,
            ),
        )
    )


def interleave_matrix(matrix: Any, mode_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Reshape a matrix into interleaved MPO physical modes."""
    source = np.asarray(matrix)
    pairs = tuple((int(row), int(column)) for row, column in mode_pairs)
    if source.ndim != 2 or source.size == 0 or source.dtype.kind not in "iu":
        raise TensorTrainRankError("a nonempty integer matrix is required")
    if not pairs or any(row <= 0 or column <= 0 for row, column in pairs):
        raise TensorTrainRankError("positive mode pairs are required")
    rows = tuple(pair[0] for pair in pairs)
    columns = tuple(pair[1] for pair in pairs)
    if source.shape != (math.prod(rows), math.prod(columns)):
        raise TensorTrainRankError("mode product does not match matrix shape")
    length = len(pairs)
    permutation = tuple(
        index
        for pair in ((offset, length + offset) for offset in range(length))
        for index in pair
    )
    tensor = source.reshape(rows + columns).transpose(permutation)
    return np.ascontiguousarray(tensor.reshape(tuple(r * c for r, c in pairs)))


def deinterleave_tensor(
    tensor: Any, mode_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Invert :func:`interleave_matrix` exactly."""
    source = np.asarray(tensor)
    pairs = tuple((int(row), int(column)) for row, column in mode_pairs)
    physical = tuple(row * column for row, column in pairs)
    if source.shape != physical:
        raise TensorTrainRankError("interleaved tensor shape mismatch")
    split_shape = tuple(value for pair in pairs for value in pair)
    length = len(pairs)
    row_axes = tuple(2 * index for index in range(length))
    column_axes = tuple(2 * index + 1 for index in range(length))
    restored = source.reshape(split_shape).transpose(row_axes + column_axes)
    return np.ascontiguousarray(
        restored.reshape(
            math.prod(pair[0] for pair in pairs),
            math.prod(pair[1] for pair in pairs),
        )
    )


def unfold_interleaved_tensor(tensor: Any, cut: int) -> np.ndarray:
    source = np.asarray(tensor)
    if source.ndim < 2:
        raise TensorTrainRankError("at least two physical modes are required")
    if cut <= 0 or cut >= source.ndim:
        raise TensorTrainRankError("cut must be internal")
    return np.ascontiguousarray(
        source.reshape(math.prod(source.shape[:cut]), math.prod(source.shape[cut:]))
    )


@dataclass(frozen=True)
class TensorTrainRankPlan:
    matrix_shape: tuple[int, int]
    mode_plan: TensorTrainModePlan
    primes: tuple[int, ...]
    cut_prime_ranks: tuple[tuple[int, ...], ...]
    bond_rank_lower_bounds: tuple[int, ...]
    full_unfolding_rank_cuts: int
    core_scalar_lower_bound: int
    baseline_operations: int
    lower_bound_operations: int
    baseline_storage_bytes: int
    lower_bound_storage_bytes: int
    baseline_query_bytes: int
    lower_bound_query_bytes: int
    witness_mismatches: int

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_shape": list(self.matrix_shape),
            **self.mode_plan.as_dict(),
            "primes": list(self.primes),
            "cut_prime_ranks": [list(row) for row in self.cut_prime_ranks],
            "bond_rank_lower_bounds": list(self.bond_rank_lower_bounds),
            "maximum_bond_rank": self.maximum_bond_rank,
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
            "witness_mismatches": self.witness_mismatches,
        }


def certify_tt_plan(
    matrix: Any,
    mode_plan: TensorTrainModePlan,
    *,
    primes: Sequence[int] = (251, 257),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
) -> TensorTrainRankPlan:
    """Certify all internal MPO unfolding ranks for one mode plan."""
    source = np.asarray(matrix)
    registered = tuple(int(prime) for prime in primes)
    if len(registered) < 2 or len(set(registered)) != len(registered):
        raise TensorTrainRankError("at least two distinct primes are required")
    if bits_per_core <= 0 or activation_bytes <= 0:
        raise TensorTrainRankError("bit and activation widths must be positive")
    tensor = interleave_matrix(source, mode_plan.mode_pairs)
    if tensor.ndim < 2:
        raise TensorTrainRankError("a nontrivial mode plan is required")

    cut_prime_ranks: list[tuple[int, ...]] = []
    bonds: list[int] = []
    witness_mismatches = 0
    full_rank_cuts = 0
    for cut in range(1, tensor.ndim):
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
        cut_prime_ranks.append(ranks)
        bonds.append(lower_bound)
        full_rank_cuts += int(lower_bound == min(unfolding.shape))

    ranks_with_boundaries = (1,) + tuple(bonds) + (1,)
    physical = mode_plan.physical_dimensions
    core_scalars = sum(
        ranks_with_boundaries[index]
        * physical[index]
        * ranks_with_boundaries[index + 1]
        for index in range(len(physical))
    )
    rows, columns = (int(value) for value in source.shape)
    output_terms = rows + (rows if has_bias else 0)
    baseline_operations = rows * columns + output_terms

    # This deliberately favors the candidate.  It charges one dense-core slot
    # operation per classical MPO core scalar plus final output terms, while
    # omitting factor construction and many contraction-index operations.
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
    favorable_intermediate_scalars = sum(bonds)
    lower_bound_query_bytes = (
        math.ceil(core_scalars * bits_per_core / 8)
        + columns * activation_bytes
        + favorable_intermediate_scalars * activation_bytes * 2
        + rows * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
        + metadata_bytes
    )
    return TensorTrainRankPlan(
        matrix_shape=(rows, columns),
        mode_plan=mode_plan,
        primes=registered,
        cut_prime_ranks=tuple(cut_prime_ranks),
        bond_rank_lower_bounds=tuple(bonds),
        full_unfolding_rank_cuts=full_rank_cuts,
        core_scalar_lower_bound=core_scalars,
        baseline_operations=baseline_operations,
        lower_bound_operations=lower_bound_operations,
        baseline_storage_bytes=baseline_storage_bytes,
        lower_bound_storage_bytes=lower_bound_storage_bytes,
        baseline_query_bytes=baseline_query_bytes,
        lower_bound_query_bytes=lower_bound_query_bytes,
        witness_mismatches=witness_mismatches,
    )


def certify_mode_family(
    matrix: Any,
    *,
    primes: Sequence[int] = (251, 257),
    maximum_modes: Sequence[int] = (2, 4, 8, 16),
    bits_per_core: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
) -> tuple[TensorTrainRankPlan, ...]:
    source = np.asarray(matrix)
    if source.ndim != 2:
        raise TensorTrainRankError("matrix must be two-dimensional")
    plans = enumerate_mode_plans(
        int(source.shape[0]),
        int(source.shape[1]),
        maximum_modes=maximum_modes,
    )
    return tuple(
        certify_tt_plan(
            source,
            plan,
            primes=primes,
            bits_per_core=bits_per_core,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        )
        for plan in plans
        if len(plan.mode_pairs) >= 2
    )


def select_favorable_tt_plan(
    plans: Iterable[TensorTrainRankPlan],
) -> TensorTrainRankPlan:
    population = tuple(plans)
    if not population:
        raise TensorTrainRankError("mode-plan population is empty")
    if any(plan.witness_mismatches for plan in population):
        raise TensorTrainRankError("cannot select a plan with invalid witnesses")
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
