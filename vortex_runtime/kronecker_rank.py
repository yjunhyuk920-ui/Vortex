"""Exact Kronecker rearrangement and modular-rank lower-bound accounting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from vortex_runtime.modular_rank import rank_certificate_mod_prime


class KroneckerRankError(ValueError):
    """Raised when a Kronecker factorization request is malformed."""


def factor_pairs(size: int) -> tuple[tuple[int, int], ...]:
    """Return all ordered nontrivial factor pairs."""
    if size <= 0:
        raise KroneckerRankError("size must be positive")
    return tuple(
        (left, size // left)
        for left in range(2, size)
        if size % left == 0 and size // left > 1
    )


def rearrange_kronecker(
    matrix: Any, *, m1: int, m2: int, n1: int, n2: int
) -> np.ndarray:
    """Map block vectors to rows so a single Kronecker product has rank one."""
    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0 or source.dtype.kind not in "iu":
        raise KroneckerRankError("a nonempty integer matrix is required")
    if source.shape != (m1 * m2, n1 * n2):
        raise KroneckerRankError("factorization shape mismatch")
    return np.ascontiguousarray(
        source.reshape(m1, m2, n1, n2)
        .transpose(0, 2, 1, 3)
        .reshape(m1 * n1, m2 * n2)
    )


def inverse_rearrangement(
    rearranged: Any, *, m1: int, m2: int, n1: int, n2: int
) -> np.ndarray:
    source = np.asarray(rearranged)
    if source.shape != (m1 * n1, m2 * n2):
        raise KroneckerRankError("rearranged shape mismatch")
    return np.ascontiguousarray(
        source.reshape(m1, n1, m2, n2)
        .transpose(0, 2, 1, 3)
        .reshape(m1 * m2, n1 * n2)
    )


@dataclass(frozen=True)
class KroneckerRankPlan:
    matrix_shape: tuple[int, int]
    factors: tuple[int, int, int, int]
    rearranged_shape: tuple[int, int]
    primes: tuple[int, ...]
    prime_ranks: tuple[int, ...]
    rank_lower_bound: int
    full_rearrangement_rank_proven: bool
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_shape": list(self.matrix_shape),
            "factors": list(self.factors),
            "rearranged_shape": list(self.rearranged_shape),
            "primes": list(self.primes),
            "prime_ranks": list(self.prime_ranks),
            "rank_lower_bound": self.rank_lower_bound,
            "full_rearrangement_rank_proven": (
                self.full_rearrangement_rank_proven
            ),
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


def certify_kronecker_plan(
    matrix: Any,
    *,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    primes: Sequence[int] = (251, 257),
    bits_per_factor: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
) -> KroneckerRankPlan:
    """Certify a necessary Kronecker-term lower bound over two prime fields."""
    source = np.asarray(matrix)
    registered = tuple(int(prime) for prime in primes)
    if len(registered) < 2 or len(set(registered)) != len(registered):
        raise KroneckerRankError("at least two distinct primes are required")
    if bits_per_factor <= 0 or activation_bytes <= 0:
        raise KroneckerRankError("bit and activation widths must be positive")
    rearranged = rearrange_kronecker(
        source, m1=m1, m2=m2, n1=n1, n2=n2
    )
    certificates = tuple(
        rank_certificate_mod_prime(rearranged, prime=prime)
        for prime in registered
    )
    witness_mismatches = 0
    for certificate in certificates:
        try:
            certificate.validate(rearranged)
        except Exception:
            witness_mismatches += 1
    ranks = tuple(certificate.rank for certificate in certificates)
    rank_lower_bound = max(ranks)
    rows, columns = (int(value) for value in source.shape)
    output_terms = rows + (rows if has_bias else 0)
    baseline_operations = rows * columns + output_terms

    # For W = sum_i A_i tensor B_i and X shaped n2 x n1, one term applies
    # B_i X A_i^T. This is a favorable lower bound: factor decomposition,
    # coefficient widening, and cross-term reduction overhead are not added.
    per_term_operations = m2 * n2 * n1 + m2 * n1 * m1
    lower_bound_operations = (
        rank_lower_bound * per_term_operations
        + max(0, rank_lower_bound - 1) * rows
        + output_terms
    )

    baseline_storage_bytes = (
        math.ceil(rows * columns * bits_per_factor / 8)
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    factor_scalars = rank_lower_bound * (m1 * n1 + m2 * n2)
    lower_bound_storage_bytes = (
        math.ceil(factor_scalars * bits_per_factor / 8)
        + rows * 4
        + (rows * 4 if has_bias else 0)
        + 16
    )

    baseline_query_bytes = (
        math.ceil(rows * columns * bits_per_factor / 8)
        + rows * columns * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    intermediate_scalars = m2 * n1
    lower_bound_query_bytes = (
        math.ceil(factor_scalars * bits_per_factor / 8)
        + rank_lower_bound * columns * activation_bytes
        + rank_lower_bound * intermediate_scalars * activation_bytes * 2
        + rows * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
        + 16
    )
    return KroneckerRankPlan(
        matrix_shape=(rows, columns),
        factors=(m1, m2, n1, n2),
        rearranged_shape=tuple(int(value) for value in rearranged.shape),
        primes=registered,
        prime_ranks=ranks,
        rank_lower_bound=rank_lower_bound,
        full_rearrangement_rank_proven=(
            rank_lower_bound == min(rearranged.shape)
        ),
        baseline_operations=baseline_operations,
        lower_bound_operations=lower_bound_operations,
        baseline_storage_bytes=baseline_storage_bytes,
        lower_bound_storage_bytes=lower_bound_storage_bytes,
        baseline_query_bytes=baseline_query_bytes,
        lower_bound_query_bytes=lower_bound_query_bytes,
        witness_mismatches=witness_mismatches,
    )


def certify_all_factorizations(
    matrix: Any,
    *,
    primes: Sequence[int] = (251, 257),
    bits_per_factor: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
) -> tuple[KroneckerRankPlan, ...]:
    source = np.asarray(matrix)
    if source.ndim != 2:
        raise KroneckerRankError("matrix must be two-dimensional")
    row_pairs = factor_pairs(int(source.shape[0]))
    column_pairs = factor_pairs(int(source.shape[1]))
    if not row_pairs or not column_pairs:
        return ()
    return tuple(
        certify_kronecker_plan(
            source,
            m1=m1,
            m2=m2,
            n1=n1,
            n2=n2,
            primes=primes,
            bits_per_factor=bits_per_factor,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        )
        for m1, m2 in row_pairs
        for n1, n2 in column_pairs
    )


def select_favorable_kronecker_plan(
    plans: Sequence[KroneckerRankPlan],
) -> KroneckerRankPlan:
    if not plans:
        raise KroneckerRankError("factorization population is empty")
    if any(plan.witness_mismatches for plan in plans):
        raise KroneckerRankError("cannot select a plan with invalid witnesses")
    return min(
        plans,
        key=lambda plan: (
            plan.operation_fraction,
            plan.query_byte_fraction,
            plan.storage_fraction,
            plan.factors,
        ),
    )
