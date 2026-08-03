"""Exact modular-rank certificates and factorization lower bounds.

A rank observed modulo a prime is a rigorous lower bound on the rank over the
integers/rationals.  Full rank modulo any prime therefore proves that no lower
exact conventional two-factor representation exists for that integer matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class ModularRankError(ValueError):
    """Raised when an exact modular rank request is malformed."""


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


def _validate_integer_matrix(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0:
        raise ModularRankError("matrix must be nonempty and two-dimensional")
    if source.dtype.kind not in "iu":
        raise ModularRankError("modular rank requires an integer matrix")
    return source


def modular_determinant(matrix: np.ndarray, *, prime: int) -> int:
    """Return an exact determinant modulo `prime` for a square integer matrix."""

    source = _validate_integer_matrix(matrix)
    if source.shape[0] != source.shape[1]:
        raise ModularRankError("determinant requires a square matrix")
    if not _is_prime(int(prime)):
        raise ModularRankError("modulus must be prime")
    work = np.mod(source.astype(np.int64, copy=False), prime).copy()
    size = work.shape[0]
    determinant = 1
    sign = 1
    for pivot_index in range(size):
        candidates = np.flatnonzero(work[pivot_index:, pivot_index])
        if candidates.size == 0:
            return 0
        pivot_row = pivot_index + int(candidates[0])
        if pivot_row != pivot_index:
            work[[pivot_index, pivot_row]] = work[[pivot_row, pivot_index]]
            sign = -sign
        pivot = int(work[pivot_index, pivot_index])
        determinant = (determinant * pivot) % prime
        inverse = pow(pivot, prime - 2, prime)
        if pivot_index + 1 < size:
            normalized_tail = (
                work[pivot_index, pivot_index + 1 :] * inverse
            ) % prime
            factors = work[pivot_index + 1 :, pivot_index].copy()
            if normalized_tail.size and factors.size:
                work[pivot_index + 1 :, pivot_index + 1 :] = (
                    work[pivot_index + 1 :, pivot_index + 1 :]
                    - factors[:, None] * normalized_tail[None, :]
                ) % prime
            work[pivot_index + 1 :, pivot_index] = 0
    return int((determinant * sign) % prime)


@dataclass(frozen=True)
class PrimeRankCertificate:
    prime: int
    rank: int
    minimum_dimension: int
    full_rank: bool
    pivot_rows: tuple[int, ...]
    pivot_columns: tuple[int, ...]
    certified_minor_determinant: int
    used_leading_minor_fast_path: bool

    def validate(self, matrix: np.ndarray) -> None:
        source = _validate_integer_matrix(matrix)
        if not _is_prime(self.prime):
            raise ModularRankError("certificate modulus is not prime")
        if self.minimum_dimension != min(source.shape):
            raise ModularRankError("certificate minimum dimension mismatch")
        if self.rank < 0 or self.rank > self.minimum_dimension:
            raise ModularRankError("certificate rank outside bounds")
        if self.full_rank != (self.rank == self.minimum_dimension):
            raise ModularRankError("certificate full-rank flag mismatch")
        if len(self.pivot_rows) != self.rank or len(self.pivot_columns) != self.rank:
            raise ModularRankError("certificate pivot count mismatch")
        if len(set(self.pivot_rows)) != self.rank or len(set(self.pivot_columns)) != self.rank:
            raise ModularRankError("certificate pivots are not unique")
        if any(index < 0 or index >= source.shape[0] for index in self.pivot_rows):
            raise ModularRankError("certificate pivot row outside matrix")
        if any(index < 0 or index >= source.shape[1] for index in self.pivot_columns):
            raise ModularRankError("certificate pivot column outside matrix")
        if self.rank:
            minor = source[np.ix_(self.pivot_rows, self.pivot_columns)]
            determinant = modular_determinant(minor, prime=self.prime)
            if determinant == 0:
                raise ModularRankError("certified minor is singular")
            if determinant != self.certified_minor_determinant:
                raise ModularRankError("certified minor determinant mismatch")
        elif self.certified_minor_determinant != 1:
            raise ModularRankError("rank-zero certificate determinant must be one")


def _full_elimination_certificate(
    matrix: np.ndarray, *, prime: int
) -> PrimeRankCertificate:
    source = _validate_integer_matrix(matrix)
    work = np.mod(source.astype(np.int64, copy=False), prime).copy()
    row_order = list(range(source.shape[0]))
    rank = 0
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    for column in range(source.shape[1]):
        if rank >= source.shape[0]:
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
        if rank == min(source.shape):
            break
    if rank:
        minor = source[np.ix_(pivot_rows, pivot_columns)]
        determinant = modular_determinant(minor, prime=prime)
        if determinant == 0:
            raise ModularRankError("internal pivot minor verification failed")
    else:
        determinant = 1
    certificate = PrimeRankCertificate(
        prime=prime,
        rank=rank,
        minimum_dimension=min(source.shape),
        full_rank=rank == min(source.shape),
        pivot_rows=tuple(pivot_rows),
        pivot_columns=tuple(pivot_columns),
        certified_minor_determinant=determinant,
        used_leading_minor_fast_path=False,
    )
    certificate.validate(source)
    return certificate


def rank_certificate_mod_prime(
    matrix: np.ndarray, *, prime: int
) -> PrimeRankCertificate:
    """Return an exact rank certificate over one prime field."""

    source = _validate_integer_matrix(matrix)
    if not _is_prime(int(prime)):
        raise ModularRankError("modulus must be prime")
    minimum = min(source.shape)
    leading = source[:minimum, :minimum]
    determinant = modular_determinant(leading, prime=prime)
    if determinant != 0:
        certificate = PrimeRankCertificate(
            prime=int(prime),
            rank=minimum,
            minimum_dimension=minimum,
            full_rank=True,
            pivot_rows=tuple(range(minimum)),
            pivot_columns=tuple(range(minimum)),
            certified_minor_determinant=determinant,
            used_leading_minor_fast_path=True,
        )
        certificate.validate(source)
        return certificate
    return _full_elimination_certificate(source, prime=int(prime))


@dataclass(frozen=True)
class IntegerRankCertificate:
    shape: tuple[int, int]
    primes_registered: tuple[int, ...]
    prime_certificates: tuple[PrimeRankCertificate, ...]
    rank_lower_bound: int
    minimum_dimension: int
    full_integer_rank_proven: bool
    certificate_prime: int | None

    def validate(self, matrix: np.ndarray) -> None:
        source = _validate_integer_matrix(matrix)
        if self.shape != tuple(source.shape):
            raise ModularRankError("integer certificate shape mismatch")
        if self.minimum_dimension != min(source.shape):
            raise ModularRankError("integer certificate dimension mismatch")
        if not self.prime_certificates:
            raise ModularRankError("integer certificate has no prime evidence")
        for certificate in self.prime_certificates:
            certificate.validate(source)
        expected = max(item.rank for item in self.prime_certificates)
        if self.rank_lower_bound != expected:
            raise ModularRankError("integer rank lower bound mismatch")
        if self.full_integer_rank_proven != (expected == self.minimum_dimension):
            raise ModularRankError("integer full-rank conclusion mismatch")
        if self.full_integer_rank_proven:
            valid_primes = {
                item.prime for item in self.prime_certificates if item.full_rank
            }
            if self.certificate_prime not in valid_primes:
                raise ModularRankError("missing full-rank certificate prime")
        elif self.certificate_prime is not None:
            raise ModularRankError("deficient certificate must not name a full-rank prime")


def certify_integer_rank(
    matrix: np.ndarray,
    *,
    primes: Sequence[int] = (251, 257, 263),
) -> IntegerRankCertificate:
    """Certify an integer/rational rank lower bound using exact prime fields.

    Evaluation stops after the first full-rank certificate because one such
    prime is sufficient to prove full rank over the integers/rationals.
    """

    source = _validate_integer_matrix(matrix)
    registered = tuple(int(value) for value in primes)
    if not registered or len(set(registered)) != len(registered):
        raise ModularRankError("registered primes must be nonempty and unique")
    if any(not _is_prime(value) for value in registered):
        raise ModularRankError("all registered moduli must be prime")
    certificates: list[PrimeRankCertificate] = []
    certificate_prime = None
    for prime in registered:
        certificate = rank_certificate_mod_prime(source, prime=prime)
        certificates.append(certificate)
        if certificate.full_rank:
            certificate_prime = prime
            break
    lower_bound = max(item.rank for item in certificates)
    result = IntegerRankCertificate(
        shape=tuple(source.shape),
        primes_registered=registered,
        prime_certificates=tuple(certificates),
        rank_lower_bound=lower_bound,
        minimum_dimension=min(source.shape),
        full_integer_rank_proven=lower_bound == min(source.shape),
        certificate_prime=certificate_prime,
    )
    result.validate(source)
    return result


def factorization_lower_bounds(
    *, rows: int, columns: int, rank_lower_bound: int
) -> dict[str, int | float | bool]:
    """Account conventional exact `W=A@B` operation/storage lower bounds."""

    if rows <= 0 or columns <= 0:
        raise ModularRankError("matrix dimensions must be positive")
    minimum = min(rows, columns)
    if rank_lower_bound < 0 or rank_lower_bound > minimum:
        raise ModularRankError("rank lower bound outside matrix dimensions")
    direct_terms = rows * columns
    factor_terms = rank_lower_bound * (rows + columns)
    operation_fraction = factor_terms / direct_terms
    storage_fraction = factor_terms / direct_terms
    maximum_rank_for_10_percent = math.floor(
        0.10 * direct_terms / (rows + columns)
    )
    maximum_rank_for_25_percent = math.floor(
        0.25 * direct_terms / (rows + columns)
    )
    maximum_rank_for_break_even = math.floor(
        direct_terms / (rows + columns)
    )
    return {
        "rows": rows,
        "columns": columns,
        "minimum_dimension": minimum,
        "rank_lower_bound": rank_lower_bound,
        "direct_scalar_terms": direct_terms,
        "factor_scalar_terms_lower_bound": factor_terms,
        "operation_fraction_lower_bound": operation_fraction,
        "storage_fraction_lower_bound": storage_fraction,
        "maximum_rank_for_10_percent": maximum_rank_for_10_percent,
        "maximum_rank_for_25_percent": maximum_rank_for_25_percent,
        "maximum_rank_for_break_even": maximum_rank_for_break_even,
        "rank_lower_bound_exceeds_10_percent_budget": (
            rank_lower_bound > maximum_rank_for_10_percent
        ),
        "rank_lower_bound_exceeds_25_percent_budget": (
            rank_lower_bound > maximum_rank_for_25_percent
        ),
        "rank_lower_bound_exceeds_break_even": (
            rank_lower_bound > maximum_rank_for_break_even
        ),
    }
