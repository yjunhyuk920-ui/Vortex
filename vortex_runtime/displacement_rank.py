"""Exact shift-displacement operators and favorable structural lower bounds.

EXP-059 measures whether a full-rank integer matrix has low exact displacement
rank under registered zero-fill or cyclic diagonal/anti-diagonal shifts.  The
module never approximates weights and never stores runtime input states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from vortex_runtime.modular_rank import (
    IntegerRankCertificate,
    ModularRankError,
    certify_integer_rank,
)


class DisplacementRankError(ValueError):
    """Raised when a registered displacement request is malformed."""


def _integer_matrix(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0:
        raise DisplacementRankError("matrix must be nonempty and two-dimensional")
    if source.dtype.kind not in "iu":
        raise DisplacementRankError("displacement rank requires integer weights")
    return source.astype(np.int64, copy=False)


def reverse_columns(matrix: np.ndarray) -> np.ndarray:
    """Return an exact column reversal used for anti-diagonal structure."""

    source = _integer_matrix(matrix)
    return np.ascontiguousarray(source[:, ::-1])


def zero_fill_diagonal_displacement(matrix: np.ndarray) -> np.ndarray:
    """Compute ``W - Z_m W Z_n.T`` with zero-fill down/right shifts."""

    source = _integer_matrix(matrix)
    result = source.copy()
    if source.shape[0] > 1 and source.shape[1] > 1:
        result[1:, 1:] -= source[:-1, :-1]
    return result


def zero_fill_antidiagonal_displacement(matrix: np.ndarray) -> np.ndarray:
    """Compute zero-fill diagonal displacement after exact column reversal."""

    return zero_fill_diagonal_displacement(reverse_columns(matrix))


def cyclic_diagonal_displacement(matrix: np.ndarray) -> np.ndarray:
    """Compute ``W - P_m W P_n.T`` for cyclic down/right shifts."""

    source = _integer_matrix(matrix)
    shifted = np.roll(source, shift=(1, 1), axis=(0, 1))
    return source - shifted


def cyclic_antidiagonal_displacement(matrix: np.ndarray) -> np.ndarray:
    """Compute cyclic diagonal displacement after exact column reversal."""

    return cyclic_diagonal_displacement(reverse_columns(matrix))


REGISTERED_OPERATORS: tuple[
    tuple[str, Callable[[np.ndarray], np.ndarray]], ...
] = (
    ("zero_fill_diagonal", zero_fill_diagonal_displacement),
    ("zero_fill_antidiagonal", zero_fill_antidiagonal_displacement),
    ("cyclic_diagonal", cyclic_diagonal_displacement),
    ("cyclic_antidiagonal", cyclic_antidiagonal_displacement),
)


def displacement_lower_bounds(
    *, rows: int, columns: int, rank_lower_bound: int
) -> dict[str, int | float | bool]:
    """Return favorable query/storage lower bounds for displacement generators.

    Query work is charged as only ``r * max(m, n)`` frequency-domain products.
    Storage is charged as only ``r * (m + n)`` generator scalars.  FFT/NTT
    transforms, boundary terms, bitwidth growth, metadata, and operator search
    are deliberately omitted, so these bounds favor the candidate.
    """

    if rows <= 0 or columns <= 0:
        raise DisplacementRankError("matrix dimensions must be positive")
    minimum = min(rows, columns)
    if rank_lower_bound < 0 or rank_lower_bound > minimum:
        raise DisplacementRankError("rank lower bound outside matrix dimensions")
    direct = rows * columns
    query_terms = rank_lower_bound * max(rows, columns)
    generator_scalars = rank_lower_bound * (rows + columns)
    max_rank_query_10 = math.floor(0.10 * direct / max(rows, columns))
    max_rank_query_25 = math.floor(0.25 * direct / max(rows, columns))
    max_rank_storage_10 = math.floor(0.10 * direct / (rows + columns))
    max_rank_storage_25 = math.floor(0.25 * direct / (rows + columns))
    return {
        "rows": rows,
        "columns": columns,
        "minimum_dimension": minimum,
        "rank_lower_bound": rank_lower_bound,
        "direct_scalar_terms": direct,
        "query_product_terms_lower_bound": query_terms,
        "generator_scalar_count_lower_bound": generator_scalars,
        "query_fraction_lower_bound": query_terms / direct,
        "storage_fraction_lower_bound": generator_scalars / direct,
        "maximum_rank_for_10_percent_query": max_rank_query_10,
        "maximum_rank_for_25_percent_query": max_rank_query_25,
        "maximum_rank_for_10_percent_storage": max_rank_storage_10,
        "maximum_rank_for_25_percent_storage": max_rank_storage_25,
        "rank_exceeds_10_percent_query_budget": rank_lower_bound > max_rank_query_10,
        "rank_exceeds_25_percent_query_budget": rank_lower_bound > max_rank_query_25,
        "rank_exceeds_10_percent_storage_budget": rank_lower_bound > max_rank_storage_10,
        "rank_exceeds_25_percent_storage_budget": rank_lower_bound > max_rank_storage_25,
    }


@dataclass(frozen=True)
class DisplacementCertificate:
    operator: str
    displacement: np.ndarray
    rank_certificate: IntegerRankCertificate
    lower_bounds: dict[str, int | float | bool]

    @property
    def rank_lower_bound(self) -> int:
        return self.rank_certificate.rank_lower_bound

    def validate(self, source: np.ndarray) -> None:
        matrix = _integer_matrix(source)
        operators = dict(REGISTERED_OPERATORS)
        if self.operator not in operators:
            raise DisplacementRankError("certificate names an unregistered operator")
        expected = operators[self.operator](matrix)
        if not np.array_equal(expected, self.displacement):
            raise DisplacementRankError("displacement matrix mismatch")
        self.rank_certificate.validate(self.displacement)
        expected_bounds = displacement_lower_bounds(
            rows=matrix.shape[0],
            columns=matrix.shape[1],
            rank_lower_bound=self.rank_certificate.rank_lower_bound,
        )
        if self.lower_bounds != expected_bounds:
            raise DisplacementRankError("displacement accounting mismatch")


def certify_registered_displacements(
    matrix: np.ndarray,
    *,
    primes: Sequence[int] = (251, 257, 263),
) -> tuple[DisplacementCertificate, ...]:
    """Certify all registered exact shift displacements for one matrix."""

    source = _integer_matrix(matrix)
    results: list[DisplacementCertificate] = []
    for name, operator in REGISTERED_OPERATORS:
        displacement = operator(source)
        try:
            certificate = certify_integer_rank(displacement, primes=primes)
        except ModularRankError as error:
            raise DisplacementRankError(str(error)) from error
        lower_bounds = displacement_lower_bounds(
            rows=source.shape[0],
            columns=source.shape[1],
            rank_lower_bound=certificate.rank_lower_bound,
        )
        result = DisplacementCertificate(
            operator=name,
            displacement=displacement,
            rank_certificate=certificate,
            lower_bounds=lower_bounds,
        )
        result.validate(source)
        results.append(result)
    return tuple(results)


def select_favorable_displacement(
    certificates: Sequence[DisplacementCertificate],
) -> DisplacementCertificate:
    """Select the most favorable completed operator after all are certified."""

    items = tuple(certificates)
    expected_names = {name for name, _ in REGISTERED_OPERATORS}
    if len(items) != len(REGISTERED_OPERATORS):
        raise DisplacementRankError("all registered operators must be supplied")
    if {item.operator for item in items} != expected_names:
        raise DisplacementRankError("registered operator population mismatch")
    return min(
        items,
        key=lambda item: (
            float(item.lower_bounds["query_fraction_lower_bound"]),
            float(item.lower_bounds["storage_fraction_lower_bound"]),
            item.rank_lower_bound,
            item.operator,
        ),
    )
