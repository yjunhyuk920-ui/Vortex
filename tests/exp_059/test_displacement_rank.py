from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.displacement_rank import (
    DisplacementRankError,
    certify_registered_displacements,
    cyclic_antidiagonal_displacement,
    cyclic_diagonal_displacement,
    displacement_lower_bounds,
    reverse_columns,
    select_favorable_displacement,
    zero_fill_antidiagonal_displacement,
    zero_fill_diagonal_displacement,
)
from vortex_runtime.modular_rank import certify_integer_rank


def toeplitz(rows: int, columns: int) -> np.ndarray:
    values = np.arange(-(rows - 1), columns, dtype=np.int64) * 3 + 1
    return np.asarray(
        [[values[column - row + rows - 1] for column in range(columns)] for row in range(rows)],
        dtype=np.int64,
    )


def hankel(rows: int, columns: int) -> np.ndarray:
    values = np.arange(rows + columns - 1, dtype=np.int64) * 5 - 7
    return np.asarray(
        [[values[row + column] for column in range(columns)] for row in range(rows)],
        dtype=np.int64,
    )


def circulant(size: int) -> np.ndarray:
    first = np.arange(size, dtype=np.int64) * 7 - 3
    return np.asarray(
        [[first[(row - column) % size] for column in range(size)] for row in range(size)],
        dtype=np.int64,
    )


def test_rejects_noninteger_or_empty_matrices() -> None:
    with pytest.raises(DisplacementRankError):
        zero_fill_diagonal_displacement(np.asarray([], dtype=np.int64))
    with pytest.raises(DisplacementRankError):
        cyclic_diagonal_displacement(np.ones((2, 2), dtype=np.float32))


def test_zero_fill_formula_is_exact() -> None:
    matrix = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    expected = np.asarray([[1, 2, 3], [4, 4, 4], [7, 4, 4]], dtype=np.int64)
    assert np.array_equal(zero_fill_diagonal_displacement(matrix), expected)


def test_cyclic_formula_is_exact() -> None:
    matrix = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    expected = matrix - np.asarray([[4, 3], [2, 1]], dtype=np.int64)
    assert np.array_equal(cyclic_diagonal_displacement(matrix), expected)


def test_toeplitz_zero_fill_displacement_rank_at_most_two() -> None:
    matrix = toeplitz(9, 13)
    displacement = zero_fill_diagonal_displacement(matrix)
    certificate = certify_integer_rank(displacement)
    assert certificate.rank_lower_bound <= 2


def test_hankel_antidiagonal_displacement_rank_at_most_two() -> None:
    matrix = hankel(11, 8)
    displacement = zero_fill_antidiagonal_displacement(matrix)
    certificate = certify_integer_rank(displacement)
    assert certificate.rank_lower_bound <= 2


def test_circulant_cyclic_displacement_has_rank_zero() -> None:
    matrix = circulant(10)
    displacement = cyclic_diagonal_displacement(matrix)
    assert np.count_nonzero(displacement) == 0
    certificate = certify_integer_rank(displacement)
    assert certificate.rank_lower_bound == 0


def test_antidiagonal_operators_equal_diagonal_after_reversal() -> None:
    matrix = np.arange(35, dtype=np.int64).reshape(5, 7)
    reversed_matrix = reverse_columns(matrix)
    assert np.array_equal(
        zero_fill_antidiagonal_displacement(matrix),
        zero_fill_diagonal_displacement(reversed_matrix),
    )
    assert np.array_equal(
        cyclic_antidiagonal_displacement(matrix),
        cyclic_diagonal_displacement(reversed_matrix),
    )


def test_dense_random_negative_control_has_high_displacement_rank() -> None:
    matrix = np.random.default_rng(590059).integers(-7, 8, size=(12, 12), dtype=np.int64)
    certificates = certify_registered_displacements(matrix)
    assert min(item.rank_lower_bound for item in certificates) >= 10


def test_all_registered_certificates_validate_and_selection_is_deterministic() -> None:
    matrix = toeplitz(8, 10)
    certificates = certify_registered_displacements(matrix)
    for certificate in certificates:
        certificate.validate(matrix)
    selected = select_favorable_displacement(certificates)
    assert selected.operator == "zero_fill_diagonal"
    assert selected.rank_lower_bound <= 2


def test_favorable_lower_bound_accounting() -> None:
    bounds = displacement_lower_bounds(rows=16, columns=64, rank_lower_bound=2)
    assert bounds["direct_scalar_terms"] == 1024
    assert bounds["query_product_terms_lower_bound"] == 128
    assert bounds["generator_scalar_count_lower_bound"] == 160
    assert bounds["query_fraction_lower_bound"] == pytest.approx(0.125)
    assert bounds["storage_fraction_lower_bound"] == pytest.approx(0.15625)
    assert bounds["rank_exceeds_10_percent_query_budget"] is True
    assert bounds["rank_exceeds_25_percent_query_budget"] is False
