from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.modular_rank import (
    ModularRankError,
    certify_integer_rank,
    factorization_lower_bounds,
    modular_determinant,
    rank_certificate_mod_prime,
)


def test_identity_has_verified_full_rank_certificate() -> None:
    matrix = np.eye(8, dtype=np.int16)
    certificate = certify_integer_rank(matrix)
    assert certificate.rank_lower_bound == 8
    assert certificate.full_integer_rank_proven is True
    assert certificate.certificate_prime == 251
    assert len(certificate.prime_certificates) == 1
    certificate.validate(matrix)


def test_known_exact_low_rank_product_has_registered_rank() -> None:
    left = np.array(
        [[1, 0], [0, 1], [1, 1], [2, -1], [3, 2]], dtype=np.int16
    )
    right = np.array(
        [[1, 2, 3, 4, 5, 6], [2, -1, 0, 1, 3, -2]], dtype=np.int16
    )
    matrix = left @ right
    certificate = certify_integer_rank(matrix)
    assert certificate.rank_lower_bound == 2
    assert certificate.full_integer_rank_proven is False
    assert len(certificate.prime_certificates) == 3
    certificate.validate(matrix)


def test_duplicate_rows_are_rank_deficient() -> None:
    matrix = np.array(
        [[1, 2, 3], [1, 2, 3], [0, 1, 1]], dtype=np.int16
    )
    certificate = certify_integer_rank(matrix)
    assert certificate.rank_lower_bound == 2
    assert certificate.minimum_dimension == 3
    assert certificate.full_integer_rank_proven is False


def test_row_and_column_permutations_preserve_rank() -> None:
    rng = np.random.default_rng(58058)
    matrix = rng.integers(-7, 8, size=(9, 7), dtype=np.int16)
    original = certify_integer_rank(matrix)
    permuted = matrix[
        rng.permutation(matrix.shape[0])
    ][:, rng.permutation(matrix.shape[1])]
    changed = certify_integer_rank(permuted)
    assert original.rank_lower_bound == changed.rank_lower_bound
    assert original.full_integer_rank_proven == changed.full_integer_rank_proven


def test_rectangular_full_column_rank_uses_valid_minor() -> None:
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 3, 5, 7],
        ],
        dtype=np.int16,
    )
    certificate = rank_certificate_mod_prime(matrix, prime=251)
    assert certificate.rank == 4
    assert certificate.full_rank
    certificate.validate(matrix)


def test_rectangular_full_row_rank_is_certified() -> None:
    matrix = np.array(
        [
            [1, 0, 0, 0, 2, 3],
            [0, 1, 0, 0, 4, 5],
            [0, 0, 1, 0, 6, 7],
            [0, 0, 0, 1, 1, -1],
        ],
        dtype=np.int16,
    )
    certificate = certify_integer_rank(matrix)
    assert certificate.rank_lower_bound == 4
    assert certificate.full_integer_rank_proven
    certificate.validate(matrix)


def test_modular_determinant_matches_known_value() -> None:
    matrix = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]], dtype=np.int16)
    # Integer determinant is 22.
    assert modular_determinant(matrix, prime=251) == 22


def test_factorization_lower_bounds_charge_both_products() -> None:
    bounds = factorization_lower_bounds(
        rows=256, columns=1024, rank_lower_bound=256
    )
    assert bounds["direct_scalar_terms"] == 256 * 1024
    assert bounds["factor_scalar_terms_lower_bound"] == 256 * (256 + 1024)
    assert bounds["operation_fraction_lower_bound"] == 1.25
    assert bounds["storage_fraction_lower_bound"] == 1.25
    assert bounds["rank_lower_bound_exceeds_break_even"] is True
    assert bounds["rank_lower_bound_exceeds_10_percent_budget"] is True


def test_zero_matrix_has_rank_zero_certificate() -> None:
    matrix = np.zeros((4, 6), dtype=np.int16)
    certificate = certify_integer_rank(matrix)
    assert certificate.rank_lower_bound == 0
    assert certificate.full_integer_rank_proven is False
    certificate.validate(matrix)
    bounds = factorization_lower_bounds(rows=4, columns=6, rank_lower_bound=0)
    assert bounds["operation_fraction_lower_bound"] == 0.0


def test_invalid_requests_fail_closed() -> None:
    with pytest.raises(ModularRankError):
        certify_integer_rank(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ModularRankError):
        certify_integer_rank(np.ones((2, 2), dtype=np.int16), primes=(251, 251))
    with pytest.raises(ModularRankError):
        certify_integer_rank(np.ones((2, 2), dtype=np.int16), primes=(9,))
    with pytest.raises(ModularRankError):
        modular_determinant(np.ones((2, 3), dtype=np.int16), prime=251)
    with pytest.raises(ModularRankError):
        factorization_lower_bounds(rows=0, columns=2, rank_lower_bound=0)
    with pytest.raises(ModularRankError):
        factorization_lower_bounds(rows=2, columns=2, rank_lower_bound=3)
