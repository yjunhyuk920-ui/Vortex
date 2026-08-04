from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.kronecker_rank import (
    KroneckerRankError,
    certify_all_factorizations,
    certify_kronecker_plan,
    factor_pairs,
    inverse_rearrangement,
    rearrange_kronecker,
    select_favorable_kronecker_plan,
)


def test_factor_pairs_are_ordered_and_nontrivial() -> None:
    assert factor_pairs(8) == ((2, 4), (4, 2))
    assert factor_pairs(7) == ()
    with pytest.raises(KroneckerRankError):
        factor_pairs(0)


def test_rearrangement_round_trip_is_exact() -> None:
    matrix = np.arange(48, dtype=np.int16).reshape(6, 8)
    rearranged = rearrange_kronecker(
        matrix, m1=2, m2=3, n1=2, n2=4
    )
    restored = inverse_rearrangement(
        rearranged, m1=2, m2=3, n1=2, n2=4
    )
    assert rearranged.shape == (4, 12)
    assert np.array_equal(restored, matrix)


def test_exact_single_kronecker_product_has_rank_one() -> None:
    left = np.asarray([[1, 2], [3, 5]], dtype=np.int16)
    right = np.asarray([[2, 0], [1, 4], [3, 1]], dtype=np.int16)
    matrix = np.kron(left, right)
    plan = certify_kronecker_plan(
        matrix, m1=2, m2=3, n1=2, n2=2, primes=(251, 257)
    )
    assert plan.prime_ranks == (1, 1)
    assert plan.rank_lower_bound == 1
    assert plan.witness_mismatches == 0
    assert plan.storage_fraction < 1.0


def test_exact_two_term_kronecker_sum_has_rank_two() -> None:
    left_a = np.asarray([[1, 0], [0, 1]], dtype=np.int16)
    left_b = np.asarray([[0, 1], [1, 0]], dtype=np.int16)
    right_a = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    right_b = np.asarray([[2, 0], [1, 3]], dtype=np.int16)
    matrix = np.kron(left_a, right_a) + np.kron(left_b, right_b)
    plan = certify_kronecker_plan(
        matrix, m1=2, m2=2, n1=2, n2=2, primes=(251, 257)
    )
    assert plan.prime_ranks == (2, 2)
    assert plan.rank_lower_bound == 2
    assert plan.witness_mismatches == 0


def test_one_scalar_mutation_raises_rank_one_control() -> None:
    left = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    right = np.asarray([[2, 1], [0, 3]], dtype=np.int16)
    matrix = np.kron(left, right)
    original = certify_kronecker_plan(
        matrix, m1=2, m2=2, n1=2, n2=2, primes=(251, 257)
    )
    mutated = matrix.copy()
    mutated[0, 0] += 1
    changed = certify_kronecker_plan(
        mutated, m1=2, m2=2, n1=2, n2=2, primes=(251, 257)
    )
    assert original.rank_lower_bound == 1
    assert changed.rank_lower_bound >= 2


def test_dense_random_rearrangement_has_high_certified_rank() -> None:
    matrix = np.random.default_rng(650065).integers(
        -7, 8, size=(8, 8), dtype=np.int16
    )
    plans = certify_all_factorizations(matrix, primes=(251, 257))
    assert len(plans) == 4
    assert all(plan.witness_mismatches == 0 for plan in plans)
    assert max(plan.rank_lower_bound for plan in plans) >= 8


def test_all_factorizations_are_deterministic() -> None:
    matrix = np.random.default_rng(77).integers(
        -7, 8, size=(8, 8), dtype=np.int16
    )
    first = [plan.as_dict() for plan in certify_all_factorizations(matrix)]
    second = [plan.as_dict() for plan in certify_all_factorizations(matrix.copy())]
    assert first == second


def test_selection_uses_favorable_certified_lower_bound() -> None:
    left = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    right = np.arange(16, dtype=np.int16).reshape(4, 4)
    matrix = np.kron(left, right)
    plans = certify_all_factorizations(matrix, primes=(251, 257))
    selected = select_favorable_kronecker_plan(plans)
    assert selected.rank_lower_bound >= 1
    assert selected.witness_mismatches == 0
    assert selected.operation_fraction > 0
    assert selected.query_byte_fraction > 0
    assert selected.storage_fraction > 0


def test_prime_and_shape_contracts_fail_closed() -> None:
    matrix = np.ones((4, 4), dtype=np.int16)
    with pytest.raises(KroneckerRankError):
        certify_kronecker_plan(
            matrix, m1=2, m2=2, n1=2, n2=2, primes=(251,)
        )
    with pytest.raises(KroneckerRankError):
        certify_kronecker_plan(
            matrix, m1=2, m2=3, n1=2, n2=2, primes=(251, 257)
        )
    with pytest.raises(KroneckerRankError):
        rearrange_kronecker(
            matrix.astype(np.float32), m1=2, m2=2, n1=2, n2=2
        )


def test_prime_certificates_agree_on_rank_one_control() -> None:
    matrix = np.kron(
        np.asarray([[1, 3], [2, 5]], dtype=np.int16),
        np.asarray([[2, 1], [4, 3]], dtype=np.int16),
    )
    plan = certify_kronecker_plan(
        matrix,
        m1=2,
        m2=2,
        n1=2,
        n2=2,
        primes=(251, 257, 263),
    )
    assert plan.prime_ranks == (1, 1, 1)
    assert plan.rank_lower_bound == 1
    assert plan.witness_mismatches == 0
