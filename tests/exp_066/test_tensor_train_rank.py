from __future__ import annotations

import math

import numpy as np
import pytest

from vortex_runtime.tensor_train_rank import (
    TensorTrainModePlan,
    TensorTrainRankError,
    certify_mode_family,
    certify_tt_plan,
    deinterleave_tensor,
    enumerate_mode_plans,
    interleave_matrix,
    prime_factors,
    radix_schedules,
    select_favorable_tt_plan,
    unfold_interleaved_tensor,
)


def paired_plan() -> TensorTrainModePlan:
    return TensorTrainModePlan(
        variant="test",
        row_schedule=(2, 2, 2),
        column_schedule=(2, 2, 2),
        mode_pairs=((2, 2), (2, 2), (2, 2)),
    )


def test_prime_factors_and_radix_schedules_are_exact() -> None:
    assert prime_factors(1) == ()
    assert prime_factors(72) == (2, 2, 2, 3, 3)
    schedules = radix_schedules(64, maximum_modes=(2, 4, 8))
    assert schedules
    assert all(math.prod(schedule) == 64 for schedule in schedules)
    assert len(schedules) == len(set(schedules))
    with pytest.raises(TensorTrainRankError):
        prime_factors(0)


def test_mode_plan_population_is_bounded_and_deterministic() -> None:
    first = enumerate_mode_plans(64, 128, maximum_modes=(2, 4, 8))
    second = enumerate_mode_plans(64, 128, maximum_modes=(2, 4, 8))
    assert first == second
    assert first
    assert len(first) < 500
    assert all(
        math.prod(pair[0] for pair in plan.mode_pairs) == 64
        and math.prod(pair[1] for pair in plan.mode_pairs) == 128
        for plan in first
    )


def test_interleave_round_trip_is_exact() -> None:
    matrix = np.arange(64, dtype=np.int16).reshape(8, 8)
    plan = paired_plan()
    tensor = interleave_matrix(matrix, plan.mode_pairs)
    restored = deinterleave_tensor(tensor, plan.mode_pairs)
    assert tensor.shape == (4, 4, 4)
    assert np.array_equal(restored, matrix)
    assert unfold_interleaved_tensor(tensor, 1).shape == (4, 16)
    assert unfold_interleaved_tensor(tensor, 2).shape == (16, 4)


def test_rank_one_mpo_has_unit_bond_ranks() -> None:
    vectors = (
        np.asarray([1, 2, 3, 5], dtype=np.int16),
        np.asarray([2, 1, 4, 3], dtype=np.int16),
        np.asarray([1, 0, 2, 1], dtype=np.int16),
    )
    tensor = np.einsum("i,j,k->ijk", *vectors, dtype=np.int64).astype(np.int16)
    matrix = deinterleave_tensor(tensor, paired_plan().mode_pairs)
    certified = certify_tt_plan(matrix, paired_plan(), primes=(251, 257))
    assert certified.bond_rank_lower_bounds == (1, 1)
    assert certified.cut_prime_ranks == ((1, 1), (1, 1))
    assert certified.witness_mismatches == 0
    assert certified.storage_fraction < 1.0


def test_one_scalar_mutation_raises_a_bond_rank() -> None:
    vectors = (
        np.asarray([1, 2, 3, 5], dtype=np.int16),
        np.asarray([2, 1, 4, 3], dtype=np.int16),
        np.asarray([1, 1, 2, 1], dtype=np.int16),
    )
    tensor = np.einsum("i,j,k->ijk", *vectors, dtype=np.int64).astype(np.int16)
    matrix = deinterleave_tensor(tensor, paired_plan().mode_pairs)
    original = certify_tt_plan(matrix, paired_plan(), primes=(251, 257))
    mutated = matrix.copy()
    mutated[0, 0] += 1
    changed = certify_tt_plan(mutated, paired_plan(), primes=(251, 257))
    assert original.maximum_bond_rank == 1
    assert changed.maximum_bond_rank >= 2
    assert changed.witness_mismatches == 0


def test_dense_random_tensor_has_full_internal_unfoldings() -> None:
    matrix = np.random.default_rng(660066).integers(
        -7, 8, size=(8, 8), dtype=np.int16
    )
    certified = certify_tt_plan(matrix, paired_plan(), primes=(251, 257))
    assert certified.bond_rank_lower_bounds == (4, 4)
    assert certified.full_unfolding_rank_cuts == 2
    assert certified.witness_mismatches == 0


def test_mode_family_and_selection_are_deterministic() -> None:
    matrix = np.random.default_rng(77).integers(
        -7, 8, size=(8, 8), dtype=np.int16
    )
    first = certify_mode_family(
        matrix, primes=(251, 257), maximum_modes=(2, 4, 8)
    )
    second = certify_mode_family(
        matrix.copy(), primes=(251, 257), maximum_modes=(2, 4, 8)
    )
    assert [plan.as_dict() for plan in first] == [
        plan.as_dict() for plan in second
    ]
    selected = select_favorable_tt_plan(first)
    assert selected.witness_mismatches == 0
    assert selected.operation_fraction > 0
    assert selected.storage_fraction > 0
    assert selected.query_byte_fraction > 0


def test_contracts_fail_closed() -> None:
    matrix = np.ones((8, 8), dtype=np.int16)
    plan = paired_plan()
    with pytest.raises(TensorTrainRankError):
        certify_tt_plan(matrix, plan, primes=(251,))
    with pytest.raises(TensorTrainRankError):
        interleave_matrix(matrix.astype(np.float32), plan.mode_pairs)
    with pytest.raises(TensorTrainRankError):
        interleave_matrix(matrix, ((2, 2), (2, 2)))
    with pytest.raises(TensorTrainRankError):
        unfold_interleaved_tensor(np.ones((4,), dtype=np.int16), 1)
    with pytest.raises(TensorTrainRankError):
        select_favorable_tt_plan(())
