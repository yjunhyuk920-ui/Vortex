from __future__ import annotations

from fractions import Fraction
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp_084a.run_experiment import prompt_state

from vortex_runtime.causal_bilinear_rank_gate import (
    CausalBilinearGateError,
    FactorizedModularSpan,
    INVALID_DECISION,
    PRIMES,
    PROMOTE_DECISION,
    REJECT_DECISION,
    exact_hit_witness,
    factor_pair_sha256,
    first_exact_residual_coordinate,
    frozen_side_residual,
    rank_one_fingerprints,
    summarize_gate,
    two_pass_mgs_basis,
    verify_exact_factorized_identity,
    verify_fingerprint_witness,
)


def _outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.outer(left, right).reshape(-1)


def test_factorized_modular_rank_matches_materialized_rank_one_queries() -> None:
    left_rows = (
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
        np.asarray([2.0, -1.0, 0.0], dtype=np.float32),
    )
    right_rows = (
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([2.0, -1.0], dtype=np.float32),
        np.asarray([3.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 0.5], dtype=np.float32),
    )
    for prime in PRIMES:
        span = FactorizedModularSpan(3, 2, prime)
        observed = []
        materialized: list[np.ndarray] = []
        for left, right in zip(left_rows, right_rows, strict=True):
            incremented, coordinate = span.add(left, right)
            observed.append(incremented)
            if incremented:
                assert coordinate is not None
            materialized.append(_outer(left, right))
        expected = int(np.linalg.matrix_rank(np.stack(materialized).astype(np.float64)))
        assert span.rank == expected
        assert sum(observed) == expected
        span.validate()


def test_duplicate_and_zero_factor_receive_no_rank_credit() -> None:
    left = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    right = np.asarray([0.5, 4.0], dtype=np.float32)
    span = FactorizedModularSpan(3, 2, PRIMES[0])
    assert span.add(left, right)[0]
    assert not span.add(left.copy(), right.copy())[0]
    assert not span.add(np.zeros(3, dtype=np.float32), right)[0]
    assert span.rank == 1


def test_exact_outer_product_witness_requires_complete_identity() -> None:
    left_a = np.asarray([1.0, 2.0], dtype=np.float32)
    right_a = np.asarray([3.0, -1.0, 0.5], dtype=np.float32)
    left_b = np.asarray([-2.0, 1.0], dtype=np.float32)
    right_b = np.asarray([1.0, 4.0, 2.0], dtype=np.float32)
    basis = [(left_a, right_a), (left_b, right_b)]
    assert verify_exact_factorized_identity(basis, (left_a, right_a), [1, 0])
    assert not verify_exact_factorized_identity(basis, (left_a, right_a), [0, 1])
    assert first_exact_residual_coordinate(
        basis, (left_a, right_a), [0, 1]
    ) == (0, 0)


def test_exact_hit_coefficients_are_solved_at_frozen_pivots() -> None:
    left_a = np.asarray([1.0, 2.0], dtype=np.float32)
    right_a = np.asarray([1.0, -1.0], dtype=np.float32)
    left_b = np.asarray([0.0, 1.0], dtype=np.float32)
    right_b = np.asarray([2.0, 3.0], dtype=np.float32)
    # A scalar multiple of a rank-one tensor remains a rank-one tensor.
    target_left = np.asarray([2.0, 4.0], dtype=np.float32)
    target_right = right_a.copy()
    hit, coefficients = exact_hit_witness(
        [(left_a, right_a), (left_b, right_b)],
        [(0, 0), (1, 1)],
        (target_left, target_right),
    )
    assert hit
    assert coefficients == (Fraction(2), Fraction(0))


def test_nonmember_never_becomes_a_hit_from_modular_nonincrease_claim() -> None:
    left_a = np.asarray([1.0, 0.0], dtype=np.float32)
    right_a = np.asarray([1.0, 0.0], dtype=np.float32)
    target = (
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
    hit, _ = exact_hit_witness([(left_a, right_a)], [(0, 0)], target)
    assert not hit


def test_two_pass_mgs_is_ordered_float32_and_residual_excludes_basis() -> None:
    rows = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    basis = two_pass_mgs_basis(rows, maximum_rank=2)
    assert basis.dtype == np.float32
    assert basis.shape == (3, 2)
    residual = frozen_side_residual(rows[2], basis)
    assert residual.dtype == np.float32
    assert np.array_equal(residual, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_hash_canonicalizes_signed_zero_and_nonfinite_fails_closed() -> None:
    left = np.asarray([0.0, 1.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0, 1.0], dtype=np.float32)
    right = np.asarray([2.0], dtype=np.float32)
    assert factor_pair_sha256(left, right) == factor_pair_sha256(negative_zero, right)
    with pytest.raises(CausalBilinearGateError):
        factor_pair_sha256(np.asarray([np.nan], dtype=np.float32), right)


def test_six_rank_one_fingerprints_preserve_exact_scalar_witness() -> None:
    left = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    right = np.asarray([3.0, 1.0], dtype=np.float32)
    target = (np.asarray([2.0, -4.0, 1.0], dtype=np.float32), right)
    seeds = (11, 23, 37, 53, 71, 97)
    values = rank_one_fingerprints(left, right, prime=PRIMES[0], seeds=seeds)
    assert len(values) == 6
    assert verify_fingerprint_witness(
        [(left, right)], target, [Fraction(2)], prime=PRIMES[0], seeds=seeds
    )


def _rows(hit_count: int, miss_count: int) -> list[dict[str, object]]:
    families = (
        "english",
        "korean",
        "code",
        "structured_json",
        "math",
        "adversarial_low_acceptance",
    )
    values: list[dict[str, object]] = []
    for index in range(hit_count):
        values.append({"family": families[index % 6], "exact_hit": True})
    for index in range(miss_count):
        values.append({"family": families[index % 6], "exact_hit": False})
    return values


def test_fifth_miss_is_immediate_scientific_rejection() -> None:
    result = summarize_gate(
        _rows(0, 5),
        control_failures=[],
        execution_complete=False,
        oracle_rank_lower_bound=5,
    )
    assert result["decision"] == REJECT_DECISION


def test_rank_28_is_immediate_scientific_rejection() -> None:
    result = summarize_gate(
        _rows(28, 0),
        control_failures=[],
        execution_complete=False,
        oracle_rank_lower_bound=28,
    )
    assert result["decision"] == REJECT_DECISION


def test_complete_population_promotes_only_with_five_hits_per_family() -> None:
    rows = []
    for family in (
        "english",
        "korean",
        "code",
        "structured_json",
        "math",
        "adversarial_low_acceptance",
    ):
        rows.extend({"family": family, "exact_hit": True} for _ in range(6))
    result = summarize_gate(
        rows,
        control_failures=[],
        execution_complete=True,
        oracle_rank_lower_bound=23,
    )
    assert result["decision"] == PROMOTE_DECISION
    rows[0]["exact_hit"] = False
    rows[1]["exact_hit"] = False
    result = summarize_gate(
        rows,
        control_failures=[],
        execution_complete=True,
        oracle_rank_lower_bound=23,
    )
    assert result["decision"] == REJECT_DECISION


def test_control_failure_has_priority_over_scientific_result() -> None:
    result = summarize_gate(
        _rows(0, 5),
        control_failures=["prefix_replay"],
        execution_complete=False,
        oracle_rank_lower_bound=30,
    )
    assert result["decision"] == INVALID_DECISION


def test_prompt_basis_uses_only_sequential_single_token_prefix_steps() -> None:
    source = inspect.getsource(prompt_state)
    assert "prefix_ids[:, position : position + 1]" in source
    assert "sequential_single_token_committed_prefix" in source
    assert "replay_indices" not in inspect.signature(prompt_state).parameters
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "experiments/exp_084a/config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["gate"]["prompt_capture"] == (
        "sequential_single_token_committed_prefix_no_batched_prompt_rows"
    )
