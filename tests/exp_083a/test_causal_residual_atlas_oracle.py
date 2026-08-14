from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.causal_residual_atlas_oracle import (
    INFRASTRUCTURE_DECISION,
    INVALID_DECISION,
    PROMOTE_DECISION,
    REJECT_DECISION,
    canonical_prefix_basis,
    native_anchored_pages,
    summarize_gate,
    target_to_candidate_kls,
)


def test_canonical_basis_uses_registered_cutoff_rank_and_sign() -> None:
    source = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )
    result = canonical_prefix_basis(source, maximum_rank=2)
    assert result.numerical_rank == 3
    assert result.retained_rank == 2
    assert result.basis.shape == (4, 2)
    for column in result.basis.T:
        pivot = int(np.argmax(np.abs(column)))
        assert column[pivot] >= 0
    assert np.allclose(result.basis.T @ result.basis, np.eye(2))


def test_basis_rejects_nonfinite_and_empty_inputs() -> None:
    with pytest.raises(ValueError):
        canonical_prefix_basis(np.zeros((0, 3)), maximum_rank=2)
    with pytest.raises(ValueError):
        canonical_prefix_basis(
            np.asarray([[1.0, np.nan]]), maximum_rank=2
        )


def test_native_anchor_enumerates_pages_and_all_page_identity() -> None:
    rng = np.random.default_rng(83)
    weight = rng.normal(size=(7, 10))
    prefix = rng.normal(size=(4, 10))
    basis = canonical_prefix_basis(prefix, maximum_rank=3).basis
    current = rng.normal(size=10)
    native = weight @ current
    result = native_anchored_pages(
        weight,
        basis,
        current,
        native,
        page_columns=4,
    )
    assert result.page_outputs.shape == (3, 7)
    assert np.allclose(result.all_page_output, native, atol=1e-12)
    assert np.all(
        result.exact_unread_error_norms
        <= result.certified_unread_radii + 1e-12
    )


def test_native_anchor_is_exact_when_residual_is_zero() -> None:
    weight = np.arange(24, dtype=np.float64).reshape(4, 6)
    basis = np.eye(6)[:, :2]
    current = np.asarray([2.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    native = weight @ current
    result = native_anchored_pages(
        weight,
        basis,
        current,
        native,
        page_columns=2,
    )
    assert np.allclose(result.page_outputs, native[None, :])
    assert np.allclose(result.exact_unread_error_norms, 0.0)


def test_target_to_candidate_kl_is_zero_for_identity() -> None:
    target = np.asarray([3.0, 2.0, -1.0])
    rows = np.asarray([target, [2.0, 3.0, -1.0]])
    kls = target_to_candidate_kls(target, rows)
    assert kls[0] == pytest.approx(0.0, abs=1e-15)
    assert kls[1] > 0


def _branch(prompt: str, family: str, projection: str, *, ok: bool, kl: float):
    return {
        "prompt_id": prompt,
        "family": family,
        "projection": projection,
        "top1_match": ok,
        "selected_kl": kl,
    }


def _token(prompt: str, family: str, *, ok: bool):
    return {"prompt_id": prompt, "family": family, "success": ok}


def test_first_valid_branch_failure_is_scientific_rejection() -> None:
    summary = summarize_gate(
        [_branch("p0", "a", "q_proj", ok=False, kl=0.2)],
        [_token("p0", "a", ok=False)],
        expected_token_states=2,
        expected_projection_branches=4,
        expected_families=("a", "b"),
        prompts_per_family=1,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=False,
    )
    assert summary["decision"] == REJECT_DECISION
    assert summary["observed_scientific_failure"]


def test_incomplete_success_only_run_is_infrastructure_failure() -> None:
    summary = summarize_gate(
        [_branch("p0", "a", "q_proj", ok=True, kl=0.0)],
        [],
        expected_token_states=2,
        expected_projection_branches=4,
        expected_families=("a", "b"),
        prompts_per_family=1,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=False,
    )
    assert summary["decision"] == INFRASTRUCTURE_DECISION


def test_control_or_leakage_failure_is_invalid_not_scientific() -> None:
    summary = summarize_gate(
        [_branch("p0", "a", "q_proj", ok=False, kl=1.0)],
        [_token("p0", "a", ok=False)],
        expected_token_states=2,
        expected_projection_branches=4,
        expected_families=("a", "b"),
        prompts_per_family=1,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        control_failures=("dense patch changed top1",),
        execution_complete=False,
    )
    assert summary["decision"] == INVALID_DECISION


def test_complete_zero_failure_population_promotes() -> None:
    branches = [
        _branch(prompt, family, projection, ok=True, kl=0.001)
        for prompt, family in (("p0", "a"), ("p1", "b"))
        for projection in ("q_proj", "down_proj")
    ]
    tokens = [_token("p0", "a", ok=True), _token("p1", "b", ok=True)]
    summary = summarize_gate(
        branches,
        tokens,
        expected_token_states=2,
        expected_projection_branches=4,
        expected_families=("a", "b"),
        prompts_per_family=1,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=True,
    )
    assert summary["decision"] == PROMOTE_DECISION
    assert all(summary["gates"].values())


def test_complete_population_rejects_kl_tail() -> None:
    branches = [
        _branch("p0", "a", "q_proj", ok=True, kl=0.0),
        _branch("p0", "a", "down_proj", ok=True, kl=0.0),
        _branch("p1", "b", "q_proj", ok=True, kl=0.0),
        _branch("p1", "b", "down_proj", ok=True, kl=0.2),
    ]
    summary = summarize_gate(
        branches,
        [_token("p0", "a", ok=True), _token("p1", "b", ok=True)],
        expected_token_states=2,
        expected_projection_branches=4,
        expected_families=("a", "b"),
        prompts_per_family=1,
        maximum_mean_kl=0.02,
        maximum_p95_kl=0.05,
        execution_complete=True,
    )
    assert summary["decision"] == REJECT_DECISION
    assert not summary["gates"]["mean_kl_within_limit"]
    assert not summary["gates"]["p95_kl_within_limit"]
