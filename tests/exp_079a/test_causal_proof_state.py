from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pytest

from vortex_runtime.causal_proof_state import (
    CausalProofStateError,
    aggregate_gate_rows,
    block_ranges,
    build_pilot_enclosure,
    certify_proof,
    commit_proof,
    dct_pilot_basis,
    exact_fallback,
    gate_decision,
    install_hot_bound,
    minimum_remaining_l2_radius,
    oracle_block_mask,
    refine_proof,
    start_proof_machine,
    target_equivalent_fraction,
    uniform_budget_plan,
)


def test_registered_target_equivalent_fractions() -> None:
    assert target_equivalent_fraction(
        baseline_billions=4, target_billions=405, latency_multiple=1.2
    ) == pytest.approx(0.011851851851851851)
    assert target_equivalent_fraction(
        baseline_billions=4, target_billions=405, latency_multiple=1.5
    ) == pytest.approx(0.014814814814814815)


def test_dct_pilot_is_orthonormal_and_enclosure_is_sound() -> None:
    rng = np.random.default_rng(790079)
    weight = rng.normal(size=(11, 17))
    basis = dct_pilot_basis(17, 4)
    assert np.allclose(basis.T @ basis, np.eye(4), rtol=0.0, atol=1e-12)
    enclosure = build_pilot_enclosure(
        weight, pilot_basis=basis, row_block=3
    )
    for _ in range(20):
        vector = rng.normal(size=17)
        center = enclosure.image @ (basis.T @ vector)
        error = weight @ vector - center
        ranges = enclosure.row_ranges
        for index, (start, stop) in enumerate(ranges):
            assert np.linalg.norm(error[start:stop]) <= (
                enclosure.residual_block_frobenius[index]
                * np.linalg.norm(vector)
                + 1e-10
            )
        global_radius = minimum_remaining_l2_radius(
            enclosure.residual_block_frobenius,
            input_l2=float(np.linalg.norm(vector)),
            selected_blocks=0,
        )
        assert np.linalg.norm(error) <= global_radius + 1e-10


def test_minimum_radius_removes_largest_disjoint_blocks() -> None:
    norms = [1.0, 4.0, 2.0, 3.0]
    assert minimum_remaining_l2_radius(
        norms, input_l2=2.0, selected_blocks=2
    ) == pytest.approx(2.0 * math.sqrt(1.0**2 + 2.0**2))
    assert minimum_remaining_l2_radius(
        norms, input_l2=2.0, selected_blocks=4
    ) == 0.0


def test_oracle_mask_is_the_exact_best_whole_row_block_choice() -> None:
    residual = np.asarray([[1.0, 1.0, 5.0, 0.0, 2.0, 2.0]])
    mask = oracle_block_mask(residual, selected_blocks=1, row_block=2)
    assert mask.tolist() == [[False, False, True, True, False, False]]

    best_error = np.linalg.norm(residual[~mask])
    exhaustive = []
    ranges = block_ranges(residual.shape[-1], 2)
    for choice in combinations(range(len(ranges)), 1):
        trial = np.zeros_like(residual, dtype=bool)
        for index in choice:
            start, stop = ranges[index]
            trial[:, start:stop] = True
        exhaustive.append(np.linalg.norm(residual[~trial]))
    assert best_error == min(exhaustive)


def test_uniform_budget_counts_sidecar_before_integer_cold_blocks() -> None:
    fraction = target_equivalent_fraction(
        baseline_billions=4, target_billions=405, latency_multiple=1.2
    )
    plan = uniform_budget_plan(
        total_parameter_count=752_000_000,
        matrix_count=24,
        rows=1024,
        columns=3584,
        pilot_rank=4,
        row_block=4,
        allowance_fraction=fraction,
    )
    assert plan.selected_blocks_per_matrix > 0
    assert plan.charged_traffic_fraction <= fraction
    assert plan.unspent_allowance_bytes >= 0
    assert (
        plan.charged_bytes_total
        + plan.matrix_count * plan.cold_bytes_per_block
        > plan.allowance_bytes
    )


def test_proof_machine_never_commits_open_or_over_budget_state() -> None:
    state = start_proof_machine(
        allowance_bytes=1000, total_blocks=4, winner_margin=1.0
    )
    state = install_hot_bound(state, sidecar_bytes=200, uncertainty=2.0)
    with pytest.raises(CausalProofStateError):
        certify_proof(state)
    state = refine_proof(state, cold_bytes=200, next_uncertainty=0.2)
    certified = certify_proof(state)
    assert commit_proof(certified).phase == "committed"

    over = start_proof_machine(
        allowance_bytes=100, total_blocks=2, winner_margin=1.0
    )
    over = install_hot_bound(over, sidecar_bytes=101, uncertainty=0.0)
    assert over.phase == "fallback_required"
    with pytest.raises(CausalProofStateError):
        certify_proof(over)
    assert exact_fallback(over).exact_fallback_used


def _row(*, matches: int, kl: float, radius: float) -> dict:
    return {
        "family": "control",
        "token_count": 1,
        "top1_matches": matches,
        "token_kls": [kl],
        "mlp_relative_l2": 0.01,
        "minimum_sound_radius_ratio": radius,
    }


def test_gate_requires_quality_family_budget_and_local_proof() -> None:
    rows = [_row(matches=1, kl=0.0, radius=0.5) for _ in range(4)]
    aggregate = aggregate_gate_rows(rows)
    gates, decision = gate_decision(
        baseline_mismatches=0,
        budget_fraction=0.01,
        maximum_budget_fraction=0.011851851851851851,
        population=aggregate,
        families={"control": aggregate},
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
        max_radius_ratio_p50=1.0,
        max_radius_ratio_p95=1.0,
    )
    assert all(gates.values())
    assert decision == "PROMOTE_TO_NONLINEAR_CAUSAL_PROOF_PROPAGATION_GATE"

    failed = aggregate_gate_rows([_row(matches=0, kl=1.0, radius=2.0)])
    gates, decision = gate_decision(
        baseline_mismatches=0,
        budget_fraction=0.01,
        maximum_budget_fraction=0.011851851851851851,
        population=failed,
        families={"control": failed},
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
        max_radius_ratio_p50=1.0,
        max_radius_ratio_p95=1.0,
    )
    assert not gates["population_quality_passed"]
    assert not gates["local_proof_radius_passed"]
    assert decision == "REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH"
