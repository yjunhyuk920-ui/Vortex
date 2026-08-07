from __future__ import annotations

import pytest

from vortex_runtime.fractal_oracle import (
    FractalOracleError,
    aggregate_quality_rows,
    gate_decision,
    quality_gate,
    selected_channel_count,
    selected_parameter_fraction,
    validate_prompt_and_trace_ids,
)


def test_selected_channel_count_respects_hard_fraction_ceiling() -> None:
    assert selected_channel_count(100, 0.01) == 1
    assert selected_channel_count(64, 0.10) == 6
    assert selected_channel_count(8, 1.0) == 8
    with pytest.raises(FractalOracleError):
        selected_channel_count(8, 0.0)


def test_selected_parameter_fraction_counts_gate_up_down() -> None:
    assert selected_parameter_fraction(
        hidden_size=16, intermediate_size=100, selected_channels=10
    ) == 0.1


def test_prompt_trace_join_requires_exact_ids_and_metadata() -> None:
    prompts = {
        "build": [{"id": "b", "family": "x", "prompt": "p"}],
        "evaluation": [{"id": "e", "family": "x", "prompt": "q"}],
    }
    traces = [
        {
            "prompt_id": "b",
            "split": "build",
            "family": "x",
            "first_target_token": 1,
            "target_verification_tokens": [2],
        },
        {
            "prompt_id": "e",
            "split": "evaluation",
            "family": "x",
            "first_target_token": 3,
            "target_verification_tokens": [4],
        },
    ]
    audit = validate_prompt_and_trace_ids(prompts, traces)
    assert audit["prompt_count"] == 2
    with pytest.raises(FractalOracleError):
        validate_prompt_and_trace_ids(prompts, traces[:-1])


def test_quality_aggregation_is_token_weighted_and_nearest_rank() -> None:
    rows = [
        {
            "family": "x",
            "token_count": 2,
            "top1_matches": 2,
            "token_kls": [0.01, 0.02],
            "mlp_relative_l2": 0.1,
        },
        {
            "family": "x",
            "token_count": 1,
            "top1_matches": 0,
            "token_kls": [0.30],
            "mlp_relative_l2": 0.2,
        },
    ]
    aggregate = aggregate_quality_rows(rows)
    assert aggregate["token_count"] == 3
    assert aggregate["top1_agreement"] == pytest.approx(2 / 3)
    assert aggregate["mean_kl"] == pytest.approx(0.11)
    assert aggregate["p95_kl"] == pytest.approx(0.30)


def test_gate_is_fail_closed_on_control_or_family_failure() -> None:
    population = {"top1_agreement": 1.0, "mean_kl": 0.0, "p95_kl": 0.0}
    families = {"x": {"top1_agreement": 1.0}}
    passed = quality_gate(
        population=population,
        families=families,
        baseline_mismatches=0,
        selected_fraction=0.1,
        max_fraction=0.1,
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
    )
    assert gate_decision(passed).startswith("PROMOTE")
    invalid = {**passed, "controls_passed": False}
    assert gate_decision(invalid).startswith("INVALID")
    rejected = {**passed, "family_passed": False}
    assert gate_decision(rejected).startswith("REJECT")


def test_gate_rejects_fraction_above_registered_budget() -> None:
    gate = quality_gate(
        population={"top1_agreement": 1.0, "mean_kl": 0.0, "p95_kl": 0.0},
        families={"x": {"top1_agreement": 1.0}},
        baseline_mismatches=0,
        selected_fraction=0.1001,
        max_fraction=0.1,
        min_top1_agreement=0.99,
        min_family_top1_agreement=0.95,
        max_mean_kl=0.02,
        max_p95_kl=0.05,
    )
    assert gate["budget_passed"] is False
    assert gate_decision(gate).startswith("REJECT")
