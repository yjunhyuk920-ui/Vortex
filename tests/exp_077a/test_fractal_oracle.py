from __future__ import annotations

import pytest

from vortex_runtime.fractal_oracle import (
    FractalOracleError,
    aggregate_quality_rows,
    gate_decision,
    homogeneous_length_batches,
    quality_gate,
    registered_teacher_forcing_tokens,
    selected_channel_count,
    selected_parameter_fraction,
    validate_prompt_and_trace_ids,
)


def test_registered_teacher_forcing_replays_proposal_inputs_not_target_outputs() -> None:
    conditioning, expected = registered_teacher_forcing_tokens(
        {
            "first_target_token": 10,
            "proposal_tokens": [20, 21, 22, 23],
            "target_verification_tokens": [30, 31, 32, 33, 34],
        },
        token_count=4,
    )
    assert conditioning == [10, 20, 21]
    assert expected == [10, 30, 31, 32]


def test_homogeneous_length_batches_never_mix_recurrent_sequence_lengths() -> None:
    lengths = [34, 39, 34, 40, 39]
    batches = homogeneous_length_batches(lengths)
    assert batches == [[0, 2], [1, 4], [3]]
    assert all(len({lengths[index] for index in batch}) == 1 for batch in batches)


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
            "proposal_tokens": [9],
            "target_verification_tokens": [2],
        },
        {
            "prompt_id": "e",
            "split": "evaluation",
            "family": "x",
            "first_target_token": 3,
            "proposal_tokens": [8],
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
