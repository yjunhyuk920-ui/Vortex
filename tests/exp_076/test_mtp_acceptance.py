from __future__ import annotations

import json
from pathlib import Path

import pytest

from vortex_runtime.mtp_acceptance import (
    MtpAcceptanceError,
    acceptance_distribution,
    derive_k_row,
    exact_commit_tokens,
    gate_decision,
    ideal_minimum_accepted_tokens,
    longest_matching_prefix,
    realized_normalized_parameter_traffic,
    select_build_k,
    validate_prompt_manifest,
)


def test_prompt_manifest_is_disjoint_complete_and_hashed() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "experiments/exp_076/prompts.json").read_text(encoding="utf-8")
    )
    audit = validate_prompt_manifest(
        manifest,
        required_families=[
            "english",
            "korean",
            "code",
            "structured_json",
            "math",
            "adversarial_low_acceptance",
        ],
        build_per_family=1,
        evaluation_per_family=3,
    )
    assert len(audit["prompt_hashes"]) == 24
    assert set(row["id"] for row in manifest["build"]).isdisjoint(
        row["id"] for row in manifest["evaluation"]
    )


def test_prompt_manifest_rejects_duplicate_ids() -> None:
    manifest = {
        "build": [{"id": "same", "family": "f", "prompt": "a"}],
        "evaluation": [{"id": "same", "family": "f", "prompt": "b"}],
    }
    with pytest.raises(MtpAcceptanceError):
        validate_prompt_manifest(
            manifest,
            required_families=["f"],
            build_per_family=1,
            evaluation_per_family=1,
        )


def test_longest_prefix_stops_at_first_mismatch() -> None:
    assert longest_matching_prefix([1, 2, 9, 4], [1, 2, 3, 4]) == 2
    assert longest_matching_prefix([9], [1]) == 0
    assert longest_matching_prefix([1, 2], [1, 2, 3]) == 2


def test_exact_commit_uses_correction_or_bonus() -> None:
    assert exact_commit_tokens(7, [1, 2, 9], [1, 2, 3, 4]) == [7, 1, 2, 3]
    assert exact_commit_tokens(7, [1, 2], [1, 2, 3]) == [7, 1, 2, 3]
    with pytest.raises(MtpAcceptanceError):
        exact_commit_tokens(7, [1, 2], [1, 2])


def test_exp074_minimum_recomputes_from_shape_cost() -> None:
    assert ideal_minimum_accepted_tokens(
        target_block_parameter_reads=10_000_000_000,
        baseline_parameters=1_000_000_000,
        allowed_multiplier=1.2,
        proposal_parameters_per_token=800_000_000,
    ) == 25
    assert ideal_minimum_accepted_tokens(
        target_block_parameter_reads=10_000_000_000,
        baseline_parameters=1_000_000_000,
        allowed_multiplier=1.2,
        proposal_parameters_per_token=1_200_000_000,
    ) is None


def test_rejected_suffix_is_fully_charged() -> None:
    assert realized_normalized_parameter_traffic(
        configured_k=8,
        accepted_prefix=4,
        target_block_parameter_reads=10_000_000_000,
        proposal_parameters_per_token=250_000_000,
        baseline_parameters=1_000_000_000,
    ) == 3.0
    assert (
        realized_normalized_parameter_traffic(
            configured_k=8,
            accepted_prefix=0,
            target_block_parameter_reads=10_000_000_000,
            proposal_parameters_per_token=250_000_000,
            baseline_parameters=1_000_000_000,
        )
        is None
    )


def test_k_row_never_counts_post_mismatch_matches() -> None:
    row = derive_k_row(
        configured_k=4,
        proposal_tokens=[1, 9, 3, 4],
        target_verification_tokens=[1, 2, 3, 4, 5],
        target_block_parameter_reads=10,
        proposal_parameters_per_token=1,
        baseline_parameters=1,
    )
    assert row["accepted_prefix"] == 1
    assert row["per_position_accepted"] == [True, False, False, False]
    assert row["first_correction_token"] == 2


def test_build_k_selection_charges_configured_suffix() -> None:
    rows = []
    for configured_k, accepted in ((2, 2), (4, 4), (8, 4)):
        for _ in range(3):
            rows.append(
                {
                    "configured_k": configured_k,
                    "accepted_prefix": accepted,
                    "normalized_parameter_traffic": (
                        10 + configured_k * 0.25
                    )
                    / accepted,
                }
            )
    assert select_build_k(rows, [2, 4, 8]) == 4


def test_distribution_uses_population_nearest_rank_and_zero_fail_closed() -> None:
    rows = [
        {
            "configured_k": 4,
            "accepted_prefix": accepted,
            "normalized_parameter_traffic": traffic,
        }
        for accepted, traffic in ((0, None), (1, 11.0), (2, 6.0), (4, 3.5))
    ]
    result = acceptance_distribution(rows)
    assert result["accepted_prefix_p05"] == 0
    assert result["accepted_prefix_p50"] == 1
    assert result["accepted_prefix_p95"] == 4
    assert result["zero_accept_rate"] == 0.25
    assert result["normalized_traffic_p50"] is None


def test_decision_ladder_is_fail_closed() -> None:
    assert gate_decision(
        integrity_passed=True,
        acceptance_passed=True,
        family_passed=True,
        traffic_passed=True,
    ).startswith("PROMOTE")
    assert gate_decision(
        integrity_passed=True,
        acceptance_passed=False,
        family_passed=True,
        traffic_passed=True,
    ).startswith("REJECT")
    assert gate_decision(
        integrity_passed=False,
        acceptance_passed=True,
        family_passed=True,
        traffic_passed=True,
    ).startswith("INVALID")
