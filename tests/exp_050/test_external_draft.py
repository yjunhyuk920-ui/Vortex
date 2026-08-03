from __future__ import annotations

import math

import pytest

from vortex_runtime.external_draft import (
    ExternalDraftCandidate,
    ExternalDraftError,
    construct_first_token_counterexample,
    dynamic_minimum_exact_prefix,
    evaluate_external_draft,
    favorable_external_draft,
)


def candidate(
    *,
    draft_id: str,
    proposal: tuple[int, ...],
    verified: tuple[int, ...],
    reference: tuple[int, ...],
    target_bytes: int = 1000,
    draft_bytes: int = 100,
) -> ExternalDraftCandidate:
    return ExternalDraftCandidate(
        target_id="target",
        draft_id=draft_id,
        block_size=len(proposal),
        target_parameter_bytes=target_bytes,
        draft_parameter_bytes=draft_bytes,
        proposal_tokens=proposal,
        target_tokens_under_proposal=verified,
        target_reference_tokens=reference,
        draft_forward_count=len(proposal),
    )


def test_external_draft_commits_matching_prefix_and_exact_correction() -> None:
    result = evaluate_external_draft(
        candidate(
            draft_id="draft-a",
            proposal=(1, 2, 9, 9),
            verified=(1, 2, 3, 7),
            reference=(1, 2, 3, 4),
        )
    )
    assert result.matching_prefix == 2
    assert result.verification.committed_tokens == (1, 2, 3)
    assert result.exact_output_match
    assert result.exact_committed_tokens == 3


def test_first_token_counterexample_has_zero_proposal_acceptance() -> None:
    result, adversarial = construct_first_token_counterexample(
        draft_first_token=4,
        vocabulary_size=16,
        block_size=8,
    )
    assert adversarial != 4
    assert result.matching_prefix == 0
    assert result.exact_committed_tokens == 1
    assert result.verification.correction_used
    assert result.exact_output_match


def test_dynamic_requirement_for_4b_draft_is_507() -> None:
    required = dynamic_minimum_exact_prefix(
        target_parameters=405_000_000_000,
        draft_parameters=4_000_000_000,
        allowed_fraction=0.011851851851851851,
    )
    assert required == 507


def test_draft_cost_can_exhaust_budget() -> None:
    with pytest.raises(ExternalDraftError):
        dynamic_minimum_exact_prefix(
            target_parameters=100,
            draft_parameters=2,
            allowed_fraction=0.01,
        )


def test_accounting_charges_every_sequential_draft_forward() -> None:
    result = evaluate_external_draft(
        candidate(
            draft_id="draft-a",
            proposal=(1, 2, 3, 4),
            verified=(1, 2, 3, 4),
            reference=(1, 2, 3, 4),
            target_bytes=1000,
            draft_bytes=100,
        )
    )
    assert result.actual_small_model_target_equivalent_fraction == pytest.approx(
        (4 * 0.1 + 1) / 4
    )
    assert result.normalized_4b_405b_fraction == pytest.approx(
        (4 * (4 / 405) + 1) / 4
    )
    assert math.isfinite(result.normalized_4b_405b_fraction)


def test_favorable_selector_prefers_longer_exact_prefix() -> None:
    short = evaluate_external_draft(
        candidate(
            draft_id="draft-small",
            proposal=(1, 9, 9, 9),
            verified=(1, 2, 3, 4),
            reference=(1, 2, 3, 4),
        )
    )
    long = evaluate_external_draft(
        candidate(
            draft_id="draft-large",
            proposal=(1, 2, 9, 9),
            verified=(1, 2, 3, 4),
            reference=(1, 2, 3, 4),
            draft_bytes=200,
        )
    )
    chosen = favorable_external_draft(
        (short, long),
        draft_parameter_bytes={"draft-small": 100, "draft-large": 200},
    )
    assert chosen.draft_id == "draft-large"
    assert chosen.matching_prefix == 2


def test_favorable_selector_tie_prefers_lower_fraction_then_smaller_draft() -> None:
    first = evaluate_external_draft(
        candidate(
            draft_id="a",
            proposal=(1, 2, 9, 9),
            verified=(1, 2, 3, 4),
            reference=(1, 2, 3, 4),
            draft_bytes=100,
        )
    )
    second = evaluate_external_draft(
        candidate(
            draft_id="b",
            proposal=(1, 2, 9, 9),
            verified=(1, 2, 3, 4),
            reference=(1, 2, 3, 4),
            draft_bytes=200,
        )
    )
    chosen = favorable_external_draft(
        (second, first), draft_parameter_bytes={"a": 100, "b": 200}
    )
    assert chosen.draft_id == "a"


def test_target_future_label_is_preserved() -> None:
    result = evaluate_external_draft(
        ExternalDraftCandidate(
            target_id="target",
            draft_id="draft",
            block_size=2,
            target_parameter_bytes=100,
            draft_parameter_bytes=1,
            proposal_tokens=(1, 2),
            target_tokens_under_proposal=(1, 2),
            target_reference_tokens=(1, 2),
            draft_forward_count=2,
            target_future_information_used=True,
        )
    )
    assert result.target_future_information_used


def test_invalid_self_draft_and_undercharged_forward_count_fail_closed() -> None:
    with pytest.raises(ExternalDraftError):
        evaluate_external_draft(
            ExternalDraftCandidate(
                target_id="same",
                draft_id="same",
                block_size=1,
                target_parameter_bytes=100,
                draft_parameter_bytes=1,
                proposal_tokens=(1,),
                target_tokens_under_proposal=(1,),
                target_reference_tokens=(1,),
                draft_forward_count=1,
            )
        )
    with pytest.raises(ExternalDraftError):
        evaluate_external_draft(
            ExternalDraftCandidate(
                target_id="target",
                draft_id="draft",
                block_size=2,
                target_parameter_bytes=100,
                draft_parameter_bytes=1,
                proposal_tokens=(1, 2),
                target_tokens_under_proposal=(1, 2),
                target_reference_tokens=(1, 2),
                draft_forward_count=1,
            )
        )
