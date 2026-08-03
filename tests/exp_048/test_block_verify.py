from __future__ import annotations

import math

import pytest

from vortex_runtime.block_verify import (
    BlockVerificationError,
    ProposalBatch,
    exact_block_generate,
    exact_sequential_generate,
    jacobi_generate_exact,
    oracle_proposal_factory,
    verify_exact_proposal,
)


def rule_next(context: tuple[int, ...]) -> int:
    return (context[-1] * 7 + len(context) * 3 + 5) % 31


def toy_target_predictor(
    prefix: tuple[int, ...], proposal: tuple[int, ...]
) -> tuple[int, ...]:
    predictions: list[int] = []
    context = list(prefix)
    for proposed in proposal:
        predictions.append(rule_next(tuple(context)))
        context.append(proposed)
    return tuple(predictions)


def direct_greedy(prefix: tuple[int, ...], count: int) -> tuple[int, ...]:
    current = list(prefix)
    generated: list[int] = []
    for _ in range(count):
        token = rule_next(tuple(current))
        generated.append(token)
        current.append(token)
    return tuple(generated)


def test_verifier_commits_matching_prefix_and_exact_correction() -> None:
    verification = verify_exact_proposal(
        proposal=(4, 8, 99, 3),
        target_tokens=(4, 8, 7, 100),
    )
    assert verification.matching_prefix == 2
    assert verification.committed_tokens == (4, 8, 7)
    assert verification.correction_used
    assert verification.rejected_scored_positions == 2


def test_verifier_does_not_use_predictions_after_first_mismatch() -> None:
    first = verify_exact_proposal((9, 2, 3), (7, 100, 101))
    second = verify_exact_proposal((9, 2, 3), (7, 4, 5))
    assert first.committed_tokens == second.committed_tokens == (7,)
    assert first.matching_prefix == second.matching_prefix == 0


def test_sequential_baseline_matches_direct_greedy() -> None:
    prefix = (2, 5, 7)
    result = exact_sequential_generate(
        prefix,
        max_new_tokens=20,
        target_block_predictor=toy_target_predictor,
    )
    assert result.generated_tokens == direct_greedy(prefix, 20)
    assert result.target_full_streams == 20.0
    assert result.accepted_tokens_per_target_verification == 1.0
    assert not result.future_information_used


def test_perfect_oracle_uses_one_target_pass_per_block() -> None:
    prefix = (1, 3, 9)
    expected = direct_greedy(prefix, 21)
    result = exact_block_generate(
        prefix,
        max_new_tokens=21,
        block_size=8,
        proposal_factory=oracle_proposal_factory(expected),
        target_block_predictor=toy_target_predictor,
    )
    assert result.generated_tokens == expected
    assert result.target_full_streams == 3.0
    assert result.accepted_tokens_per_target_verification == 7.0
    assert result.target_equivalent_streams_per_accepted_token == pytest.approx(3 / 21)
    assert result.future_information_used
    assert result.accepted_proposal_tokens == 21
    assert result.rejected_scored_positions == 0


def test_bad_causal_proposals_still_match_exact_greedy() -> None:
    prefix = (4, 2)

    def always_wrong(current: tuple[int, ...], width: int) -> ProposalBatch:
        del current
        return ProposalBatch(
            tokens=tuple(30 for _ in range(width)),
            draft_layer_equivalent_streams=0.25 * width,
            draft_output_head_equivalent_streams=0.10 * width,
            draft_steps=width,
            label="bad_causal",
        )

    result = exact_block_generate(
        prefix,
        max_new_tokens=12,
        block_size=6,
        proposal_factory=always_wrong,
        target_block_predictor=toy_target_predictor,
    )
    assert result.generated_tokens == direct_greedy(prefix, 12)
    assert len(result.cycles) == 12
    assert all(cycle.matching_prefix == 0 for cycle in result.cycles)
    assert all(cycle.correction_used for cycle in result.cycles)
    assert result.target_full_streams == 12.0
    assert result.draft_steps == 72
    assert result.draft_layer_equivalent_streams == pytest.approx(18.0)
    assert result.draft_output_head_equivalent_streams == pytest.approx(7.2)
    assert result.target_equivalent_streams_per_accepted_token > 3.0
    assert not result.future_information_used


def test_jacobi_control_is_exact_and_charges_every_pass() -> None:
    prefix = (6, 1, 4)
    result = jacobi_generate_exact(
        prefix,
        max_new_tokens=18,
        block_size=6,
        max_iterations=5,
        fill_token=0,
        target_block_predictor=toy_target_predictor,
    )
    assert result.generated_tokens == direct_greedy(prefix, 18)
    assert result.target_full_streams >= len(result.cycles)
    assert result.target_equivalent_streams_per_accepted_token == pytest.approx(
        result.target_full_streams / 18
    )
    assert not result.future_information_used


def test_commit_limit_never_overcommits() -> None:
    verification = verify_exact_proposal(
        proposal=(1, 2, 3, 4),
        target_tokens=(1, 2, 3, 4),
        commit_limit=2,
    )
    assert verification.committed_tokens == (1, 2)
    assert not verification.correction_used


def test_invalid_contracts_fail_closed() -> None:
    with pytest.raises(BlockVerificationError):
        verify_exact_proposal((), ())
    with pytest.raises(BlockVerificationError):
        verify_exact_proposal((1, 2), (1,))
    with pytest.raises(BlockVerificationError):
        ProposalBatch(tokens=(1,), draft_layer_equivalent_streams=math.nan).validate(1)
    with pytest.raises(BlockVerificationError):
        exact_block_generate(
            (),
            max_new_tokens=1,
            block_size=1,
            proposal_factory=lambda prefix, width: ProposalBatch(tokens=(0,)),
            target_block_predictor=toy_target_predictor,
        )


def test_zero_generation_is_well_defined() -> None:
    result = exact_sequential_generate(
        (1,), max_new_tokens=0, target_block_predictor=toy_target_predictor
    )
    assert result.generated_tokens == ()
    assert result.cycles == ()
    assert math.isinf(result.target_equivalent_streams_per_accepted_token)
