"""Reference contracts for target-independent external drafting.

This module contains no model-specific training or calibration. It separates:

- causal external proposal generation;
- exact longest-prefix target verification;
- target/draft stream accounting;
- the universal first-token counterexample;
- explicitly reference-selected favorable pool choice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .block_verify import ProposalVerification, verify_exact_proposal


class ExternalDraftError(ValueError):
    """Raised when proposal, accounting, or selector contracts are invalid."""


@dataclass(frozen=True)
class ExternalDraftCandidate:
    target_id: str
    draft_id: str
    block_size: int
    target_parameter_bytes: int
    draft_parameter_bytes: int
    proposal_tokens: tuple[int, ...]
    target_tokens_under_proposal: tuple[int, ...]
    target_reference_tokens: tuple[int, ...]
    draft_forward_count: int
    target_verification_count: int = 1
    target_future_information_used: bool = False

    def validate(self) -> None:
        if not self.target_id or not self.draft_id:
            raise ExternalDraftError("target_id and draft_id are required")
        if self.target_id == self.draft_id:
            raise ExternalDraftError("target checkpoint cannot draft for itself")
        if self.block_size <= 0:
            raise ExternalDraftError("block_size must be positive")
        if self.target_parameter_bytes <= 0 or self.draft_parameter_bytes <= 0:
            raise ExternalDraftError("parameter bytes must be positive")
        if len(self.proposal_tokens) != self.block_size:
            raise ExternalDraftError("proposal width mismatch")
        if len(self.target_tokens_under_proposal) != self.block_size:
            raise ExternalDraftError("target verification width mismatch")
        if len(self.target_reference_tokens) < self.block_size:
            raise ExternalDraftError("target reference is too short")
        if self.draft_forward_count < self.block_size:
            raise ExternalDraftError(
                "one sequential draft forward must be charged per proposal token"
            )
        if self.target_verification_count != 1:
            raise ExternalDraftError("one exact target block verification is required")
        for sequence in (
            self.proposal_tokens,
            self.target_tokens_under_proposal,
            self.target_reference_tokens[: self.block_size],
        ):
            if any(
                not isinstance(token, int) or token < 0 for token in sequence
            ):
                raise ExternalDraftError("tokens must be non-negative integers")


@dataclass(frozen=True)
class ExternalDraftResult:
    target_id: str
    draft_id: str
    block_size: int
    verification: ProposalVerification
    exact_output_match: bool
    matching_prefix: int
    exact_committed_tokens: int
    actual_small_model_target_equivalent_fraction: float
    normalized_4b_405b_fraction: float
    projected_dynamic_minimum_exact_prefix: int
    draft_parameter_ratio: float
    target_future_information_used: bool


def dynamic_minimum_exact_prefix(
    *,
    target_parameters: int,
    draft_parameters: int,
    allowed_fraction: float,
) -> int:
    if target_parameters <= 0 or draft_parameters < 0:
        raise ExternalDraftError("parameter counts are invalid")
    if not math.isfinite(allowed_fraction) or allowed_fraction <= 0.0:
        raise ExternalDraftError("allowed_fraction must be finite and positive")
    draft_ratio = draft_parameters / target_parameters
    remaining = allowed_fraction - draft_ratio
    if remaining <= 0.0:
        raise ExternalDraftError(
            "draft cost alone consumes the entire target-equivalent budget"
        )
    return math.ceil(1.0 / remaining)


def evaluate_external_draft(
    candidate: ExternalDraftCandidate,
    *,
    projected_target_parameters: int = 405_000_000_000,
    projected_draft_parameters: int = 4_000_000_000,
    projected_allowed_fraction: float = 0.011851851851851851,
) -> ExternalDraftResult:
    candidate.validate()
    verification = verify_exact_proposal(
        candidate.proposal_tokens,
        candidate.target_tokens_under_proposal,
    )
    exact_reference = candidate.target_reference_tokens[
        : len(verification.committed_tokens)
    ]
    exact_output_match = verification.committed_tokens == exact_reference
    if not exact_output_match:
        raise ExternalDraftError(
            "exact verifier committed tokens diverging from target greedy reference"
        )
    accepted = len(verification.committed_tokens)
    if accepted <= 0:
        raise ExternalDraftError("exact verifier must commit at least correction token")
    actual_ratio = candidate.draft_parameter_bytes / candidate.target_parameter_bytes
    actual_fraction = (
        candidate.draft_forward_count * actual_ratio
        + candidate.target_verification_count
    ) / accepted
    projected_ratio = projected_draft_parameters / projected_target_parameters
    normalized_fraction = (
        candidate.block_size * projected_ratio
        + candidate.target_verification_count
    ) / accepted
    dynamic_minimum = dynamic_minimum_exact_prefix(
        target_parameters=projected_target_parameters,
        draft_parameters=projected_draft_parameters,
        allowed_fraction=projected_allowed_fraction,
    )
    for value in (actual_ratio, actual_fraction, normalized_fraction):
        if not math.isfinite(value) or value < 0.0:
            raise ExternalDraftError("non-finite or negative accounting result")
    return ExternalDraftResult(
        target_id=candidate.target_id,
        draft_id=candidate.draft_id,
        block_size=candidate.block_size,
        verification=verification,
        exact_output_match=True,
        matching_prefix=verification.matching_prefix,
        exact_committed_tokens=accepted,
        actual_small_model_target_equivalent_fraction=actual_fraction,
        normalized_4b_405b_fraction=normalized_fraction,
        projected_dynamic_minimum_exact_prefix=dynamic_minimum,
        draft_parameter_ratio=actual_ratio,
        target_future_information_used=candidate.target_future_information_used,
    )


def favorable_external_draft(
    candidates: Iterable[ExternalDraftResult],
    *,
    draft_parameter_bytes: dict[str, int],
) -> ExternalDraftResult:
    rows = tuple(candidates)
    if not rows:
        raise ExternalDraftError("candidate pool must not be empty")
    if any(row.draft_id not in draft_parameter_bytes for row in rows):
        raise ExternalDraftError("draft parameter bytes missing from selector")
    target_ids = {row.target_id for row in rows}
    if len(target_ids) != 1:
        raise ExternalDraftError("favorable selector requires one target")
    return min(
        rows,
        key=lambda row: (
            -row.matching_prefix,
            row.normalized_4b_405b_fraction,
            draft_parameter_bytes[row.draft_id],
            row.draft_id,
            row.block_size,
        ),
    )


@dataclass(frozen=True)
class DeterministicFirstTokenDraft:
    token: int

    def propose(self, prompt: Sequence[int], block_size: int) -> tuple[int, ...]:
        if block_size <= 0:
            raise ExternalDraftError("block_size must be positive")
        if self.token < 0:
            raise ExternalDraftError("draft token must be non-negative")
        del prompt
        return tuple(self.token for _ in range(block_size))


@dataclass(frozen=True)
class AdversarialFirstTokenTarget:
    first_token: int
    continuation_token: int

    def verify(
        self, prompt: Sequence[int], proposal: Sequence[int]
    ) -> tuple[int, ...]:
        if not proposal:
            raise ExternalDraftError("proposal must not be empty")
        if self.first_token < 0 or self.continuation_token < 0:
            raise ExternalDraftError("target tokens must be non-negative")
        del prompt
        return (self.first_token,) + tuple(
            self.continuation_token for _ in range(len(proposal) - 1)
        )

    def greedy_reference(self, length: int) -> tuple[int, ...]:
        if length <= 0:
            raise ExternalDraftError("length must be positive")
        return (self.first_token,) + tuple(
            self.continuation_token for _ in range(length - 1)
        )


def construct_first_token_counterexample(
    *,
    draft_first_token: int,
    vocabulary_size: int,
    block_size: int,
) -> tuple[ExternalDraftResult, int]:
    if vocabulary_size <= 1:
        raise ExternalDraftError("vocabulary_size must exceed one")
    if not 0 <= draft_first_token < vocabulary_size:
        raise ExternalDraftError("draft token outside vocabulary")
    adversarial = (draft_first_token + 1) % vocabulary_size
    draft = DeterministicFirstTokenDraft(draft_first_token)
    target = AdversarialFirstTokenTarget(
        first_token=adversarial,
        continuation_token=adversarial,
    )
    proposal = draft.propose((), block_size)
    verified = target.verify((), proposal)
    reference = target.greedy_reference(block_size)
    result = evaluate_external_draft(
        ExternalDraftCandidate(
            target_id="adversarial-target",
            draft_id="fixed-external-draft",
            block_size=block_size,
            target_parameter_bytes=100,
            draft_parameter_bytes=1,
            proposal_tokens=proposal,
            target_tokens_under_proposal=verified,
            target_reference_tokens=reference,
            draft_forward_count=block_size,
        )
    )
    return result, adversarial
