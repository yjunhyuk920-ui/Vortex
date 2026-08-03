"""Exact causal block proposal verification for EXP-048.

This module is model-independent reference machinery. A target callback must
return the exact next-token predictions for every proposed position from one
teacher-forced causal target pass. Only the longest matching proposal prefix is
committed. At the first mismatch, the target prediction at that position is an
exact correction because all preceding proposal tokens matched the exact path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence


class BlockVerificationError(ValueError):
    """Raised when a proposal/verifier contract is malformed."""


@dataclass(frozen=True)
class ProposalBatch:
    tokens: tuple[int, ...]
    draft_layer_equivalent_streams: float = 0.0
    draft_output_head_equivalent_streams: float = 0.0
    draft_steps: int = 0
    future_information_used: bool = False
    label: str = "proposal"

    def validate(self, expected_width: int) -> None:
        if expected_width <= 0:
            raise BlockVerificationError("expected_width must be positive")
        if len(self.tokens) != expected_width:
            raise BlockVerificationError(
                f"proposal length {len(self.tokens)} != expected width {expected_width}"
            )
        if any(not isinstance(token, int) or token < 0 for token in self.tokens):
            raise BlockVerificationError("proposal tokens must be non-negative integers")
        for name, value in (
            ("draft_layer_equivalent_streams", self.draft_layer_equivalent_streams),
            ("draft_output_head_equivalent_streams", self.draft_output_head_equivalent_streams),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise BlockVerificationError(f"{name} must be finite and non-negative")
        if self.draft_steps < 0:
            raise BlockVerificationError("draft_steps must be non-negative")

    @property
    def draft_target_equivalent_streams(self) -> float:
        return (
            self.draft_layer_equivalent_streams
            + self.draft_output_head_equivalent_streams
        )


@dataclass(frozen=True)
class ProposalVerification:
    proposal: tuple[int, ...]
    target_tokens: tuple[int, ...]
    matching_prefix: int
    committed_tokens: tuple[int, ...]
    correction_used: bool

    @property
    def accepted_proposal_tokens(self) -> int:
        return self.matching_prefix

    @property
    def rejected_scored_positions(self) -> int:
        return len(self.proposal) - self.matching_prefix


@dataclass(frozen=True)
class VerificationCycle:
    cycle_index: int
    prefix_length_before: int
    requested_width: int
    proposal: tuple[int, ...]
    target_tokens: tuple[int, ...]
    matching_prefix: int
    committed_tokens: tuple[int, ...]
    correction_used: bool
    target_full_streams: float
    correction_target_streams: float
    draft_layer_equivalent_streams: float
    draft_output_head_equivalent_streams: float
    draft_steps: int
    future_information_used: bool
    proposal_label: str

    @property
    def accepted_tokens(self) -> int:
        return len(self.committed_tokens)

    @property
    def target_equivalent_streams(self) -> float:
        return (
            self.target_full_streams
            + self.correction_target_streams
            + self.draft_layer_equivalent_streams
            + self.draft_output_head_equivalent_streams
        )


@dataclass(frozen=True)
class BlockGenerationResult:
    generated_tokens: tuple[int, ...]
    cycles: tuple[VerificationCycle, ...]
    target_full_streams: float
    correction_target_streams: float
    draft_layer_equivalent_streams: float
    draft_output_head_equivalent_streams: float
    draft_steps: int
    proposed_tokens: int
    accepted_proposal_tokens: int
    rejected_scored_positions: int
    future_information_used: bool

    @property
    def generated_count(self) -> int:
        return len(self.generated_tokens)

    @property
    def target_equivalent_streams(self) -> float:
        return (
            self.target_full_streams
            + self.correction_target_streams
            + self.draft_layer_equivalent_streams
            + self.draft_output_head_equivalent_streams
        )

    @property
    def target_equivalent_streams_per_accepted_token(self) -> float:
        if not self.generated_tokens:
            return math.inf
        return self.target_equivalent_streams / len(self.generated_tokens)

    @property
    def accepted_tokens_per_target_verification(self) -> float:
        if self.target_full_streams <= 0.0:
            return math.inf
        return len(self.generated_tokens) / self.target_full_streams

    @property
    def mean_matching_prefix(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(cycle.matching_prefix for cycle in self.cycles) / len(self.cycles)

    @property
    def max_matching_prefix(self) -> int:
        return max((cycle.matching_prefix for cycle in self.cycles), default=0)


TargetBlockPredictor = Callable[[tuple[int, ...], tuple[int, ...]], Sequence[int]]
ProposalFactory = Callable[[tuple[int, ...], int], ProposalBatch]


def verify_exact_proposal(
    proposal: Sequence[int],
    target_tokens: Sequence[int],
    *,
    commit_limit: int | None = None,
) -> ProposalVerification:
    """Verify one proposal block and return an exact safe commit.

    `target_tokens[i]` must be the target model's exact prediction for proposal
    position `i` from a single causal teacher-forced pass over
    `prefix + proposal`. Predictions after the first mismatch are ignored.
    """

    proposed = tuple(int(token) for token in proposal)
    targets = tuple(int(token) for token in target_tokens)
    if not proposed:
        raise BlockVerificationError("proposal must not be empty")
    if len(proposed) != len(targets):
        raise BlockVerificationError("proposal and target token lengths must match")
    if any(token < 0 for token in proposed + targets):
        raise BlockVerificationError("tokens must be non-negative")
    if commit_limit is None:
        commit_limit = len(proposed)
    if commit_limit <= 0:
        raise BlockVerificationError("commit_limit must be positive")

    matching = 0
    for proposed_token, target_token in zip(proposed, targets):
        if proposed_token != target_token:
            break
        matching += 1

    if matching == len(proposed):
        committed = proposed[:commit_limit]
        correction_used = False
    elif matching >= commit_limit:
        committed = proposed[:commit_limit]
        correction_used = False
    else:
        committed = proposed[:matching] + (targets[matching],)
        committed = committed[:commit_limit]
        correction_used = len(committed) > matching

    if not committed:
        raise BlockVerificationError("verification must commit at least one token")
    return ProposalVerification(
        proposal=proposed,
        target_tokens=targets,
        matching_prefix=matching,
        committed_tokens=committed,
        correction_used=correction_used,
    )


def exact_block_generate(
    prefix: Sequence[int],
    *,
    max_new_tokens: int,
    block_size: int,
    proposal_factory: ProposalFactory,
    target_block_predictor: TargetBlockPredictor,
) -> BlockGenerationResult:
    """Generate exact tokens using proposal blocks and one target pass per cycle."""

    if max_new_tokens < 0:
        raise BlockVerificationError("max_new_tokens must be non-negative")
    if block_size <= 0:
        raise BlockVerificationError("block_size must be positive")
    current = tuple(int(token) for token in prefix)
    if not current:
        raise BlockVerificationError("prefix must not be empty")
    if any(token < 0 for token in current):
        raise BlockVerificationError("prefix tokens must be non-negative")

    generated: list[int] = []
    cycles: list[VerificationCycle] = []
    totals = {
        "target_full_streams": 0.0,
        "correction_target_streams": 0.0,
        "draft_layer_equivalent_streams": 0.0,
        "draft_output_head_equivalent_streams": 0.0,
        "draft_steps": 0,
        "proposed_tokens": 0,
        "accepted_proposal_tokens": 0,
        "rejected_scored_positions": 0,
    }
    future_information_used = False

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        width = min(block_size, remaining)
        proposal_batch = proposal_factory(current, width)
        proposal_batch.validate(width)
        target_tokens = tuple(
            int(token)
            for token in target_block_predictor(current, proposal_batch.tokens)
        )
        verification = verify_exact_proposal(
            proposal_batch.tokens,
            target_tokens,
            commit_limit=remaining,
        )

        cycle = VerificationCycle(
            cycle_index=len(cycles),
            prefix_length_before=len(current),
            requested_width=width,
            proposal=proposal_batch.tokens,
            target_tokens=verification.target_tokens,
            matching_prefix=verification.matching_prefix,
            committed_tokens=verification.committed_tokens,
            correction_used=verification.correction_used,
            target_full_streams=1.0,
            correction_target_streams=0.0,
            draft_layer_equivalent_streams=proposal_batch.draft_layer_equivalent_streams,
            draft_output_head_equivalent_streams=(
                proposal_batch.draft_output_head_equivalent_streams
            ),
            draft_steps=proposal_batch.draft_steps,
            future_information_used=proposal_batch.future_information_used,
            proposal_label=proposal_batch.label,
        )
        cycles.append(cycle)
        generated.extend(cycle.committed_tokens)
        current += cycle.committed_tokens

        totals["target_full_streams"] += cycle.target_full_streams
        totals["correction_target_streams"] += cycle.correction_target_streams
        totals["draft_layer_equivalent_streams"] += (
            cycle.draft_layer_equivalent_streams
        )
        totals["draft_output_head_equivalent_streams"] += (
            cycle.draft_output_head_equivalent_streams
        )
        totals["draft_steps"] += cycle.draft_steps
        totals["proposed_tokens"] += len(cycle.proposal)
        totals["accepted_proposal_tokens"] += cycle.matching_prefix
        totals["rejected_scored_positions"] += (
            len(cycle.proposal) - cycle.matching_prefix
        )
        future_information_used = (
            future_information_used or cycle.future_information_used
        )

    return BlockGenerationResult(
        generated_tokens=tuple(generated[:max_new_tokens]),
        cycles=tuple(cycles),
        target_full_streams=totals["target_full_streams"],
        correction_target_streams=totals["correction_target_streams"],
        draft_layer_equivalent_streams=totals[
            "draft_layer_equivalent_streams"
        ],
        draft_output_head_equivalent_streams=totals[
            "draft_output_head_equivalent_streams"
        ],
        draft_steps=int(totals["draft_steps"]),
        proposed_tokens=int(totals["proposed_tokens"]),
        accepted_proposal_tokens=int(totals["accepted_proposal_tokens"]),
        rejected_scored_positions=int(totals["rejected_scored_positions"]),
        future_information_used=future_information_used,
    )


def exact_sequential_generate(
    prefix: Sequence[int],
    *,
    max_new_tokens: int,
    target_block_predictor: TargetBlockPredictor,
) -> BlockGenerationResult:
    """B0 exact greedy baseline with one target stream per token."""

    def no_proposal(current: tuple[int, ...], width: int) -> ProposalBatch:
        del current
        if width != 1:
            raise BlockVerificationError("sequential baseline requires width one")
        return ProposalBatch(tokens=(0,), label="B0_placeholder")

    return exact_block_generate(
        prefix,
        max_new_tokens=max_new_tokens,
        block_size=1,
        proposal_factory=no_proposal,
        target_block_predictor=target_block_predictor,
    )


def oracle_proposal_factory(reference_tokens: Sequence[int]) -> ProposalFactory:
    """Return a non-deployable future-aware B1 proposal factory."""

    reference = tuple(int(token) for token in reference_tokens)
    offset = 0

    def propose(current: tuple[int, ...], width: int) -> ProposalBatch:
        nonlocal offset
        del current
        tokens = reference[offset : offset + width]
        if len(tokens) != width:
            raise BlockVerificationError("oracle reference exhausted")
        offset += width
        return ProposalBatch(
            tokens=tokens,
            future_information_used=True,
            label="B1_perfect_future_oracle",
        )

    return propose


def jacobi_generate_exact(
    prefix: Sequence[int],
    *,
    max_new_tokens: int,
    block_size: int,
    max_iterations: int,
    fill_token: int,
    target_block_predictor: TargetBlockPredictor,
) -> BlockGenerationResult:
    """B2 exact Jacobi control with every target pass charged.

    Stable proposal prefixes are exact by induction. If no stable prefix appears
    within `max_iterations`, the first target prediction from the final pass is
    an exact correction because it depends only on the committed prefix.
    """

    if max_new_tokens < 0:
        raise BlockVerificationError("max_new_tokens must be non-negative")
    if block_size <= 0 or max_iterations <= 0:
        raise BlockVerificationError("block_size and max_iterations must be positive")
    if fill_token < 0:
        raise BlockVerificationError("fill_token must be non-negative")
    current = tuple(int(token) for token in prefix)
    if not current:
        raise BlockVerificationError("prefix must not be empty")

    generated: list[int] = []
    cycles: list[VerificationCycle] = []
    target_passes = 0.0
    proposed_tokens = 0
    accepted_proposal_tokens = 0
    rejected_positions = 0

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        width = min(block_size, remaining)
        guesses = tuple(fill_token for _ in range(width))
        committed: tuple[int, ...] | None = None
        last_targets: tuple[int, ...] | None = None
        stable_prefix = 0
        passes_this_cycle = 0

        for _ in range(max_iterations):
            targets = tuple(
                int(token) for token in target_block_predictor(current, guesses)
            )
            if len(targets) != width:
                raise BlockVerificationError(
                    "target predictor returned the wrong Jacobi width"
                )
            target_passes += 1.0
            passes_this_cycle += 1
            last_targets = targets
            stable_prefix = 0
            for guessed, target in zip(guesses, targets):
                if guessed != target:
                    break
                stable_prefix += 1
            if stable_prefix > 0:
                committed = targets[:stable_prefix]
                break
            guesses = targets

        assert last_targets is not None
        correction_used = False
        if committed is None:
            committed = (last_targets[0],)
            correction_used = True
        committed = committed[:remaining]
        if not committed:
            raise BlockVerificationError("Jacobi cycle committed no tokens")

        cycle = VerificationCycle(
            cycle_index=len(cycles),
            prefix_length_before=len(current),
            requested_width=width,
            proposal=guesses,
            target_tokens=last_targets,
            matching_prefix=stable_prefix,
            committed_tokens=committed,
            correction_used=correction_used,
            target_full_streams=float(passes_this_cycle),
            correction_target_streams=0.0,
            draft_layer_equivalent_streams=0.0,
            draft_output_head_equivalent_streams=0.0,
            draft_steps=0,
            future_information_used=False,
            proposal_label="B2_jacobi",
        )
        cycles.append(cycle)
        generated.extend(committed)
        current += committed
        proposed_tokens += width * passes_this_cycle
        accepted_proposal_tokens += stable_prefix
        rejected_positions += max(0, width - stable_prefix)

    return BlockGenerationResult(
        generated_tokens=tuple(generated[:max_new_tokens]),
        cycles=tuple(cycles),
        target_full_streams=target_passes,
        correction_target_streams=0.0,
        draft_layer_equivalent_streams=0.0,
        draft_output_head_equivalent_streams=0.0,
        draft_steps=0,
        proposed_tokens=proposed_tokens,
        accepted_proposal_tokens=accepted_proposal_tokens,
        rejected_scored_positions=rejected_positions,
        future_information_used=False,
    )
