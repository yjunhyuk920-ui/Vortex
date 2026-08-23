from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Iterable, Sequence


class SpeculativeContractError(ValueError):
    """Raised when the causal/exact accounting contract is violated."""


@dataclass(frozen=True)
class VerificationOutcome:
    proposed_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    accepted_draft_tokens: int
    committed_tokens: int
    target_positions_evaluated: int
    target_incremental_repair_calls: int
    mismatch_index: int | None
    exact_committed_tokens: tuple[int, ...]

    @property
    def candidate_ratio(self) -> float:
        if self.committed_tokens <= 0:
            raise SpeculativeContractError("committed_tokens must be positive")
        return self.target_positions_evaluated / self.committed_tokens


@dataclass(frozen=True)
class TrafficRequirement:
    target_fraction: float
    compression_ratio: float
    minimum_committed_tokens: int


@dataclass(frozen=True)
class CaseAccounting:
    case_id: str
    block_length: int
    accepted_draft_tokens: int
    committed_tokens: int
    target_positions_evaluated: int
    draft_prefill_ns: int
    draft_decode_ns: int
    bridge_ns: int
    target_verify_ns: int
    target_repair_ns: int
    draft_rollback_ns: int
    baseline_decode_ns: int
    exact_token_match: bool
    exact_terminal_state_match: bool

    @property
    def candidate_ratio(self) -> float:
        return self.target_positions_evaluated / self.committed_tokens

    @property
    def speculative_ns(self) -> int:
        return (
            self.draft_prefill_ns
            + self.draft_decode_ns
            + self.bridge_ns
            + self.target_verify_ns
            + self.target_repair_ns
            + self.draft_rollback_ns
        )

    @property
    def latency_ratio(self) -> float:
        if self.baseline_decode_ns <= 0:
            raise SpeculativeContractError("baseline_decode_ns must be positive")
        return self.speculative_ns / self.baseline_decode_ns


def minimum_committed_tokens(
    *, target_fraction: float, compression_ratio: float = 1.0
) -> TrafficRequirement:
    """Return the exact integer block length needed for one target sweep.

    No unmeasured compression credit is permitted: callers use 1.0 unless the
    exact checkpoint representation and decoder throughput were measured in the
    same run.  The one-sweep logical traffic fraction is

        1 / (compression_ratio * committed_tokens).
    """

    if not 0.0 < target_fraction <= 1.0:
        raise SpeculativeContractError("target_fraction must be in (0, 1]")
    if compression_ratio < 1.0:
        raise SpeculativeContractError("compression_ratio must be >= 1")
    required = ceil(1.0 / (target_fraction * compression_ratio))
    return TrafficRequirement(
        target_fraction=float(target_fraction),
        compression_ratio=float(compression_ratio),
        minimum_committed_tokens=int(required),
    )


def verify_greedy_block(
    proposed_tokens: Sequence[int], target_tokens: Sequence[int]
) -> VerificationOutcome:
    """Verify a causal draft block against already-computed target argmaxes.

    `target_tokens[i]` is the exact greedy target token for the causal prefix
    containing proposed tokens before i.  The function commits the exact common
    prefix.  On a mismatch it also commits the exact target mismatch token,
    which requires one real incremental target call to materialize successor
    state.  On full acceptance it commits only the proposed block; no free bonus
    token is credited.
    """

    proposed = tuple(int(value) for value in proposed_tokens)
    target = tuple(int(value) for value in target_tokens)
    if not proposed:
        raise SpeculativeContractError("proposed block is empty")
    if len(target) < len(proposed):
        raise SpeculativeContractError("target token vector is too short")

    mismatch: int | None = None
    for index, token in enumerate(proposed):
        if token != target[index]:
            mismatch = index
            break

    if mismatch is None:
        accepted = len(proposed)
        committed = proposed
        repairs = 0
    else:
        accepted = mismatch
        committed = proposed[:accepted] + (target[mismatch],)
        repairs = 1

    return VerificationOutcome(
        proposed_tokens=proposed,
        target_tokens=target[: len(proposed)],
        accepted_draft_tokens=accepted,
        committed_tokens=len(committed),
        target_positions_evaluated=len(proposed),
        target_incremental_repair_calls=repairs,
        mismatch_index=mismatch,
        exact_committed_tokens=committed,
    )


def select_build_block_length(
    rows: Iterable[CaseAccounting],
    *,
    raw_required_tokens: int,
) -> int:
    """Select K using build rows only, with deterministic reality-first order.

    Prefer plans that satisfy the raw, no-compression traffic requirement on
    every build case.  Then minimize p95-like maximum latency ratio, median
    latency ratio, maximum N/A, and finally K.  If no K satisfies the raw byte
    requirement, retain the plan with the largest minimum committed population
    and then the same cost ordering.  No holdout observation may enter.
    """

    grouped: dict[int, list[CaseAccounting]] = {}
    for row in rows:
        grouped.setdefault(int(row.block_length), []).append(row)
    if not grouped:
        raise SpeculativeContractError("build population is empty")

    def key(item: tuple[int, list[CaseAccounting]]) -> tuple[float, ...]:
        block_length, cases = item
        min_committed = min(row.committed_tokens for row in cases)
        ratios = sorted(row.latency_ratio for row in cases)
        median_ratio = median(ratios)
        max_ratio = max(ratios)
        max_candidate_ratio = max(row.candidate_ratio for row in cases)
        exact = all(
            row.exact_token_match and row.exact_terminal_state_match
            for row in cases
        )
        raw_pass = exact and min_committed >= raw_required_tokens
        return (
            0.0 if raw_pass else 1.0,
            0.0 if exact else 1.0,
            -float(min_committed),
            float(max_ratio),
            float(median_ratio),
            float(max_candidate_ratio),
            float(block_length),
        )

    return min(grouped.items(), key=key)[0]


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise SpeculativeContractError("empty percentile population")
    if not 0.0 <= probability <= 1.0:
        raise SpeculativeContractError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_holdout_gate(
    rows: Sequence[CaseAccounting],
    *,
    selected_block_length: int,
    raw_required_tokens: int,
    p50_latency_limit: float,
    p95_latency_limit: float,
    p95_candidate_ratio_limit: float,
) -> dict[str, object]:
    selected = [
        row for row in rows if row.block_length == selected_block_length
    ]
    if not selected:
        raise SpeculativeContractError("selected holdout population is empty")

    exact = all(
        row.exact_token_match and row.exact_terminal_state_match
        for row in selected
    )
    committed = [row.committed_tokens for row in selected]
    latency = [row.latency_ratio for row in selected]
    candidate = [row.candidate_ratio for row in selected]

    metrics = {
        "case_count": len(selected),
        "minimum_committed_tokens": min(committed),
        "committed_tokens_p50": percentile(committed, 0.50),
        "committed_tokens_p95": percentile(committed, 0.95),
        "latency_ratio_p50": percentile(latency, 0.50),
        "latency_ratio_p95": percentile(latency, 0.95),
        "candidate_ratio_p95": percentile(candidate, 0.95),
        "exact_token_and_state": exact,
    }
    metrics["gate_passed"] = bool(
        exact
        and metrics["minimum_committed_tokens"] >= raw_required_tokens
        and metrics["latency_ratio_p50"] <= p50_latency_limit
        and metrics["latency_ratio_p95"] <= p95_latency_limit
        and metrics["candidate_ratio_p95"] <= p95_candidate_ratio_limit
    )
    return metrics
