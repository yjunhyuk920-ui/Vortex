from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Sequence

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class PrecisionConsensusRow:
    position: int
    exact_token: int
    q4_token: int
    q6_token: int
    q6_margin: float

    @property
    def q4_exact(self) -> bool:
        return self.q4_token == self.exact_token

    @property
    def q6_exact(self) -> bool:
        return self.q6_token == self.exact_token

    @property
    def agrees(self) -> bool:
        return self.q4_token == self.q6_token


@dataclass(frozen=True)
class ConsensusReport:
    tokens: int
    margin_threshold: float
    raw_agreement_tokens: int
    raw_agreement_exact_tokens: int
    raw_agreement_precision: float
    accepted_tokens: int
    accepted_exact_tokens: int
    accepted_precision: float
    accepted_error_tokens: int
    refinement_tokens: int
    refinement_fraction: float
    q4_errors: int
    q6_errors: int
    q4_errors_flagged: int
    q6_errors_flagged: int
    all_exact_errors_flagged: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressiveRefinementBudget:
    block_positions: int
    refinement_fraction: float
    refined_layer_fraction: float
    consensus_bits: int
    residual_bits: int
    consensus_weight_gib: float
    residual_weight_gib: float
    transfer_seconds_per_block: float
    consensus_compute_seconds_per_block: float
    residual_compute_seconds_per_block: float
    total_compute_seconds_per_block: float
    ideal_seconds_per_token: float
    serialized_seconds_per_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    required_overlap_fraction: float
    maximum_refinement_fraction_at_layer_fraction: float
    ideal_pass: bool
    serialized_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def analyze_precision_consensus(
    rows: Iterable[PrecisionConsensusRow],
    *,
    margin_threshold: float,
) -> ConsensusReport:
    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one consensus row is required")
    if margin_threshold < 0:
        raise ValueError("margin_threshold must be non-negative")

    accepted = [
        row
        for row in materialized
        if row.agrees and row.q6_margin >= margin_threshold
    ]
    refined = [row for row in materialized if row not in accepted]
    raw_agreement = [row for row in materialized if row.agrees]
    q4_errors = [row for row in materialized if not row.q4_exact]
    q6_errors = [row for row in materialized if not row.q6_exact]
    accepted_errors = [
        row for row in accepted if row.q6_token != row.exact_token
    ]

    return ConsensusReport(
        tokens=len(materialized),
        margin_threshold=margin_threshold,
        raw_agreement_tokens=len(raw_agreement),
        raw_agreement_exact_tokens=sum(row.q6_exact for row in raw_agreement),
        raw_agreement_precision=(
            sum(row.q6_exact for row in raw_agreement) / len(raw_agreement)
            if raw_agreement
            else 0.0
        ),
        accepted_tokens=len(accepted),
        accepted_exact_tokens=sum(row.q6_exact for row in accepted),
        accepted_precision=(
            sum(row.q6_exact for row in accepted) / len(accepted)
            if accepted
            else 0.0
        ),
        accepted_error_tokens=len(accepted_errors),
        refinement_tokens=len(refined),
        refinement_fraction=len(refined) / len(materialized),
        q4_errors=len(q4_errors),
        q6_errors=len(q6_errors),
        q4_errors_flagged=sum(row in refined for row in q4_errors),
        q6_errors_flagged=sum(row in refined for row in q6_errors),
        all_exact_errors_flagged=not accepted_errors,
    )


def sweep_consensus_thresholds(
    rows: Sequence[PrecisionConsensusRow],
    thresholds: Iterable[float],
) -> list[ConsensusReport]:
    reports = [
        analyze_precision_consensus(rows, margin_threshold=float(threshold))
        for threshold in thresholds
    ]
    if not reports:
        raise ValueError("at least one threshold is required")
    return reports


def progressive_refinement_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    block_positions: int,
    refinement_fraction: float,
    refined_layer_fraction: float,
    consensus_bits: int = 6,
    residual_bits: int = 2,
    host_to_device_gib_s: float = 24.0,
    consensus_effective_tops: float = 120.0,
    residual_effective_tops: float = 320.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> ProgressiveRefinementBudget:
    """Budget Q6 consensus plus Q8 residual work on uncertain token/layer pairs.

    The consensus pass processes every block position. The additional two-bit
    residual stream is charged once for the selected layer fraction, while its
    arithmetic is charged only for the uncertain token fraction. This is an
    optimistic roofline until a tiled overlap kernel is implemented.
    """

    if block_positions <= 0:
        raise ValueError("block_positions must be positive")
    if not 0 <= refinement_fraction <= 1:
        raise ValueError("refinement_fraction must be in [0, 1]")
    if not 0 <= refined_layer_fraction <= 1:
        raise ValueError("refined_layer_fraction must be in [0, 1]")
    if consensus_bits <= 0 or residual_bits <= 0:
        raise ValueError("precision bits must be positive")
    if consensus_bits + residual_bits > target.weight_bits:
        raise ValueError("progressive precision exceeds source precision")
    positive = (
        host_to_device_gib_s,
        consensus_effective_tops,
        residual_effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("hardware and target values must be positive")

    consensus_weight_gib = target.parameters * consensus_bits / 8 / GIB
    residual_weight_gib = (
        target.parameters
        * residual_bits
        / 8
        / GIB
        * refined_layer_fraction
    )
    transfer_seconds = (
        consensus_weight_gib + residual_weight_gib
    ) / host_to_device_gib_s

    operations_per_token = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    consensus_compute = (
        operations_per_token
        * block_positions
        / (consensus_effective_tops * 1e12)
    )
    residual_compute = (
        operations_per_token
        * block_positions
        * refinement_fraction
        * refined_layer_fraction
        / (residual_effective_tops * 1e12)
    )
    total_compute = consensus_compute + residual_compute

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_operations = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_operations / (
        baseline_effective_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_per_token = target_ratio * baseline_seconds
    allowed_block_seconds = allowed_per_token * block_positions

    ideal_block_seconds = max(transfer_seconds, total_compute)
    serialized_block_seconds = transfer_seconds + total_compute
    overlap_needed = max(0.0, serialized_block_seconds - allowed_block_seconds)
    overlap_capacity = min(transfer_seconds, total_compute)
    required_overlap = (
        overlap_needed / overlap_capacity
        if overlap_capacity > 0
        else (0.0 if overlap_needed <= 0 else inf)
    )

    residual_seconds_per_full_fraction = (
        operations_per_token
        * block_positions
        * refined_layer_fraction
        / (residual_effective_tops * 1e12)
    )
    compute_headroom = allowed_block_seconds - consensus_compute
    maximum_refinement = (
        max(0.0, min(1.0, compute_headroom / residual_seconds_per_full_fraction))
        if residual_seconds_per_full_fraction > 0
        else 1.0
    )

    return ProgressiveRefinementBudget(
        block_positions=block_positions,
        refinement_fraction=refinement_fraction,
        refined_layer_fraction=refined_layer_fraction,
        consensus_bits=consensus_bits,
        residual_bits=residual_bits,
        consensus_weight_gib=consensus_weight_gib,
        residual_weight_gib=residual_weight_gib,
        transfer_seconds_per_block=transfer_seconds,
        consensus_compute_seconds_per_block=consensus_compute,
        residual_compute_seconds_per_block=residual_compute,
        total_compute_seconds_per_block=total_compute,
        ideal_seconds_per_token=ideal_block_seconds / block_positions,
        serialized_seconds_per_token=serialized_block_seconds / block_positions,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_per_token,
        required_overlap_fraction=required_overlap,
        maximum_refinement_fraction_at_layer_fraction=maximum_refinement,
        ideal_pass=ideal_block_seconds <= allowed_block_seconds,
        serialized_pass=serialized_block_seconds <= allowed_block_seconds,
    )
