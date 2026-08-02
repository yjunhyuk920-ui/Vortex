from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class TokenRoutedRefinementBudget:
    block_positions: int
    union_layer_fraction: float
    mean_token_layer_fraction: float
    consensus_bits: int
    residual_bits: int
    consensus_weight_gib: float
    union_residual_weight_gib: float
    transfer_seconds_per_block: float
    consensus_compute_seconds_per_block: float
    routed_residual_compute_seconds_per_block: float
    total_compute_seconds_per_block: float
    ideal_seconds_per_token: float
    serialized_seconds_per_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    required_overlap_fraction: float
    ideal_pass: bool
    serialized_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def token_routed_refinement_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    block_positions: int,
    union_layer_fraction: float,
    mean_token_layer_fraction: float,
    consensus_bits: int = 6,
    residual_bits: int = 2,
    host_to_device_gib_s: float = 24.0,
    consensus_effective_tops: float = 120.0,
    residual_effective_tops: float = 320.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> TokenRoutedRefinementBudget:
    """Budget a block whose residual layers differ by token.

    Residual transfer is charged once for the union of all layer groups used by
    any token in the block. Residual arithmetic is charged by the mean fraction
    of target layers actually traversed per block token. This separates the two
    costs that a uniform refinement model necessarily conflates.
    """

    if block_positions <= 0:
        raise ValueError("block_positions must be positive")
    if not 0 <= union_layer_fraction <= 1:
        raise ValueError("union_layer_fraction must be in [0, 1]")
    if not 0 <= mean_token_layer_fraction <= union_layer_fraction:
        raise ValueError(
            "mean_token_layer_fraction must be in [0, union_layer_fraction]"
        )
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
    union_residual_weight_gib = (
        target.parameters
        * residual_bits
        / 8
        / GIB
        * union_layer_fraction
    )
    transfer_seconds = (
        consensus_weight_gib + union_residual_weight_gib
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
    routed_residual_compute = (
        operations_per_token
        * block_positions
        * mean_token_layer_fraction
        / (residual_effective_tops * 1e12)
    )
    total_compute = consensus_compute + routed_residual_compute

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

    return TokenRoutedRefinementBudget(
        block_positions=block_positions,
        union_layer_fraction=union_layer_fraction,
        mean_token_layer_fraction=mean_token_layer_fraction,
        consensus_bits=consensus_bits,
        residual_bits=residual_bits,
        consensus_weight_gib=consensus_weight_gib,
        union_residual_weight_gib=union_residual_weight_gib,
        transfer_seconds_per_block=transfer_seconds,
        consensus_compute_seconds_per_block=consensus_compute,
        routed_residual_compute_seconds_per_block=routed_residual_compute,
        total_compute_seconds_per_block=total_compute,
        ideal_seconds_per_token=ideal_block_seconds / block_positions,
        serialized_seconds_per_token=serialized_block_seconds / block_positions,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_per_token,
        required_overlap_fraction=required_overlap,
        ideal_pass=ideal_block_seconds <= allowed_block_seconds,
        serialized_pass=serialized_block_seconds <= allowed_block_seconds,
    )
