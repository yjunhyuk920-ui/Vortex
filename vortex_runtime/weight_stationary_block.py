from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class StreamedBlockHardware:
    """Hardware parameters for a layer-streamed exact block verifier.

    The target weights are assumed to live outside the 8 GiB device and to be
    streamed through the host-to-device link once per target-model pass. A pass
    evaluates every draft position as a batched matrix operation. Layer
    transfer and tensor-core work may overlap, so both an ideal-overlap and a
    conservative serialized bound are reported.
    """

    host_to_device_gib_s: float = 24.0
    target_tensor_tflops: float = 80.0
    baseline_gpu_memory_gib_s: float = 300.0
    baseline_tensor_tflops: float = 40.0

    def validate(self) -> None:
        values = (
            self.host_to_device_gib_s,
            self.target_tensor_tflops,
            self.baseline_gpu_memory_gib_s,
            self.baseline_tensor_tflops,
        )
        if any(value <= 0 for value in values):
            raise ValueError("hardware bandwidth and throughput values must be positive")


@dataclass(frozen=True)
class StreamedBlockBudget:
    draft_positions: int
    committed_tokens: int
    target_passes: int
    target_weight_gib: float
    target_flops_per_pass_tflop: float
    transfer_seconds_per_pass: float
    compute_seconds_per_pass: float
    ideal_overlap_seconds_per_pass: float
    serialized_seconds_per_pass: float
    ideal_seconds_per_committed_token: float
    serialized_seconds_per_committed_token: float
    baseline_seconds_per_token: float
    ideal_speed_ratio_to_baseline: float
    serialized_speed_ratio_to_baseline: float
    target_ratio: float
    ideal_pass: bool
    serialized_pass: bool
    minimum_committed_tokens_ideal: int
    minimum_committed_tokens_serialized: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def streamed_exact_block_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    draft_positions: int,
    committed_tokens: int,
    target_passes: int,
    hardware: StreamedBlockHardware = StreamedBlockHardware(),
    target_ratio: float = 1.2,
) -> StreamedBlockBudget:
    """Return a roofline time bound for exact weight-stationary block decoding.

    Unlike the earlier Gate 0 arithmetic-ratio proxy, this model allows batched
    target computation to use tensor cores while the 4B single-token baseline
    remains memory-bandwidth bound. No target weight approximation is assumed.
    The result is still a lower bound: tokenizer work, kernel launches, KV
    movement, dequantization, and draft construction are omitted.
    """

    hardware.validate()
    if draft_positions <= 0:
        raise ValueError("draft_positions must be positive")
    if committed_tokens <= 0 or committed_tokens > draft_positions:
        raise ValueError("committed_tokens must be in [1, draft_positions]")
    if target_passes <= 0:
        raise ValueError("target_passes must be positive")
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")

    target_weight_gib = target.weight_bytes / GIB
    target_flops_per_token = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    target_flops_per_pass = target_flops_per_token * draft_positions
    transfer_seconds = target_weight_gib / hardware.host_to_device_gib_s
    compute_seconds = target_flops_per_pass / (hardware.target_tensor_tflops * 1e12)
    ideal_pass_seconds = max(transfer_seconds, compute_seconds)
    serialized_pass_seconds = transfer_seconds + compute_seconds

    baseline_weight_seconds = (
        baseline.weight_bytes / GIB / hardware.baseline_gpu_memory_gib_s
    )
    baseline_flops_per_token = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_flops_per_token / (
        hardware.baseline_tensor_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)

    ideal_seconds_per_token = (
        ideal_pass_seconds * target_passes / committed_tokens
    )
    serialized_seconds_per_token = (
        serialized_pass_seconds * target_passes / committed_tokens
    )
    allowed_seconds = target_ratio * baseline_seconds

    return StreamedBlockBudget(
        draft_positions=draft_positions,
        committed_tokens=committed_tokens,
        target_passes=target_passes,
        target_weight_gib=target_weight_gib,
        target_flops_per_pass_tflop=target_flops_per_pass / 1e12,
        transfer_seconds_per_pass=transfer_seconds,
        compute_seconds_per_pass=compute_seconds,
        ideal_overlap_seconds_per_pass=ideal_pass_seconds,
        serialized_seconds_per_pass=serialized_pass_seconds,
        ideal_seconds_per_committed_token=ideal_seconds_per_token,
        serialized_seconds_per_committed_token=serialized_seconds_per_token,
        baseline_seconds_per_token=baseline_seconds,
        ideal_speed_ratio_to_baseline=ideal_seconds_per_token / baseline_seconds,
        serialized_speed_ratio_to_baseline=(
            serialized_seconds_per_token / baseline_seconds
        ),
        target_ratio=target_ratio,
        ideal_pass=ideal_seconds_per_token <= allowed_seconds,
        serialized_pass=serialized_seconds_per_token <= allowed_seconds,
        minimum_committed_tokens_ideal=ceil(
            ideal_pass_seconds * target_passes / allowed_seconds
        ),
        minimum_committed_tokens_serialized=ceil(
            serialized_pass_seconds * target_passes / allowed_seconds
        ),
    )


def jacobi_token_update(
    *,
    draft_tokens: torch.Tensor,
    prompt_next_token: torch.Tensor,
    draft_logits: torch.Tensor,
) -> torch.Tensor:
    """Apply one causal Jacobi token update to a fixed draft window.

    ``draft_logits[:, i]`` predicts the token after ``draft_tokens[:, i]``.
    Therefore the prompt logit fixes position zero and logits from positions
    ``0..K-2`` update positions ``1..K-1`` in parallel.
    """

    if draft_tokens.ndim != 2:
        raise ValueError("draft_tokens must have shape [batch, positions]")
    if draft_logits.ndim != 3:
        raise ValueError("draft_logits must have shape [batch, positions, vocab]")
    if draft_logits.shape[:2] != draft_tokens.shape:
        raise ValueError("draft token and logit leading dimensions must match")
    batch, positions = draft_tokens.shape
    if positions <= 0:
        raise ValueError("draft window must contain at least one position")

    prompt = prompt_next_token.reshape(-1)
    if prompt.numel() == 1 and batch > 1:
        prompt = prompt.expand(batch)
    if prompt.numel() != batch:
        raise ValueError("prompt_next_token must provide one token per batch")

    updated = torch.empty_like(draft_tokens)
    updated[:, 0] = prompt.to(device=draft_tokens.device, dtype=draft_tokens.dtype)
    if positions > 1:
        updated[:, 1:] = torch.argmax(draft_logits[:, :-1, :], dim=-1).to(
            dtype=draft_tokens.dtype
        )
    return updated


def longest_common_prefix(left: torch.Tensor, right: torch.Tensor) -> int:
    """Return the common prefix length for two single-sequence token tensors."""

    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.numel() != right_flat.numel():
        raise ValueError("token sequences must have equal length")
    unequal = torch.nonzero(left_flat != right_flat, as_tuple=False)
    if unequal.numel() == 0:
        return int(left_flat.numel())
    return int(unequal[0, 0].item())


def certified_fixed_prefix(previous: torch.Tensor, updated: torch.Tensor) -> int:
    """Return the exact greedy prefix certified by one Jacobi fixed-point step.

    For a deterministic causal decoder, a prefix unchanged by ``F`` is the
    unique greedy prefix: position zero is fixed by the prompt; each later
    unchanged token is conditioned only on the already-certified earlier
    prefix. This certificate does not inspect future reference tokens.
    """

    return longest_common_prefix(previous, updated)
