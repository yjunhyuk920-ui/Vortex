from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable

import torch
from torch import nn

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class TensorPrecisionStats:
    name: str
    elements: int
    hot_bits: int
    source_bits: int
    original_bytes: float
    hot_bytes: float
    residual_bitplane_bytes: float
    relative_l2_error: float
    maximum_absolute_error: float
    mean_absolute_error: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPrecisionStats:
    tensors: int
    elements: int
    hot_bits: int
    source_bits: int
    original_gib: float
    hot_gib: float
    residual_bitplane_gib: float
    element_weighted_relative_l2_error: float
    maximum_absolute_error: float
    weighted_mean_absolute_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class FullRankHotBudget:
    hot_bits: int
    block_positions: int
    hot_weight_gib: float
    hot_transfer_seconds_per_block: float
    hot_compute_seconds_per_block: float
    hot_compute_seconds_per_token: float
    ideal_seconds_per_token_at_full_commit: float
    serialized_seconds_per_token_at_full_commit: float
    baseline_seconds_per_token: float
    target_ratio: float
    minimum_full_commit_block_ideal: int | None
    minimum_full_commit_block_serialized: int | None
    ideal_pass: bool
    serialized_pass: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def symmetric_per_row_fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    source_bits: int = 16,
    name: str = "tensor",
) -> tuple[torch.Tensor, TensorPrecisionStats]:
    """Return a full-rank per-row symmetric fake-quantized tensor.

    The returned tensor remains floating point for portable diagnostics. The
    storage accounting treats the representation as a coarse `bits`-wide
    component plus the remaining exact source bitplanes. No row, column, or
    singular direction is removed.
    """

    if tensor.ndim < 2:
        raise ValueError("full-rank quantization expects at least two dimensions")
    if not 2 <= bits < source_bits:
        raise ValueError("bits must be in [2, source_bits)")
    if source_bits <= 0:
        raise ValueError("source_bits must be positive")

    source = tensor.detach().to("cpu", torch.float32)
    flat = source.reshape(source.shape[0], -1)
    qmax = (1 << (bits - 1)) - 1
    row_maximum = flat.abs().amax(dim=1, keepdim=True)
    scale = torch.where(
        row_maximum > 0,
        row_maximum / qmax,
        torch.ones_like(row_maximum),
    )
    quantized = torch.round(flat / scale).clamp(-qmax, qmax)
    restored = (quantized * scale).reshape_as(source)
    residual = source - restored
    source_norm = torch.linalg.vector_norm(source)
    residual_norm = torch.linalg.vector_norm(residual)
    relative_error = float(
        (residual_norm / torch.clamp(source_norm, min=1e-12)).item()
    )
    elements = source.numel()
    stats = TensorPrecisionStats(
        name=name,
        elements=elements,
        hot_bits=bits,
        source_bits=source_bits,
        original_bytes=elements * source_bits / 8,
        hot_bytes=elements * bits / 8,
        residual_bitplane_bytes=elements * (source_bits - bits) / 8,
        relative_l2_error=relative_error,
        maximum_absolute_error=float(residual.abs().amax().item()),
        mean_absolute_error=float(residual.abs().mean().item()),
    )
    return restored, stats


def fake_quantize_full_rank_modules(
    model: nn.Module,
    *,
    bits: int,
    source_bits: int = 16,
    module_types: Iterable[type[nn.Module]] = (nn.Linear, nn.Embedding),
) -> tuple[ModelPrecisionStats, list[TensorPrecisionStats]]:
    """Fake-quantize unique Linear/Embedding weights in place.

    Shared/tied tensors are quantized once by storage identity. Biases and norm
    parameters remain at source precision. This is an information-retention
    diagnostic, not yet an exact runtime: the original checkpoint remains the
    source of residual bitplanes for a later progressive repair stage.
    """

    selected_types = tuple(module_types)
    if not selected_types:
        raise ValueError("at least one module type is required")
    seen: set[tuple[int, int]] = set()
    per_tensor: list[TensorPrecisionStats] = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, selected_types):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                continue
            identity = (weight.untyped_storage().data_ptr(), weight.storage_offset())
            if identity in seen:
                continue
            seen.add(identity)
            restored, stats = symmetric_per_row_fake_quantize(
                weight,
                bits=bits,
                source_bits=source_bits,
                name=f"{name}.weight" if name else "weight",
            )
            weight.copy_(restored.to(device=weight.device, dtype=weight.dtype))
            per_tensor.append(stats)

    if not per_tensor:
        raise RuntimeError("no eligible full-rank weights were found")
    total_elements = sum(item.elements for item in per_tensor)
    squared_source_proxy = 0.0
    squared_residual_proxy = 0.0
    weighted_mae = 0.0
    maximum_error = 0.0
    for item in per_tensor:
        squared_source_proxy += item.elements
        squared_residual_proxy += item.elements * item.relative_l2_error**2
        weighted_mae += item.elements * item.mean_absolute_error
        maximum_error = max(maximum_error, item.maximum_absolute_error)
    aggregate = ModelPrecisionStats(
        tensors=len(per_tensor),
        elements=total_elements,
        hot_bits=bits,
        source_bits=source_bits,
        original_gib=total_elements * source_bits / 8 / GIB,
        hot_gib=total_elements * bits / 8 / GIB,
        residual_bitplane_gib=total_elements * (source_bits - bits) / 8 / GIB,
        element_weighted_relative_l2_error=(
            squared_residual_proxy / squared_source_proxy
        ) ** 0.5,
        maximum_absolute_error=maximum_error,
        weighted_mean_absolute_error=weighted_mae / total_elements,
    )
    return aggregate, per_tensor


def full_rank_hot_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    hot_bits: int,
    block_positions: int,
    host_to_device_gib_s: float,
    hot_effective_tops: float,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> FullRankHotBudget:
    """Compute the best-case roofline for a full-rank coarse-precision pass."""

    if not 2 <= hot_bits < target.weight_bits:
        raise ValueError("hot_bits must be below target source precision")
    if block_positions <= 0:
        raise ValueError("block_positions must be positive")
    if min(
        host_to_device_gib_s,
        hot_effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    ) <= 0:
        raise ValueError("hardware and target parameters must be positive")

    hot_weight_gib = target.parameters * hot_bits / 8 / GIB
    transfer_seconds = hot_weight_gib / host_to_device_gib_s
    target_operations_per_token = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    compute_seconds_per_token = target_operations_per_token / (
        hot_effective_tops * 1e12
    )
    compute_seconds = compute_seconds_per_token * block_positions
    ideal_per_token = max(transfer_seconds, compute_seconds) / block_positions
    serialized_per_token = transfer_seconds / block_positions + compute_seconds_per_token

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_flops = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_flops / (baseline_effective_tflops * 1e12)
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed = target_ratio * baseline_seconds

    if compute_seconds_per_token <= allowed:
        minimum_ideal = ceil(transfer_seconds / allowed)
    else:
        minimum_ideal = None
    if compute_seconds_per_token < allowed:
        minimum_serialized = ceil(
            transfer_seconds / (allowed - compute_seconds_per_token)
        )
    else:
        minimum_serialized = None

    return FullRankHotBudget(
        hot_bits=hot_bits,
        block_positions=block_positions,
        hot_weight_gib=hot_weight_gib,
        hot_transfer_seconds_per_block=transfer_seconds,
        hot_compute_seconds_per_block=compute_seconds,
        hot_compute_seconds_per_token=compute_seconds_per_token,
        ideal_seconds_per_token_at_full_commit=ideal_per_token,
        serialized_seconds_per_token_at_full_commit=serialized_per_token,
        baseline_seconds_per_token=baseline_seconds,
        target_ratio=target_ratio,
        minimum_full_commit_block_ideal=minimum_ideal,
        minimum_full_commit_block_serialized=minimum_serialized,
        ideal_pass=ideal_per_token <= allowed,
        serialized_pass=serialized_per_token <= allowed,
    )
