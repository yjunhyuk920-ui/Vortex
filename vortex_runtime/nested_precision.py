from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Mapping

import torch
from torch import nn

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.progressive_precision import (
    ModelPrecisionStats,
    TensorPrecisionStats,
)


@dataclass(frozen=True)
class NestedBitplaneBudget:
    base_bits: int
    maximum_bits: int
    block_positions: int
    maximum_reached_bits: int
    weight_stream_gib: float
    transfer_seconds_per_block: float
    base_compute_seconds_per_block: float
    incremental_compute_seconds_per_block: float
    total_compute_seconds_per_block: float
    fractions_reaching_bits: dict[int, float]
    ideal_seconds_per_token: float
    serialized_seconds_per_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    required_overlap_fraction: float
    ideal_pass: bool
    serialized_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def nested_symmetric_per_row_fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    maximum_bits: int = 8,
    source_bits: int = 16,
    name: str = "tensor",
    row_chunk: int = 256,
) -> tuple[torch.Tensor, TensorPrecisionStats]:
    """Quantize through one shared maximum-precision integer code.

    A signed `maximum_bits` code is produced once. Lower precision levels retain
    the sign and truncate low magnitude bitplanes toward zero. Consequently all
    levels are nested and each transition exposes exactly one additional stored
    bitplane per weight direction.
    """

    if tensor.ndim < 2:
        raise ValueError("nested quantization expects at least two dimensions")
    if not 2 <= bits <= maximum_bits < source_bits:
        raise ValueError("expected 2 <= bits <= maximum_bits < source_bits")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    source = tensor.detach().to("cpu", torch.float32)
    flat = source.reshape(source.shape[0], -1)
    restored_flat = torch.empty_like(flat)
    qmax = (1 << (maximum_bits - 1)) - 1
    shift = maximum_bits - bits
    step = 1 << shift
    source_square_sum = 0.0
    residual_square_sum = 0.0
    residual_absolute_sum = 0.0
    maximum_error = 0.0

    for start in range(0, flat.shape[0], row_chunk):
        end = min(start + row_chunk, flat.shape[0])
        chunk = flat[start:end]
        row_maximum = chunk.abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            row_maximum > 0,
            row_maximum / qmax,
            torch.ones_like(row_maximum),
        )
        maximum_code = torch.round(chunk / scale).clamp(-qmax, qmax)
        magnitude = maximum_code.abs().to(torch.int16)
        truncated_magnitude = torch.div(
            magnitude,
            step,
            rounding_mode="floor",
        ) * step
        nested_code = truncated_magnitude.to(torch.float32) * maximum_code.sign()
        restored = nested_code * scale
        residual = chunk - restored
        restored_flat[start:end].copy_(restored)
        source_square_sum += float(chunk.square().sum().item())
        residual_square_sum += float(residual.square().sum().item())
        residual_absolute_sum += float(residual.abs().sum().item())
        maximum_error = max(maximum_error, float(residual.abs().amax().item()))

    elements = source.numel()
    relative_error = (
        residual_square_sum / max(source_square_sum, 1e-24)
    ) ** 0.5
    stats = TensorPrecisionStats(
        name=name,
        elements=elements,
        hot_bits=bits,
        source_bits=source_bits,
        original_bytes=elements * source_bits / 8,
        hot_bytes=elements * bits / 8,
        residual_bitplane_bytes=elements * (source_bits - bits) / 8,
        relative_l2_error=relative_error,
        maximum_absolute_error=maximum_error,
        mean_absolute_error=residual_absolute_sum / elements,
    )
    return restored_flat.reshape_as(source), stats


def fake_quantize_nested_modules(
    model: nn.Module,
    *,
    bits: int,
    maximum_bits: int = 8,
    source_bits: int = 16,
    row_chunk: int = 256,
    module_types: Iterable[type[nn.Module]] = (nn.Linear, nn.Embedding),
) -> tuple[ModelPrecisionStats, list[TensorPrecisionStats]]:
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
            restored, stats = nested_symmetric_per_row_fake_quantize(
                weight,
                bits=bits,
                maximum_bits=maximum_bits,
                source_bits=source_bits,
                name=f"{name}.weight" if name else "weight",
                row_chunk=row_chunk,
            )
            weight.copy_(restored.to(device=weight.device, dtype=weight.dtype))
            per_tensor.append(stats)
            del restored

    if not per_tensor:
        raise RuntimeError("no eligible nested-precision weights were found")
    total_elements = sum(item.elements for item in per_tensor)
    squared_source_proxy = sum(item.elements for item in per_tensor)
    squared_residual_proxy = sum(
        item.elements * item.relative_l2_error**2 for item in per_tensor
    )
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
        maximum_absolute_error=max(item.maximum_absolute_error for item in per_tensor),
        weighted_mean_absolute_error=sum(
            item.elements * item.mean_absolute_error for item in per_tensor
        ) / total_elements,
    )
    return aggregate, per_tensor


def nested_bitplane_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    base_bits: int,
    maximum_bits: int,
    block_positions: int,
    fractions_reaching_bits: Mapping[int, float],
    host_to_device_gib_s: float = 24.0,
    base_effective_tops: float = 120.0,
    bitplane_effective_tops: float = 640.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> NestedBitplaneBudget:
    """Budget nested precision where later bitplanes touch token subsets only."""

    if not 2 <= base_bits < maximum_bits <= target.weight_bits:
        raise ValueError("invalid nested precision range")
    if block_positions <= 0:
        raise ValueError("block_positions must be positive")
    positive = (
        host_to_device_gib_s,
        base_effective_tops,
        bitplane_effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("hardware and target values must be positive")

    normalized: dict[int, float] = {}
    previous_fraction = 1.0
    for bits in range(base_bits + 1, maximum_bits + 1):
        fraction = float(fractions_reaching_bits.get(bits, 0.0))
        if not 0 <= fraction <= previous_fraction:
            raise ValueError("later bitplane fractions must be non-increasing")
        normalized[bits] = fraction
        previous_fraction = fraction
    maximum_reached = base_bits
    for bits, fraction in normalized.items():
        if fraction > 0:
            maximum_reached = bits

    weight_stream_gib = target.parameters * maximum_reached / 8 / GIB
    transfer_seconds = weight_stream_gib / host_to_device_gib_s
    operations_per_token = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    base_compute = (
        operations_per_token * block_positions / (base_effective_tops * 1e12)
    )
    incremental_compute = sum(
        operations_per_token
        * block_positions
        * fraction
        / (bitplane_effective_tops * 1e12)
        for fraction in normalized.values()
    )
    total_compute = base_compute + incremental_compute

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

    return NestedBitplaneBudget(
        base_bits=base_bits,
        maximum_bits=maximum_bits,
        block_positions=block_positions,
        maximum_reached_bits=maximum_reached,
        weight_stream_gib=weight_stream_gib,
        transfer_seconds_per_block=transfer_seconds,
        base_compute_seconds_per_block=base_compute,
        incremental_compute_seconds_per_block=incremental_compute,
        total_compute_seconds_per_block=total_compute,
        fractions_reaching_bits=normalized,
        ideal_seconds_per_token=ideal_block_seconds / block_positions,
        serialized_seconds_per_token=serialized_block_seconds / block_positions,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_per_token,
        required_overlap_fraction=required_overlap,
        ideal_pass=ideal_block_seconds <= allowed_block_seconds,
        serialized_pass=serialized_block_seconds <= allowed_block_seconds,
    )
