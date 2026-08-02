from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.partitioned_signed_dual import (
    PartitionedSignedDualKernel,
    compile_partitioned_signed_dual_kernel,
)


@dataclass(frozen=True)
class QuantizedPartitionMetadataBudget:
    block_size: int
    blocks_per_hidden_vector: int
    norm_bits: int
    scale_bits: int
    norm_values: int
    scale_values: int
    metadata_gib: float
    metadata_limit_gib: float
    metadata_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def quantize_nonnegative_rows_upward(
    values: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize nonnegative row metadata without understating any bound.

    Each row stores one scale and unsigned integer codes. Codes use `ceil`, so
    every reconstructed value is greater than or equal to the source value.
    This is required for a sound certificate; round-to-nearest is not allowed.
    """

    source = values.detach().to("cpu", torch.float32)
    if source.ndim != 2 or torch.any(source < 0):
        raise ValueError("metadata must be a nonnegative matrix")
    if bits >= 32:
        return source.contiguous(), torch.ones(source.shape[0], dtype=torch.float32)
    if bits <= 0:
        raise ValueError("metadata bits must be positive")
    qmax = (1 << bits) - 1
    maximum = source.amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    codes = torch.ceil(source / scale).clamp(0, qmax)
    restored = codes * scale
    # One nextafter step protects against a downward float32 multiplication
    # rounding at the exact code*scale reconstruction boundary.
    restored = torch.nextafter(restored, torch.full_like(restored, torch.inf))
    return restored.contiguous(), scale[:, 0].contiguous()


def compile_quantized_partitioned_kernel(
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    bits: int,
    block_size: int,
    metadata_bits: int,
) -> PartitionedSignedDualKernel:
    kernel = compile_partitioned_signed_dual_kernel(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        bits=bits,
        block_size=block_size,
    )
    kernel.gate_block_residual_norms, _ = quantize_nonnegative_rows_upward(
        kernel.gate_block_residual_norms,
        bits=metadata_bits,
    )
    kernel.up_block_residual_norms, _ = quantize_nonnegative_rows_upward(
        kernel.up_block_residual_norms,
        bits=metadata_bits,
    )
    kernel.down_block_residual_norms, _ = quantize_nonnegative_rows_upward(
        kernel.down_block_residual_norms,
        bits=metadata_bits,
    )
    return kernel


def quantized_partition_metadata_budget(
    *,
    target: ModelSpec,
    block_size: int,
    norm_bits: int = 8,
    scale_bits: int = 16,
    metadata_limit_gib: float = 2.5,
) -> QuantizedPartitionMetadataBudget:
    if min(block_size, norm_bits, scale_bits) <= 0 or metadata_limit_gib <= 0:
        raise ValueError("metadata dimensions and limit must be positive")
    blocks = ceil(target.hidden_size / block_size)
    norm_values = target.layers * target.intermediate_size * 3 * blocks
    scale_values = target.layers * target.intermediate_size * 3
    metadata_gib = (
        norm_values * norm_bits / 8 + scale_values * scale_bits / 8
    ) / GIB
    return QuantizedPartitionMetadataBudget(
        block_size=block_size,
        blocks_per_hidden_vector=blocks,
        norm_bits=norm_bits,
        scale_bits=scale_bits,
        norm_values=norm_values,
        scale_values=scale_values,
        metadata_gib=metadata_gib,
        metadata_limit_gib=metadata_limit_gib,
        metadata_pass=metadata_gib <= metadata_limit_gib,
    )
