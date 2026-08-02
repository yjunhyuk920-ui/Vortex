from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Mapping

import torch

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


@dataclass(frozen=True)
class TensorQuantizationStats:
    bits: int
    elements: int
    logical_payload_bytes: int
    scale_bytes: int
    maximum_absolute_error: float
    relative_l2_error: float

    @property
    def logical_total_bytes(self) -> int:
        return self.logical_payload_bytes + self.scale_bytes

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "logical_total_bytes": self.logical_total_bytes,
        }


@dataclass(frozen=True)
class CapsuleQuantizationStats:
    bits: int
    modules: int
    tensors: int
    elements: int
    logical_payload_bytes: int
    scale_bytes: int
    maximum_absolute_error: float
    maximum_relative_l2_error: float

    @property
    def logical_total_bytes(self) -> int:
        return self.logical_payload_bytes + self.scale_bytes

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "logical_total_bytes": self.logical_total_bytes,
        }


def _relative_l2_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def fake_quantize_columns(
    tensor: torch.Tensor,
    *,
    bits: int,
    scale_bits: int = 16,
) -> tuple[torch.Tensor, TensorQuantizationStats]:
    """Symmetrically fake-quantize every response-basis column.

    The returned tensor is dequantized to the input dtype for existing PyTorch
    kernels. Accounting records the packed payload plus one signed scale per
    column. This separates numerical falsification from later packed-kernel
    implementation while preserving the declared storage/traffic contract.
    """

    if tensor.ndim != 2:
        raise ValueError("capsule tensor must be two-dimensional")
    if not 2 <= bits <= 16:
        raise ValueError("bits must be between 2 and 16")
    if scale_bits not in {16, 32}:
        raise ValueError("scale_bits must be 16 or 32")

    reference = tensor.detach().to("cpu", torch.float32)
    qmax = (1 << (bits - 1)) - 1
    maximum = reference.abs().amax(dim=0, keepdim=True)
    scale = maximum / max(1, qmax)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(reference / safe_scale).clamp(-qmax, qmax)
    dequantized = quantized * safe_scale
    dequantized = torch.where(maximum > 0, dequantized, torch.zeros_like(dequantized))

    elements = reference.numel()
    payload_bytes = ceil(elements * bits / 8)
    scale_bytes = reference.shape[1] * scale_bits // 8
    stats = TensorQuantizationStats(
        bits=bits,
        elements=elements,
        logical_payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        maximum_absolute_error=float(
            torch.max(torch.abs(reference - dequantized)).item()
        ),
        relative_l2_error=_relative_l2_error(reference, dequantized),
    )
    return dequantized.to(dtype=tensor.dtype), stats


def fake_quantize_response_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    bits: int,
    scale_bits: int = 16,
) -> tuple[CapsuleQuantizationStats, dict[str, dict[str, object]]]:
    if not replacements:
        raise ValueError("at least one response capsule is required")

    per_module: dict[str, dict[str, object]] = {}
    tensor_stats: list[TensorQuantizationStats] = []
    for name, module in replacements.items():
        basis, basis_stats = fake_quantize_columns(
            module.atlas.input_basis,
            bits=bits,
            scale_bits=scale_bits,
        )
        image, image_stats = fake_quantize_columns(
            module.atlas.output_image,
            bits=bits,
            scale_bits=scale_bits,
        )
        module.atlas.input_basis = basis.contiguous()
        module.atlas.output_image = image.contiguous()
        tensor_stats.extend((basis_stats, image_stats))
        per_module[name] = {
            "input_basis": basis_stats.to_dict(),
            "output_image": image_stats.to_dict(),
        }

    aggregate = CapsuleQuantizationStats(
        bits=bits,
        modules=len(replacements),
        tensors=len(tensor_stats),
        elements=sum(item.elements for item in tensor_stats),
        logical_payload_bytes=sum(
            item.logical_payload_bytes for item in tensor_stats
        ),
        scale_bytes=sum(item.scale_bytes for item in tensor_stats),
        maximum_absolute_error=max(
            item.maximum_absolute_error for item in tensor_stats
        ),
        maximum_relative_l2_error=max(
            item.relative_l2_error for item in tensor_stats
        ),
    )
    return aggregate, per_module
