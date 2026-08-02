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


@dataclass(frozen=True)
class MixedCapsuleQuantizationStats:
    global_bits: int
    session_bits: int
    modules: int
    tensors: int
    global_columns: int
    session_columns: int
    global_elements: int
    session_elements: int
    global_payload_bytes: int
    session_payload_bytes: int
    scale_bytes: int
    maximum_global_relative_l2_error: float
    maximum_session_relative_l2_error: float
    maximum_absolute_error: float

    @property
    def logical_payload_bytes(self) -> int:
        return self.global_payload_bytes + self.session_payload_bytes

    @property
    def logical_total_bytes(self) -> int:
        return self.logical_payload_bytes + self.scale_bytes

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "logical_payload_bytes": self.logical_payload_bytes,
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
    if tensor.shape[1] <= 0:
        raise ValueError("capsule tensor must contain at least one column")
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


def _quantize_group(
    tensor: torch.Tensor,
    *,
    start: int,
    end: int,
    bits: int,
    scale_bits: int,
) -> tuple[torch.Tensor, TensorQuantizationStats | None]:
    if start == end:
        return tensor[:, start:end].clone(), None
    return fake_quantize_columns(
        tensor[:, start:end],
        bits=bits,
        scale_bits=scale_bits,
    )


def fake_quantize_mixed_response_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    global_ranks: Mapping[str, int],
    global_bits: int,
    session_bits: int,
    scale_bits: int = 16,
) -> tuple[MixedCapsuleQuantizationStats, dict[str, dict[str, object]]]:
    """Quantize global and prompt-derived columns at different precision.

    Hybrid capsules append session directions after the existing global prior.
    ``global_ranks`` therefore defines the split column for every module. Both
    ``U`` and ``WU`` use the same split so their column semantics remain aligned.
    """

    if not replacements:
        raise ValueError("at least one response capsule is required")
    if set(global_ranks) != set(replacements):
        raise ValueError("global-rank map must match replacement modules")
    if not 2 <= global_bits <= 16 or not 2 <= session_bits <= 16:
        raise ValueError("mixed capsule bits must be between 2 and 16")

    per_module: dict[str, dict[str, object]] = {}
    global_stats: list[TensorQuantizationStats] = []
    session_stats: list[TensorQuantizationStats] = []
    for name, module in replacements.items():
        rank = module.atlas.rank
        split = int(global_ranks[name])
        if split < 0 or split > rank:
            raise ValueError(f"invalid global/session split for {name}")

        tensor_reports: dict[str, object] = {}
        quantized_tensors: list[torch.Tensor] = []
        for tensor_name, tensor in (
            ("input_basis", module.atlas.input_basis),
            ("output_image", module.atlas.output_image),
        ):
            global_tensor, global_report = _quantize_group(
                tensor,
                start=0,
                end=split,
                bits=global_bits,
                scale_bits=scale_bits,
            )
            session_tensor, session_report = _quantize_group(
                tensor,
                start=split,
                end=rank,
                bits=session_bits,
                scale_bits=scale_bits,
            )
            quantized_tensors.append(
                torch.cat((global_tensor, session_tensor), dim=1).contiguous()
            )
            if global_report is not None:
                global_stats.append(global_report)
            if session_report is not None:
                session_stats.append(session_report)
            tensor_reports[tensor_name] = {
                "global": (
                    None if global_report is None else global_report.to_dict()
                ),
                "session": (
                    None if session_report is None else session_report.to_dict()
                ),
            }

        module.atlas.input_basis = quantized_tensors[0]
        module.atlas.output_image = quantized_tensors[1]
        per_module[name] = {
            "global_rank": split,
            "session_rank": rank - split,
            **tensor_reports,
        }

    all_stats = global_stats + session_stats
    if not all_stats:
        raise RuntimeError("mixed quantization produced no tensor groups")
    aggregate = MixedCapsuleQuantizationStats(
        global_bits=global_bits,
        session_bits=session_bits,
        modules=len(replacements),
        tensors=len(replacements) * 2,
        global_columns=sum(int(global_ranks[name]) for name in replacements),
        session_columns=sum(
            replacements[name].atlas.rank - int(global_ranks[name])
            for name in replacements
        ),
        global_elements=sum(item.elements for item in global_stats),
        session_elements=sum(item.elements for item in session_stats),
        global_payload_bytes=sum(
            item.logical_payload_bytes for item in global_stats
        ),
        session_payload_bytes=sum(
            item.logical_payload_bytes for item in session_stats
        ),
        scale_bytes=sum(item.scale_bytes for item in all_stats),
        maximum_global_relative_l2_error=max(
            (item.relative_l2_error for item in global_stats),
            default=0.0,
        ),
        maximum_session_relative_l2_error=max(
            (item.relative_l2_error for item in session_stats),
            default=0.0,
        ),
        maximum_absolute_error=max(
            item.maximum_absolute_error for item in all_stats
        ),
    )
    return aggregate, per_module
