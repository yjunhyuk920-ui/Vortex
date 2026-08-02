from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DiagonalTransportStats:
    rows: int
    cols: int
    iterations: int
    baseline_relative_l2_error: float
    adapted_relative_l2_error: float
    metadata_bytes: int
    input_scale_minimum: float
    input_scale_maximum: float
    output_scale_minimum: float
    output_scale_maximum: float

    @property
    def relative_error_reduction(self) -> float:
        if self.baseline_relative_l2_error <= 0:
            return 0.0
        return 1.0 - (
            self.adapted_relative_l2_error / self.baseline_relative_l2_error
        )

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["relative_error_reduction"] = self.relative_error_reduction
        return payload


def _relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def fit_diagonal_transport(
    *,
    target_weight: torch.Tensor,
    representative_weight: torch.Tensor,
    iterations: int = 6,
    epsilon: float = 1e-8,
    scale_limit: float = 16.0,
    metadata_bits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, DiagonalTransportStats]:
    """Fit ``W ~= diag(a) D diag(b)`` without training data.

    The alternating least-squares solve uses only the unchanged target and one
    resident representative weight. The two scale vectors are tiny compared
    with the matrix and can be applied around the shared matrix multiplication.
    """

    if target_weight.ndim != 2 or representative_weight.ndim != 2:
        raise ValueError("diagonal transport expects two-dimensional weights")
    if target_weight.shape != representative_weight.shape:
        raise ValueError("target and representative shapes must match")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if epsilon <= 0 or scale_limit <= 0 or metadata_bits <= 0:
        raise ValueError("epsilon, scale limit and metadata bits must be positive")

    target = target_weight.detach().to("cpu", torch.float32)
    representative = representative_weight.detach().to("cpu", torch.float32)
    rows, cols = target.shape
    input_scale = torch.ones(cols, dtype=torch.float32)
    output_scale = torch.ones(rows, dtype=torch.float32)

    for _ in range(iterations):
        row_design = representative * input_scale.unsqueeze(0)
        row_numerator = (target * row_design).sum(dim=1)
        row_denominator = row_design.square().sum(dim=1).clamp_min(epsilon)
        output_scale = (row_numerator / row_denominator).clamp(
            -scale_limit,
            scale_limit,
        )

        column_design = representative * output_scale.unsqueeze(1)
        column_numerator = (target * column_design).sum(dim=0)
        column_denominator = column_design.square().sum(dim=0).clamp_min(epsilon)
        input_scale = (column_numerator / column_denominator).clamp(
            -scale_limit,
            scale_limit,
        )

        rms = torch.sqrt(input_scale.square().mean().clamp_min(epsilon))
        input_scale = input_scale / rms
        output_scale = (output_scale * rms).clamp(-scale_limit, scale_limit)

    adapted = (
        representative
        * output_scale.unsqueeze(1)
        * input_scale.unsqueeze(0)
    )
    stats = DiagonalTransportStats(
        rows=rows,
        cols=cols,
        iterations=iterations,
        baseline_relative_l2_error=_relative_l2(target, representative),
        adapted_relative_l2_error=_relative_l2(target, adapted),
        metadata_bytes=(rows + cols) * metadata_bits // 8,
        input_scale_minimum=float(input_scale.min().item()),
        input_scale_maximum=float(input_scale.max().item()),
        output_scale_minimum=float(output_scale.min().item()),
        output_scale_maximum=float(output_scale.max().item()),
    )
    return input_scale, output_scale, stats


def materialize_diagonal_transport(
    *,
    representative_weight: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    if representative_weight.ndim != 2:
        raise ValueError("representative weight must be two-dimensional")
    rows, cols = representative_weight.shape
    if input_scale.numel() != cols or output_scale.numel() != rows:
        raise ValueError("transport scale dimension mismatch")
    return (
        representative_weight
        * output_scale.to(
            device=representative_weight.device,
            dtype=representative_weight.dtype,
        ).reshape(rows, 1)
        * input_scale.to(
            device=representative_weight.device,
            dtype=representative_weight.dtype,
        ).reshape(1, cols)
    )


def diagonal_transport_linear(
    x: torch.Tensor,
    *,
    representative_weight: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[-1] != representative_weight.shape[1]:
        raise ValueError("input feature dimension mismatch")
    scaled_input = x * input_scale.to(device=x.device, dtype=x.dtype)
    output = torch.nn.functional.linear(scaled_input, representative_weight, bias=None)
    output = output * output_scale.to(device=output.device, dtype=output.dtype)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output
