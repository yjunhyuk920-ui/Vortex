from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class LowRankResidualStats:
    rows: int
    cols: int
    rank: int
    oversample: int
    power_iterations: int
    baseline_relative_l2_error: float
    corrected_relative_l2_error: float
    factor_elements: int
    factor_bytes: int
    factor_bits: int

    @property
    def relative_error_reduction(self) -> float:
        if self.baseline_relative_l2_error <= 0:
            return 0.0
        return 1.0 - (
            self.corrected_relative_l2_error / self.baseline_relative_l2_error
        )

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["relative_error_reduction"] = self.relative_error_reduction
        return payload


def _relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def _fake_quantize_columns(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    if not 2 <= bits <= 16:
        raise ValueError("factor bits must be in [2, 16]")
    levels = (1 << (bits - 1)) - 1
    maximum = tensor.abs().amax(dim=0, keepdim=True)
    scale = torch.where(maximum > 0, maximum / levels, torch.ones_like(maximum))
    quantized = torch.clamp(torch.round(tensor / scale), -levels, levels)
    return quantized * scale


def fit_randomized_low_rank_residual(
    *,
    target_weight: torch.Tensor,
    base_weight: torch.Tensor,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 0,
    factor_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, LowRankResidualStats]:
    """Fit a deterministic randomized low-rank correction to a weight residual.

    ``target ~= base + left @ right.T``

    The randomized range finder touches only checkpoint weights. It is an
    automatic representation build step, not gradient training or activation
    calibration. Returned factors are fake-quantized column-wise to the storage
    precision used by the 405B memory certificate.
    """

    if target_weight.ndim != 2 or base_weight.ndim != 2:
        raise ValueError("low-rank residual expects two-dimensional weights")
    if target_weight.shape != base_weight.shape:
        raise ValueError("target and base shapes must match")
    rows, cols = target_weight.shape
    maximum_rank = min(rows, cols)
    if not 0 < rank <= maximum_rank:
        raise ValueError("rank must be in [1, min(rows, cols)]")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power iterations must be non-negative")

    target = target_weight.detach().to("cpu", torch.float32)
    base = base_weight.detach().to("cpu", torch.float32)
    residual = target - base
    sample_rank = min(maximum_rank, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(cols, sample_rank, generator=generator)
    sample = residual @ omega
    for _ in range(power_iterations):
        sample = residual @ (residual.T @ sample)
    basis = torch.linalg.qr(sample, mode="reduced").Q
    small = basis.T @ residual
    left_small, singular_values, right_transpose = torch.linalg.svd(
        small,
        full_matrices=False,
    )
    effective_rank = min(rank, singular_values.numel())
    left = basis @ left_small[:, :effective_rank]
    left = left * singular_values[:effective_rank].reshape(1, -1)
    right = right_transpose[:effective_rank, :].T.contiguous()

    left = _fake_quantize_columns(left, bits=factor_bits).contiguous()
    right = _fake_quantize_columns(right, bits=factor_bits).contiguous()
    corrected = base + left @ right.T
    factor_elements = left.numel() + right.numel()
    stats = LowRankResidualStats(
        rows=rows,
        cols=cols,
        rank=effective_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        baseline_relative_l2_error=_relative_l2(target, base),
        corrected_relative_l2_error=_relative_l2(target, corrected),
        factor_elements=factor_elements,
        factor_bytes=(factor_elements * factor_bits + 7) // 8,
        factor_bits=factor_bits,
    )
    return left, right, stats


def materialize_low_rank_correction(
    *,
    base_weight: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if base_weight.ndim != 2 or left.ndim != 2 or right.ndim != 2:
        raise ValueError("base and factors must be two-dimensional")
    if left.shape[0] != base_weight.shape[0]:
        raise ValueError("left factor output dimension mismatch")
    if right.shape[0] != base_weight.shape[1]:
        raise ValueError("right factor input dimension mismatch")
    if left.shape[1] != right.shape[1]:
        raise ValueError("factor ranks must match")
    return base_weight + left.to(
        device=base_weight.device,
        dtype=base_weight.dtype,
    ) @ right.to(
        device=base_weight.device,
        dtype=base_weight.dtype,
    ).T


def low_rank_corrected_linear(
    x: torch.Tensor,
    *,
    base_weight: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the shared base GEMM plus two thin residual GEMMs."""

    if x.shape[-1] != base_weight.shape[1]:
        raise ValueError("input feature dimension mismatch")
    base_output = torch.nn.functional.linear(x, base_weight, bias=None)
    residual_coordinates = torch.nn.functional.linear(x, right.T, bias=None)
    residual_output = torch.nn.functional.linear(
        residual_coordinates,
        left,
        bias=None,
    )
    output = base_output + residual_output
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output
