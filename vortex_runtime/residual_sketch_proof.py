from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.feasibility import GIB
from vortex_runtime.residual_proof import ArgmaxCertificate


@dataclass(frozen=True)
class ResidualSketchBudget:
    rows: int
    columns: int
    rank: int
    metadata_bits: int
    basis_elements: int
    coefficient_elements: int
    norm_elements: int
    metadata_elements: int
    metadata_gib: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class OrthogonalResidualSketch:
    basis: torch.Tensor
    coefficients: torch.Tensor
    remainder_row_norms: torch.Tensor
    rank: int
    relative_remainder_l2: float
    maximum_row_remainder_norm: float
    mean_row_remainder_norm: float

    def stats(self) -> dict[str, int | float]:
        return {
            "rank": self.rank,
            "relative_remainder_l2": self.relative_remainder_l2,
            "maximum_row_remainder_norm": self.maximum_row_remainder_norm,
            "mean_row_remainder_norm": self.mean_row_remainder_norm,
        }


def residual_sketch_budget(
    *,
    rows: int,
    columns: int,
    rank: int,
    metadata_bits: int = 32,
) -> ResidualSketchBudget:
    if min(rows, columns, rank, metadata_bits) <= 0:
        raise ValueError("dimensions, rank and metadata precision must be positive")
    if rank > columns:
        raise ValueError("rank cannot exceed the input dimension")
    basis = columns * rank
    coefficients = rows * rank
    norms = rows
    elements = basis + coefficients + norms
    return ResidualSketchBudget(
        rows=rows,
        columns=columns,
        rank=rank,
        metadata_bits=metadata_bits,
        basis_elements=basis,
        coefficient_elements=coefficients,
        norm_elements=norms,
        metadata_elements=elements,
        metadata_gib=elements * metadata_bits / 8 / GIB,
    )


def compile_orthogonal_residual_sketch(
    residual: torch.Tensor,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 0,
) -> OrthogonalResidualSketch:
    """Compile a sound low-rank residual correction plus remainder certificate.

    The shared orthonormal basis ``U`` approximates residual input directions.
    ``C = R U`` is retained exactly in the diagnostic. The remainder
    ``E = R - C U.T`` is not read at runtime; only ``||E[row]||_2`` is stored.
    Since ``E U = 0``, only the activation component orthogonal to ``U`` enters
    the remainder bound.
    """

    if residual.ndim != 2:
        raise ValueError("residual must have shape [rows, columns]")
    if rank <= 0 or rank > residual.shape[1]:
        raise ValueError("rank must be in [1, columns]")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations cannot be negative")

    source = residual.detach().to("cpu", torch.float32)
    rows, columns = source.shape
    sample_rank = min(columns, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(rows, sample_rank, generator=generator)
    subspace = source.T @ omega
    for _ in range(power_iterations):
        subspace = source.T @ (source @ subspace)
    basis_full, _ = torch.linalg.qr(subspace, mode="reduced")
    basis = basis_full[:, :rank].contiguous()
    coefficients = (source @ basis).contiguous()
    remainder = source - coefficients @ basis.T
    row_norms = torch.linalg.vector_norm(remainder, dim=1).contiguous()
    source_norm = torch.linalg.vector_norm(source).clamp_min(1e-24)
    relative = float((torch.linalg.vector_norm(remainder) / source_norm).item())
    return OrthogonalResidualSketch(
        basis=basis,
        coefficients=coefficients,
        remainder_row_norms=row_norms,
        rank=rank,
        relative_remainder_l2=relative,
        maximum_row_remainder_norm=float(row_norms.max().item()),
        mean_row_remainder_norm=float(row_norms.mean().item()),
    )


def apply_residual_sketch(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    sketch: OrthogonalResidualSketch,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if hot_logits.ndim != 1 or activation.ndim != 1:
        raise ValueError("hot_logits and activation must be one-dimensional")
    if sketch.basis.shape[0] != activation.numel():
        raise ValueError("basis input dimension does not match activation")
    if sketch.coefficients.shape[0] != hot_logits.numel():
        raise ValueError("coefficient output dimension does not match logits")

    x = activation.detach().to("cpu", torch.float32)
    projection = sketch.basis.T @ x
    correction = sketch.coefficients @ projection
    refined = hot_logits.detach().to("cpu", torch.float32) + correction
    perpendicular = x - sketch.basis @ projection
    perpendicular_norm = float(torch.linalg.vector_norm(perpendicular).item())
    effects = sketch.remainder_row_norms * perpendicular_norm
    return refined, effects, perpendicular_norm


def certify_sketch_argmax(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    sketch: OrthogonalResidualSketch,
) -> ArgmaxCertificate:
    refined, effects, _ = apply_residual_sketch(
        hot_logits=hot_logits,
        activation=activation,
        sketch=sketch,
    )
    candidate = int(torch.argmax(refined).item())
    lower = float((refined[candidate] - effects[candidate]).item())
    upper_values = refined + effects
    upper_values[candidate] = -torch.inf
    strongest = int(torch.argmax(upper_values).item())
    upper = float(upper_values[strongest].item())
    margin = lower - upper
    return ArgmaxCertificate(
        candidate=candidate,
        certified=margin > 0.0,
        certified_margin=margin,
        candidate_lower_bound=lower,
        strongest_competitor_upper_bound=upper,
        strongest_competitor=strongest,
    )
