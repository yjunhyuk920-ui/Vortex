from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.feasibility import GIB
from vortex_runtime.residual_proof import ArgmaxCertificate


@dataclass(frozen=True)
class ActivationProofAtlasBudget:
    rows: int
    columns: int
    rank: int
    metadata_bits: int
    basis_elements: int
    image_elements: int
    remainder_norm_elements: int
    metadata_elements: int
    metadata_gib: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationProofAtlas:
    basis: torch.Tensor
    residual_images: torch.Tensor
    remainder_row_norms: torch.Tensor
    rank: int
    build_samples: int
    build_max_perpendicular_ratio: float
    build_mean_perpendicular_ratio: float
    relative_residual_remainder_l2: float

    def stats(self) -> dict[str, int | float]:
        return {
            "rank": self.rank,
            "build_samples": self.build_samples,
            "build_max_perpendicular_ratio": self.build_max_perpendicular_ratio,
            "build_mean_perpendicular_ratio": self.build_mean_perpendicular_ratio,
            "relative_residual_remainder_l2": self.relative_residual_remainder_l2,
        }


def activation_proof_atlas_budget(
    *,
    rows: int,
    columns: int,
    rank: int,
    metadata_bits: int = 32,
) -> ActivationProofAtlasBudget:
    if min(rows, columns, rank, metadata_bits) <= 0:
        raise ValueError("dimensions, rank and precision must be positive")
    if rank > columns:
        raise ValueError("rank cannot exceed columns")
    basis_elements = columns * rank
    image_elements = rows * rank
    norm_elements = rows
    total = basis_elements + image_elements + norm_elements
    return ActivationProofAtlasBudget(
        rows=rows,
        columns=columns,
        rank=rank,
        metadata_bits=metadata_bits,
        basis_elements=basis_elements,
        image_elements=image_elements,
        remainder_norm_elements=norm_elements,
        metadata_elements=total,
        metadata_gib=total * metadata_bits / 8 / GIB,
    )


def compile_activation_proof_atlas(
    *,
    residual: torch.Tensor,
    build_activations: torch.Tensor,
    rank: int,
) -> ActivationProofAtlas:
    """Compile an exact-on-prompt residual atlas and a sound remainder proof.

    The basis is derived only from causally available prompt-prefill activation
    vectors. ``residual_images = R U`` makes residual correction exact for every
    vector in the basis span. The unread remainder is bounded rowwise against
    the continuation component orthogonal to that span.
    """

    if residual.ndim != 2:
        raise ValueError("residual must have shape [rows, columns]")
    if build_activations.ndim != 2:
        raise ValueError("build_activations must have shape [samples, columns]")
    rows, columns = residual.shape
    if build_activations.shape[1] != columns:
        raise ValueError("activation dimension does not match residual columns")
    if not 0 < rank <= min(columns, build_activations.shape[0]):
        raise ValueError("rank must not exceed columns or build samples")

    source = residual.detach().to("cpu", torch.float32)
    activations = build_activations.detach().to("cpu", torch.float32)
    basis_full, _ = torch.linalg.qr(activations.T, mode="reduced")
    basis = basis_full[:, :rank].contiguous()
    images = (source @ basis).contiguous()
    remainder = source - images @ basis.T
    row_norms = torch.linalg.vector_norm(remainder, dim=1).contiguous()

    projection = activations @ basis @ basis.T
    perpendicular = activations - projection
    activation_norms = torch.linalg.vector_norm(activations, dim=1).clamp_min(1e-24)
    ratios = torch.linalg.vector_norm(perpendicular, dim=1) / activation_norms
    source_norm = torch.linalg.vector_norm(source).clamp_min(1e-24)
    remainder_ratio = float(
        (torch.linalg.vector_norm(remainder) / source_norm).item()
    )
    return ActivationProofAtlas(
        basis=basis,
        residual_images=images,
        remainder_row_norms=row_norms,
        rank=rank,
        build_samples=int(activations.shape[0]),
        build_max_perpendicular_ratio=float(ratios.max().item()),
        build_mean_perpendicular_ratio=float(ratios.mean().item()),
        relative_residual_remainder_l2=remainder_ratio,
    )


def apply_activation_proof_atlas(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    atlas: ActivationProofAtlas,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if hot_logits.ndim != 1 or activation.ndim != 1:
        raise ValueError("hot_logits and activation must be one-dimensional")
    if atlas.basis.shape[0] != activation.numel():
        raise ValueError("basis dimension does not match activation")
    if atlas.residual_images.shape[0] != hot_logits.numel():
        raise ValueError("residual image rows do not match logits")

    x = activation.detach().to("cpu", torch.float32)
    coordinates = atlas.basis.T @ x
    correction = atlas.residual_images @ coordinates
    refined = hot_logits.detach().to("cpu", torch.float32) + correction
    perpendicular = x - atlas.basis @ coordinates
    perpendicular_norm = float(torch.linalg.vector_norm(perpendicular).item())
    total_norm = float(torch.linalg.vector_norm(x).clamp_min(1e-24).item())
    perpendicular_ratio = perpendicular_norm / total_norm
    effects = atlas.remainder_row_norms * perpendicular_norm
    return refined, effects, perpendicular_norm, perpendicular_ratio


def certify_activation_atlas_argmax(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    atlas: ActivationProofAtlas,
) -> ArgmaxCertificate:
    refined, effects, _, _ = apply_activation_proof_atlas(
        hot_logits=hot_logits,
        activation=activation,
        atlas=atlas,
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
