from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.activation_proof_atlas import (
    ActivationProofAtlas,
    compile_activation_proof_atlas,
)
from vortex_runtime.feasibility import GIB
from vortex_runtime.residual_proof import ArgmaxCertificate


@dataclass(frozen=True)
class OnlineAtlasTrafficBudget:
    rows: int
    columns: int
    source_bits: int
    hot_bits: int
    residual_stream_gib_per_expansion: float
    expansions: int
    tokens: int
    amortized_residual_gib_per_token: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def online_atlas_traffic_budget(
    *,
    rows: int,
    columns: int,
    expansions: int,
    tokens: int,
    source_bits: int = 16,
    hot_bits: int = 4,
) -> OnlineAtlasTrafficBudget:
    if min(rows, columns, tokens, source_bits, hot_bits) <= 0:
        raise ValueError("dimensions, tokens and precision must be positive")
    if expansions < 0:
        raise ValueError("expansions cannot be negative")
    if hot_bits >= source_bits:
        raise ValueError("hot_bits must be below source_bits")
    stream_gib = rows * columns * (source_bits - hot_bits) / 8 / GIB
    return OnlineAtlasTrafficBudget(
        rows=rows,
        columns=columns,
        source_bits=source_bits,
        hot_bits=hot_bits,
        residual_stream_gib_per_expansion=stream_gib,
        expansions=expansions,
        tokens=tokens,
        amortized_residual_gib_per_token=stream_gib * expansions / tokens,
    )


class OnlineActivationProofAtlas:
    """Incrementally exact residual atlas with a sound orthogonal remainder.

    A proof miss adds the current activation's component orthogonal to the atlas
    as a new basis direction. Computing ``R u`` requires one residual matrix
    stream, but future activations may reuse that exact image. Rowwise remainder
    norms update exactly by Pythagoras because the new direction is orthogonal to
    all existing basis columns.
    """

    def __init__(
        self,
        *,
        basis: torch.Tensor,
        residual_images: torch.Tensor,
        remainder_row_norms: torch.Tensor,
    ) -> None:
        if basis.ndim != 2 or residual_images.ndim != 2:
            raise ValueError("basis and residual_images must be matrices")
        if remainder_row_norms.ndim != 1:
            raise ValueError("remainder_row_norms must be one-dimensional")
        if residual_images.shape[1] != basis.shape[1]:
            raise ValueError("basis and residual image ranks must match")
        if residual_images.shape[0] != remainder_row_norms.numel():
            raise ValueError("one remainder norm is required per output row")
        self.basis = basis.detach().to("cpu", torch.float32).contiguous()
        self.residual_images = (
            residual_images.detach().to("cpu", torch.float32).contiguous()
        )
        self.remainder_row_norm_sq = (
            remainder_row_norms.detach().to("cpu", torch.float32).square()
        )
        self.expansions = 0

    @classmethod
    def from_prompt(
        cls,
        *,
        residual: torch.Tensor,
        build_activations: torch.Tensor,
        rank: int,
    ) -> "OnlineActivationProofAtlas":
        atlas = compile_activation_proof_atlas(
            residual=residual,
            build_activations=build_activations,
            rank=rank,
        )
        return cls(
            basis=atlas.basis,
            residual_images=atlas.residual_images,
            remainder_row_norms=atlas.remainder_row_norms,
        )

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def remainder_row_norms(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.remainder_row_norm_sq, min=0.0))

    def as_static(self) -> ActivationProofAtlas:
        return ActivationProofAtlas(
            basis=self.basis,
            residual_images=self.residual_images,
            remainder_row_norms=self.remainder_row_norms,
            rank=self.rank,
            build_samples=self.rank,
            build_max_perpendicular_ratio=0.0,
            build_mean_perpendicular_ratio=0.0,
            relative_residual_remainder_l2=float("nan"),
        )

    def project(
        self,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        x = activation.detach().to("cpu", torch.float32)
        coordinates = self.basis.T @ x
        perpendicular = x - self.basis @ coordinates
        perpendicular_norm = float(torch.linalg.vector_norm(perpendicular).item())
        total_norm = float(torch.linalg.vector_norm(x).clamp_min(1e-24).item())
        return coordinates, perpendicular, perpendicular_norm, perpendicular_norm / total_norm

    def apply(
        self,
        *,
        hot_logits: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        coordinates, _, perpendicular_norm, ratio = self.project(activation)
        correction = self.residual_images @ coordinates
        refined = hot_logits.detach().to("cpu", torch.float32) + correction
        effects = self.remainder_row_norms * perpendicular_norm
        return refined, effects, perpendicular_norm, ratio

    def certify(
        self,
        *,
        hot_logits: torch.Tensor,
        activation: torch.Tensor,
    ) -> ArgmaxCertificate:
        refined, effects, _, _ = self.apply(
            hot_logits=hot_logits,
            activation=activation,
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

    def expand(
        self,
        *,
        activation: torch.Tensor,
        residual: torch.Tensor,
        minimum_perpendicular_norm: float = 1e-7,
    ) -> bool:
        if residual.ndim != 2:
            raise ValueError("residual must have shape [rows, columns]")
        if residual.shape != (
            self.residual_images.shape[0],
            self.basis.shape[0],
        ):
            raise ValueError("residual shape does not match atlas")
        _, perpendicular, perpendicular_norm, _ = self.project(activation)
        if perpendicular_norm <= minimum_perpendicular_norm:
            return False
        direction = perpendicular / perpendicular_norm
        image = residual.detach().to("cpu", torch.float32) @ direction
        self.basis = torch.cat((self.basis, direction[:, None]), dim=1).contiguous()
        self.residual_images = torch.cat(
            (self.residual_images, image[:, None]),
            dim=1,
        ).contiguous()
        self.remainder_row_norm_sq = torch.clamp(
            self.remainder_row_norm_sq - image.square(),
            min=0.0,
        )
        self.expansions += 1
        return True
