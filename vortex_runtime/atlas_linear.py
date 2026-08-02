from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from safetensors.torch import load_file, save_file
import torch


@dataclass
class AtlasStats:
    calls: int = 0
    vectors: int = 0
    fast_vectors: int = 0
    cold_vectors: int = 0
    cold_weight_reads: int = 0
    weight_bytes_read: int = 0
    capsule_bytes_read: int = 0
    rejected_rank_limit: int = 0

    @property
    def fast_fraction(self) -> float:
        return self.fast_vectors / max(1, self.vectors)

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "fast_fraction": self.fast_fraction}


class OnlineAtlasLinear:
    """Exact-on-subspace linear operator with cold-weight expansion.

    The operator stores an orthonormal input basis ``U`` and its exact image
    ``WU``. For an input represented by that basis, ``W @ x`` is evaluated as
    ``(WU) @ (U.T @ x)`` without loading the original matrix.

    Inputs outside the current span fall back to the exact weight loader. The
    orthogonal residual becomes a new basis direction, allowing future inputs
    in the expanded span to use the fast path. The original weight is never
    discarded and remains the correctness fallback.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight_loader: Callable[[], torch.Tensor],
        max_rank: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        basis_dtype: torch.dtype = torch.float32,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if atol < 0 or rtol < 0:
            raise ValueError("tolerances must be non-negative")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_loader = weight_loader
        self.max_rank = min(int(max_rank), self.in_features)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.basis_dtype = basis_dtype
        self.input_basis = torch.empty((self.in_features, 0), dtype=basis_dtype)
        self.output_image = torch.empty((self.out_features, 0), dtype=basis_dtype)
        self.stats = AtlasStats()

    @property
    def rank(self) -> int:
        return self.input_basis.shape[1]

    @property
    def capsule_bytes(self) -> int:
        return (
            self.input_basis.numel() * self.input_basis.element_size()
            + self.output_image.numel() * self.output_image.element_size()
        )

    def _coordinates_and_residual(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        x_basis = x.to(dtype=self.basis_dtype, device="cpu")
        if self.rank == 0:
            residual = x_basis.clone()
            return torch.empty(0, dtype=self.basis_dtype), residual, float(
                torch.linalg.vector_norm(residual).item()
            )

        coordinates = self.input_basis.T @ x_basis
        residual = x_basis - self.input_basis @ coordinates
        correction = self.input_basis.T @ residual
        coordinates = coordinates + correction
        residual = residual - self.input_basis @ correction
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        return coordinates, residual, residual_norm

    def _is_fast(self, x: torch.Tensor, residual_norm: float) -> bool:
        x_norm = float(
            torch.linalg.vector_norm(x.to(dtype=self.basis_dtype, device="cpu")).item()
        )
        return residual_norm <= self.atol + self.rtol * x_norm

    def _append_direction(
        self,
        residual: torch.Tensor,
        residual_norm: float,
        exact_weight: torch.Tensor,
    ) -> None:
        if self.rank >= self.max_rank or residual_norm == 0.0:
            return
        direction = residual / residual_norm
        if self.rank:
            direction = direction - self.input_basis @ (
                self.input_basis.T @ direction
            )
            norm = torch.linalg.vector_norm(direction)
            if float(norm.item()) <= self.atol:
                return
            direction = direction / norm

        weight_basis = exact_weight.to(dtype=self.basis_dtype, device="cpu")
        image = weight_basis @ direction
        self.input_basis = torch.cat((self.input_basis, direction[:, None]), dim=1)
        self.output_image = torch.cat((self.output_image, image[:, None]), dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        self.stats.calls += 1
        original_shape = x.shape[:-1]
        flat = x.detach().reshape(-1, self.in_features)
        self.stats.vectors += flat.shape[0]

        outputs: list[torch.Tensor] = []
        exact_weight: torch.Tensor | None = None
        for row in flat:
            coordinates, residual, residual_norm = self._coordinates_and_residual(row)
            if self.rank and self._is_fast(row, residual_norm):
                result = self.output_image @ coordinates
                self.stats.fast_vectors += 1
                self.stats.capsule_bytes_read += self.capsule_bytes
                outputs.append(result.to(dtype=x.dtype, device=x.device))
                continue

            if exact_weight is None:
                exact_weight = self.weight_loader().detach().to("cpu").contiguous()
                if exact_weight.shape != (self.out_features, self.in_features):
                    raise ValueError(
                        "weight loader returned shape "
                        f"{tuple(exact_weight.shape)}, expected "
                        f"{(self.out_features, self.in_features)}"
                    )
                self.stats.cold_weight_reads += 1
                self.stats.weight_bytes_read += (
                    exact_weight.numel() * exact_weight.element_size()
                )

            exact_result = exact_weight.to(dtype=self.basis_dtype) @ row.to(
                dtype=self.basis_dtype, device="cpu"
            )
            self.stats.cold_vectors += 1
            if self.rank < self.max_rank:
                self._append_direction(residual, residual_norm, exact_weight)
            else:
                self.stats.rejected_rank_limit += 1
            outputs.append(exact_result.to(dtype=x.dtype, device=x.device))

        return torch.stack(outputs, dim=0).reshape(*original_shape, self.out_features)

    def save(self, directory: str | Path) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "input_basis": self.input_basis.contiguous(),
                "output_image": self.output_image.contiguous(),
            },
            output / "atlas.safetensors",
        )
        return output

    def load(self, directory: str | Path) -> None:
        payload = load_file(str(Path(directory) / "atlas.safetensors"))
        input_basis = payload["input_basis"].to(self.basis_dtype)
        output_image = payload["output_image"].to(self.basis_dtype)
        if input_basis.shape[0] != self.in_features:
            raise ValueError("saved atlas input dimension does not match")
        if output_image.shape[0] != self.out_features:
            raise ValueError("saved atlas output dimension does not match")
        if input_basis.shape[1] != output_image.shape[1]:
            raise ValueError("saved atlas rank is inconsistent")
        if input_basis.shape[1] > self.max_rank:
            raise ValueError("saved atlas rank exceeds configured maximum")
        self.input_basis = input_basis.contiguous()
        self.output_image = output_image.contiguous()
