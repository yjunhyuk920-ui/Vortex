from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GatedProjectionStats:
    calls: int = 0
    rows: int = 0
    fast_rows: int = 0
    slow_rows: int = 0
    cold_weight_reads: int = 0
    cold_weight_bytes: int = 0
    capsule_bytes_read: int = 0

    @property
    def fast_fraction(self) -> float:
        return self.fast_rows / max(1, self.rows)

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "fast_fraction": self.fast_fraction}


def activation_basis(
    samples: torch.Tensor | Iterable[torch.Tensor],
    rank: int,
    *,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Return an orthonormal basis using the sample-side Gram matrix."""
    if isinstance(samples, torch.Tensor):
        matrix = samples
    else:
        parts = [part.reshape(-1, part.shape[-1]) for part in samples]
        if not parts:
            raise ValueError("at least one activation sample is required")
        matrix = torch.cat(parts, dim=0)
    if matrix.ndim != 2:
        raise ValueError("samples must have shape [rows, features]")
    if rank <= 0:
        raise ValueError("rank must be positive")
    matrix = matrix.detach().to(dtype=torch.float32, device="cpu")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("samples must be non-empty")

    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = eigenvalues > atol
    if not torch.any(keep):
        raise ValueError("activation samples have zero numerical rank")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    chosen = min(rank, eigenvalues.numel(), matrix.shape[1])
    eigenvalues = eigenvalues[:chosen]
    eigenvectors = eigenvectors[:, :chosen]
    basis = matrix.T @ (eigenvectors / torch.sqrt(eigenvalues)[None, :])
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis[:, :chosen].contiguous()


class GatedProjectedLinear(nn.Module):
    """Replace a real linear operation with projected fast and exact cold paths."""

    def __init__(
        self,
        *,
        basis: torch.Tensor,
        image: torch.Tensor,
        exact_weight: torch.Tensor,
        exact_bias: torch.Tensor | None,
        epsilon: float,
        offload_exact: bool = True,
    ) -> None:
        super().__init__()
        if basis.ndim != 2 or image.ndim != 2 or exact_weight.ndim != 2:
            raise ValueError("basis, image, and exact_weight must be matrices")
        if basis.shape[0] != exact_weight.shape[1]:
            raise ValueError("basis input width does not match exact weight")
        if image.shape != (exact_weight.shape[0], basis.shape[1]):
            raise ValueError("projected image has inconsistent shape")
        if exact_bias is not None and exact_bias.shape != (exact_weight.shape[0],):
            raise ValueError("bias has inconsistent shape")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        capsule_dtype = exact_weight.dtype
        self.register_buffer("basis", basis.to(dtype=capsule_dtype).contiguous())
        self.register_buffer("image", image.to(dtype=capsule_dtype).contiguous())
        self.register_buffer(
            "bias",
            None if exact_bias is None else exact_bias.detach().to("cpu").contiguous(),
            persistent=True,
        )
        exact_device = "cpu" if offload_exact else exact_weight.device
        self.register_buffer(
            "exact_weight",
            exact_weight.detach().to(exact_device).contiguous(),
            persistent=False,
        )
        self.epsilon = float(epsilon)
        self.offload_exact = bool(offload_exact)
        self.in_features = exact_weight.shape[1]
        self.out_features = exact_weight.shape[0]
        self.stats = GatedProjectionStats()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        basis: torch.Tensor,
        *,
        epsilon: float,
        offload_exact: bool = True,
    ) -> "GatedProjectedLinear":
        weight_cpu = linear.weight.detach().to(dtype=torch.float32, device="cpu")
        basis_cpu = basis.detach().to(dtype=torch.float32, device="cpu")
        image = weight_cpu @ basis_cpu
        module = cls(
            basis=basis_cpu,
            image=image,
            exact_weight=linear.weight.detach(),
            exact_bias=(None if linear.bias is None else linear.bias.detach()),
            epsilon=epsilon,
            offload_exact=offload_exact,
        )
        module.basis = module.basis.to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        module.image = module.image.to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        return module

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def capsule_bytes(self) -> int:
        total = self.basis.numel() * self.basis.element_size()
        total += self.image.numel() * self.image.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    @property
    def exact_weight_bytes(self) -> int:
        return self.exact_weight.numel() * self.exact_weight.element_size()

    def reset_stats(self) -> None:
        self.stats = GatedProjectionStats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {x.shape[-1]}"
            )
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        image = self.image.to(device=x.device, dtype=x.dtype)
        coordinates = flat @ basis
        input_norm_sq = flat.to(torch.float32).square().sum(dim=-1)
        coord_norm_sq = coordinates.to(torch.float32).square().sum(dim=-1)
        residual_sq = torch.clamp(input_norm_sq - coord_norm_sq, min=0.0)
        residual_ratio = torch.sqrt(residual_sq) / torch.sqrt(
            torch.clamp(input_norm_sq, min=torch.finfo(torch.float32).tiny)
        )
        fast_mask = residual_ratio <= self.epsilon
        result = coordinates @ image.T
        bias = None if self.bias is None else self.bias.to(x.device, x.dtype)
        if bias is not None:
            result = result + bias

        self.stats.calls += 1
        self.stats.rows += flat.shape[0]
        fast_rows = int(fast_mask.sum().item())
        slow_rows = flat.shape[0] - fast_rows
        self.stats.fast_rows += fast_rows
        self.stats.slow_rows += slow_rows
        self.stats.capsule_bytes_read += self.capsule_bytes

        if slow_rows:
            weight = self.exact_weight.to(device=x.device, dtype=x.dtype)
            exact = F.linear(flat[~fast_mask], weight, bias)
            result[~fast_mask] = exact
            self.stats.cold_weight_reads += 1
            self.stats.cold_weight_bytes += self.exact_weight_bytes
            if self.offload_exact and weight.device.type != "cpu":
                del weight

        return result.reshape(*original_shape, self.out_features)
