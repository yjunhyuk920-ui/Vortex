from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable

import torch
from torch import nn

from vortex_runtime.atlas_linear import OnlineAtlasLinear


@dataclass(frozen=True)
class ReplacementSnapshot:
    rank: int
    capsule_bytes: int
    vectors: int
    fast_vectors: int
    cold_vectors: int
    cold_weight_reads: int
    weight_bytes_read: int


@dataclass(frozen=True)
class RepairEfficiency:
    generated_tokens: int
    logical_cold_bytes: int
    managed_weight_bytes: int
    full_model_weight_bytes: int
    managed_repair_fraction: float
    full_model_repair_fraction: float
    zero_cold_reads: bool
    tokens_per_managed_repair_equivalent: float | None
    tokens_per_full_repair_equivalent: float | None

    @property
    def full_model_efficiency_for_gate(self) -> float:
        if self.zero_cold_reads:
            return inf
        assert self.tokens_per_full_repair_equivalent is not None
        return self.tokens_per_full_repair_equivalent

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


class AtlasLinearModule(nn.Module):
    """Drop-in nn.Linear falsification wrapper.

    Inputs inside the atlas span use the cached operator image. Span misses use
    the original exact weight and expand the atlas. The wrapper replaces the
    actual model operation, while its counters expose logical cold-weight use.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        max_rank: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.exact = linear
        self.atlas = OnlineAtlasLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_loader=lambda: self.exact.weight.detach(),
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
            basis_dtype=torch.float32,
        )

    @property
    def logical_weight_bytes(self) -> int:
        weight = self.exact.weight
        return weight.numel() * weight.element_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.atlas(x)
        if self.exact.bias is not None:
            output = output + self.exact.bias.to(
                dtype=output.dtype,
                device=output.device,
            )
        return output

    def snapshot(self) -> ReplacementSnapshot:
        stats = self.atlas.stats
        return ReplacementSnapshot(
            rank=self.atlas.rank,
            capsule_bytes=self.atlas.capsule_bytes,
            vectors=stats.vectors,
            fast_vectors=stats.fast_vectors,
            cold_vectors=stats.cold_vectors,
            cold_weight_reads=stats.cold_weight_reads,
            weight_bytes_read=stats.weight_bytes_read,
        )


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_modules(
    model: nn.Module,
    *,
    suffixes: Iterable[str],
    max_rank: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, AtlasLinearModule]:
    selected = tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in selected
        ):
            matches.append((name, module))

    replacements: dict[str, AtlasLinearModule] = {}
    for name, linear in matches:
        parent, attribute = _resolve_parent(model, name)
        wrapper = AtlasLinearModule(
            linear,
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
        )
        setattr(parent, attribute, wrapper)
        replacements[name] = wrapper
    return replacements


def snapshot_replacements(
    replacements: dict[str, AtlasLinearModule],
) -> dict[str, ReplacementSnapshot]:
    return {name: module.snapshot() for name, module in replacements.items()}


def replacement_delta(
    after: dict[str, ReplacementSnapshot],
    before: dict[str, ReplacementSnapshot],
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for name, current in after.items():
        previous = before.get(
            name,
            ReplacementSnapshot(0, 0, 0, 0, 0, 0, 0),
        )
        vectors = current.vectors - previous.vectors
        fast = current.fast_vectors - previous.fast_vectors
        result[name] = {
            "rank_before": previous.rank,
            "rank_after": current.rank,
            "rank_growth": current.rank - previous.rank,
            "capsule_bytes": current.capsule_bytes,
            "vectors": vectors,
            "fast_vectors": fast,
            "cold_vectors": current.cold_vectors - previous.cold_vectors,
            "cold_weight_reads": (
                current.cold_weight_reads - previous.cold_weight_reads
            ),
            "logical_cold_bytes": (
                current.weight_bytes_read - previous.weight_bytes_read
            ),
            "fast_fraction": fast / max(1, vectors),
        }
    return result


def compute_repair_efficiency(
    *,
    generated_tokens: int,
    logical_cold_bytes: int,
    managed_weight_bytes: int,
    full_model_weight_bytes: int,
) -> RepairEfficiency:
    if generated_tokens < 0:
        raise ValueError("generated_tokens must be non-negative")
    if min(logical_cold_bytes, managed_weight_bytes, full_model_weight_bytes) < 0:
        raise ValueError("byte counts must be non-negative")
    if managed_weight_bytes == 0 or full_model_weight_bytes == 0:
        raise ValueError("weight byte denominators must be positive")

    managed_fraction = logical_cold_bytes / managed_weight_bytes
    full_fraction = logical_cold_bytes / full_model_weight_bytes
    zero_cold_reads = logical_cold_bytes == 0
    return RepairEfficiency(
        generated_tokens=generated_tokens,
        logical_cold_bytes=logical_cold_bytes,
        managed_weight_bytes=managed_weight_bytes,
        full_model_weight_bytes=full_model_weight_bytes,
        managed_repair_fraction=managed_fraction,
        full_model_repair_fraction=full_fraction,
        zero_cold_reads=zero_cold_reads,
        tokens_per_managed_repair_equivalent=(
            None if zero_cold_reads else generated_tokens / managed_fraction
        ),
        tokens_per_full_repair_equivalent=(
            None if zero_cold_reads else generated_tokens / full_fraction
        ),
    )
