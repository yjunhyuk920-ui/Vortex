from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from typing import Mapping

import torch


@dataclass(frozen=True)
class ModuleRankProfile:
    name: str
    input_features: int
    output_features: int
    numerical_rank: int
    maximum_rank: int
    bits: int
    column_bytes: float
    output_energy: float
    marginal_output_energy: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.output_features <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.numerical_rank <= 0 or self.maximum_rank <= 0:
            raise ValueError("profile ranks must be positive")
        if self.maximum_rank > self.numerical_rank:
            raise ValueError("maximum rank cannot exceed numerical rank")
        if len(self.marginal_output_energy) != self.maximum_rank:
            raise ValueError("marginal energy length must equal maximum rank")
        if self.bits <= 0 or self.column_bytes <= 0:
            raise ValueError("precision and column cost must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def captured_fraction(self, rank: int) -> float:
        if rank < 0 or rank > self.maximum_rank:
            raise ValueError("rank is outside the profiled range")
        captured = sum(self.marginal_output_energy[:rank])
        return captured / max(self.output_energy, 1e-30)


@dataclass(frozen=True)
class RankAllocation:
    ranks: dict[str, int]
    used_bytes: float
    byte_budget: float
    estimated_captured_output_fraction: float
    per_module_captured_fraction: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_module_rank_value(
    *,
    name: str,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    maximum_rank: int,
    bits: int,
    rank_rtol: float = 1e-6,
    scale_bytes_per_column: int = 4,
) -> ModuleRankProfile:
    """Measure prompt-output energy captured by each causal response-basis column.

    The profile uses only exact prompt-prefill inputs and outputs.  For
    ``X = U S V^T`` and exact bias-free outputs ``Y``, a rank-r response basis
    fitted from ``X`` reproduces the projection of ``Y`` onto the first r left
    singular vectors.  Therefore ``||u_i^T Y||_F^2`` is the exact training
    output-energy gain of adding column i, without reading continuation data.
    """

    if maximum_rank <= 0:
        raise ValueError("maximum_rank must be positive")
    if bits <= 0:
        raise ValueError("bits must be positive")
    if rank_rtol < 0:
        raise ValueError("rank_rtol must be non-negative")
    if input_tensor.shape[:-1] != output_tensor.shape[:-1]:
        raise ValueError("input/output leading dimensions must match")

    inputs = input_tensor.detach().to("cpu", torch.float32).reshape(
        -1, input_tensor.shape[-1]
    )
    outputs = output_tensor.detach().to("cpu", torch.float32).reshape(
        -1, output_tensor.shape[-1]
    )
    if inputs.shape[0] == 0:
        raise ValueError("at least one prompt vector is required")
    if bias is not None:
        outputs = outputs - bias.detach().to("cpu", torch.float32).reshape(1, -1)

    u, singular_values, _ = torch.linalg.svd(inputs, full_matrices=False)
    if singular_values.numel() == 0:
        raise RuntimeError("prompt activation SVD produced no singular values")
    threshold = float(singular_values[0].item()) * rank_rtol
    numerical_rank = int(torch.count_nonzero(singular_values > threshold).item())
    chosen = min(maximum_rank, numerical_rank, inputs.shape[1], inputs.shape[0])
    if chosen <= 0:
        raise RuntimeError("prompt activation matrix has zero numerical rank")

    coefficients = u[:, :chosen].T @ outputs
    gains = coefficients.square().sum(dim=1)
    output_energy = float(outputs.square().sum().item())
    column_bytes = (
        (inputs.shape[1] + outputs.shape[1]) * bits / 8.0
        + 2 * scale_bytes_per_column
    )
    return ModuleRankProfile(
        name=name,
        input_features=int(inputs.shape[1]),
        output_features=int(outputs.shape[1]),
        numerical_rank=numerical_rank,
        maximum_rank=chosen,
        bits=bits,
        column_bytes=float(column_bytes),
        output_energy=output_energy,
        marginal_output_energy=tuple(float(value) for value in gains.tolist()),
    )


def uniform_equivalent_byte_budget(
    profiles: Mapping[str, ModuleRankProfile],
    *,
    rank: int,
) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not profiles:
        raise ValueError("at least one profile is required")
    return sum(
        profile.column_bytes * min(rank, profile.maximum_rank)
        for profile in profiles.values()
    )


def allocate_nonuniform_ranks(
    profiles: Mapping[str, ModuleRankProfile],
    *,
    byte_budget: float,
    minimum_rank: int = 1,
) -> RankAllocation:
    """Allocate contiguous per-module ranks by marginal output gain per byte."""

    if not profiles:
        raise ValueError("at least one profile is required")
    if byte_budget <= 0:
        raise ValueError("byte_budget must be positive")
    if minimum_rank < 0:
        raise ValueError("minimum_rank must be non-negative")

    ranks = {
        name: min(minimum_rank, profile.maximum_rank)
        for name, profile in profiles.items()
    }
    used = sum(profiles[name].column_bytes * rank for name, rank in ranks.items())
    if used > byte_budget + 1e-6:
        raise ValueError("byte budget cannot fund the requested minimum rank")

    # Only the next contiguous column of each module is eligible.  Once it is
    # selected, the following column is pushed, preserving a valid prefix rank.
    heap: list[tuple[float, str, int]] = []

    def push_next(name: str) -> None:
        profile = profiles[name]
        index = ranks[name]
        if index >= profile.maximum_rank:
            return
        gain = profile.marginal_output_energy[index]
        value_per_byte = gain / profile.column_bytes
        heapq.heappush(heap, (-value_per_byte, name, index))

    for module_name in profiles:
        push_next(module_name)

    while heap:
        _negative_value, name, expected_index = heapq.heappop(heap)
        if ranks[name] != expected_index:
            continue
        cost = profiles[name].column_bytes
        if used + cost > byte_budget + 1e-6:
            continue
        ranks[name] += 1
        used += cost
        push_next(name)

    total_energy = sum(profile.output_energy for profile in profiles.values())
    captured = sum(
        sum(profiles[name].marginal_output_energy[:rank])
        for name, rank in ranks.items()
    )
    per_module = {
        name: profiles[name].captured_fraction(rank)
        for name, rank in ranks.items()
    }
    return RankAllocation(
        ranks=dict(sorted(ranks.items())),
        used_bytes=float(used),
        byte_budget=float(byte_budget),
        estimated_captured_output_fraction=float(
            captured / max(total_energy, 1e-30)
        ),
        per_module_captured_fraction=per_module,
    )
