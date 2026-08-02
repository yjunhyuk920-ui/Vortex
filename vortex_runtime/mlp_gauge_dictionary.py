from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from vortex_runtime.mlp_neuron_dictionary import (
    CompiledSwiGLUDictionary,
    MLPDictionaryFitStats,
    compile_swiglu_dictionary,
)


@dataclass(frozen=True)
class UpDownGaugeStats:
    neurons: int
    zero_norm_neurons: int
    minimum_up_norm: float
    maximum_up_norm: float
    mean_up_norm: float
    exact_function_relative_l2_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def normalize_swiglu_up_down_gauge(
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, UpDownGaugeStats]:
    """Apply the exact SwiGLU gauge ``u_j -> u_j/s, d_j -> s*d_j``.

    For every neuron ``j`` and positive scalar ``s``:

    ``d_j * SiLU(g_j x) * (u_j x)``

    is unchanged by replacing ``u_j`` with ``u_j / s`` and ``d_j`` with
    ``s * d_j``. Choosing ``s = ||u_j||`` removes irrelevant up-vector scale
    before prototype clustering while preserving the checkpoint function
    exactly. Gate rows are not normalized because SiLU is not homogeneous.
    """

    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("SwiGLU weights must be two-dimensional")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have identical shapes")
    neurons, hidden = gate_weight.shape
    if down_weight.shape != (hidden, neurons):
        raise ValueError("down weight must have shape [hidden, neurons]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    gate = gate_weight.detach().to("cpu", torch.float32).contiguous()
    up = up_weight.detach().to("cpu", torch.float32).contiguous()
    down = down_weight.detach().to("cpu", torch.float32).contiguous()
    norms = torch.linalg.vector_norm(up, dim=1)
    zero = norms <= epsilon
    safe = torch.where(zero, torch.ones_like(norms), norms)
    normalized_up = up / safe.unsqueeze(1)
    scaled_down = down * safe.unsqueeze(0)

    generator = torch.Generator(device="cpu").manual_seed(5171)
    probe = torch.randn(8, hidden, generator=generator)
    original = (
        torch.nn.functional.silu(probe @ gate.T)
        * (probe @ up.T)
    ) @ down.T
    transformed = (
        torch.nn.functional.silu(probe @ gate.T)
        * (probe @ normalized_up.T)
    ) @ scaled_down.T
    numerator = torch.linalg.vector_norm(original - transformed)
    denominator = torch.linalg.vector_norm(original).clamp_min(1e-12)

    stats = UpDownGaugeStats(
        neurons=neurons,
        zero_norm_neurons=int(zero.sum().item()),
        minimum_up_norm=float(norms.min().item()),
        maximum_up_norm=float(norms.max().item()),
        mean_up_norm=float(norms.mean().item()),
        exact_function_relative_l2_error=float((numerator / denominator).item()),
    )
    return gate, normalized_up, scaled_down, stats


def _linear_from_weight(weight: torch.Tensor) -> nn.Linear:
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def compile_gauge_normalized_swiglu_dictionary(
    *,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    prototypes: int,
    projection_dim: int = 64,
    iterations: int = 5,
    factor_bits: int = 8,
    seed: int = 0,
) -> tuple[
    CompiledSwiGLUDictionary,
    MLPDictionaryFitStats,
    UpDownGaugeStats,
]:
    """Compile a neuron dictionary after exact up/down gauge normalization."""

    gate, normalized_up, scaled_down, gauge_stats = (
        normalize_swiglu_up_down_gauge(
            gate_weight=gate_proj.weight,
            up_weight=up_proj.weight,
            down_weight=down_proj.weight,
        )
    )
    transformed_gate = _linear_from_weight(gate)
    transformed_up = _linear_from_weight(normalized_up)
    transformed_down = _linear_from_weight(scaled_down)
    compiled, fit_stats = compile_swiglu_dictionary(
        gate_proj=transformed_gate,
        up_proj=transformed_up,
        down_proj=transformed_down,
        prototypes=prototypes,
        projection_dim=projection_dim,
        iterations=iterations,
        factor_bits=factor_bits,
        seed=seed,
    )
    return compiled, fit_stats, gauge_stats
