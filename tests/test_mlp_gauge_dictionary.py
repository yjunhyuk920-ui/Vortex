from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.mlp_gauge_dictionary import (
    compile_gauge_normalized_swiglu_dictionary,
    normalize_swiglu_up_down_gauge,
)


def _linear(weight: torch.Tensor) -> nn.Linear:
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def _swiglu(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def test_up_down_gauge_is_exact() -> None:
    generator = torch.Generator().manual_seed(5213)
    gate = torch.randn(17, 9, generator=generator)
    up = torch.randn(17, 9, generator=generator)
    scales = torch.linspace(0.1, 4.0, 17)
    up = up * scales[:, None]
    down = torch.randn(9, 17, generator=generator)
    normalized_gate, normalized_up, scaled_down, stats = (
        normalize_swiglu_up_down_gauge(
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
        )
    )
    x = torch.randn(5, 3, 9, generator=generator)
    expected = _swiglu(x, gate, up, down)
    actual = _swiglu(x, normalized_gate, normalized_up, scaled_down)
    assert stats.exact_function_relative_l2_error < 1e-6
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    norms = torch.linalg.vector_norm(normalized_up, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_gauge_normalization_exposes_repeated_directions() -> None:
    generator = torch.Generator().manual_seed(5231)
    hidden = 8
    prototypes = 4
    repeats = 3
    gate_base = torch.randn(prototypes, hidden, generator=generator)
    up_base = torch.randn(prototypes, hidden, generator=generator)
    assignments = torch.arange(prototypes).repeat_interleave(repeats)
    scales = torch.linspace(0.4, 2.4, prototypes * repeats)
    gate = gate_base[assignments]
    up = up_base[assignments] * scales[:, None]
    down = torch.randn(hidden, prototypes * repeats, generator=generator)

    compiled, fit_stats, gauge_stats = compile_gauge_normalized_swiglu_dictionary(
        gate_proj=_linear(gate),
        up_proj=_linear(up),
        down_proj=_linear(down),
        prototypes=prototypes,
        projection_dim=16,
        iterations=6,
        factor_bits=16,
        seed=5233,
    )
    x = torch.randn(6, hidden, generator=generator)
    expected = _swiglu(x, gate, up, down)
    actual = compiled(x)
    assert gauge_stats.exact_function_relative_l2_error < 1e-6
    assert fit_stats.gate_up_relative_l2_error < 1e-5
    assert torch.allclose(actual, expected, atol=5e-4, rtol=5e-4)
