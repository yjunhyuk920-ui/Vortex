from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.mlp_functional_skeleton import (
    compile_swiglu_functional_skeleton,
    deterministic_rademacher_probes,
    select_response_skeleton,
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


def test_rademacher_probes_have_unit_rms() -> None:
    probes = deterministic_rademacher_probes(
        count=17,
        hidden_size=23,
        seed=6101,
    )
    rms = torch.sqrt(probes.square().mean(dim=1))
    assert torch.allclose(rms, torch.ones_like(rms))


def test_response_skeleton_selects_independent_columns() -> None:
    generator = torch.Generator().manual_seed(6103)
    basis = torch.randn(32, 4, generator=generator)
    coefficients = torch.randn(4, 19, generator=generator)
    responses = basis @ coefficients
    indices, orthogonal = select_response_skeleton(
        responses,
        prototypes=4,
    )
    selected = responses[:, indices]
    reconstructed = selected @ torch.linalg.lstsq(selected, responses).solution
    relative = torch.linalg.vector_norm(responses - reconstructed) / torch.linalg.vector_norm(responses)
    assert indices.numel() == 4
    assert orthogonal.shape == (32, 4)
    assert float(relative.item()) < 1e-5


def test_functional_skeleton_recovers_repeated_scaled_neurons() -> None:
    generator = torch.Generator().manual_seed(6113)
    hidden = 10
    prototypes = 5
    repeats = 4
    gate_base = torch.randn(prototypes, hidden, generator=generator)
    up_base = torch.randn(prototypes, hidden, generator=generator)
    assignment = torch.arange(prototypes).repeat_interleave(repeats)
    scales = torch.linspace(0.25, 2.5, prototypes * repeats)
    gate = gate_base[assignment]
    up = up_base[assignment] * scales[:, None]
    down = torch.randn(hidden, prototypes * repeats, generator=generator)

    compiled, stats, gauge = compile_swiglu_functional_skeleton(
        gate_proj=_linear(gate),
        up_proj=_linear(up),
        down_proj=_linear(down),
        prototypes=prototypes,
        probe_count=96,
        heldout_probe_count=64,
        factor_bits=16,
        ridge=1e-7,
        seed=6119,
    )
    x = torch.randn(7, 3, hidden, generator=generator)
    expected = _swiglu(x, gate, up, down)
    actual = compiled(x)
    assert gauge.exact_function_relative_l2_error < 1e-6
    assert stats.prototypes_selected == prototypes
    assert stats.heldout_output_relative_l2_error < 1e-4
    assert torch.allclose(actual, expected, atol=8e-4, rtol=8e-4)
