from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.feasibility import default_specs
from vortex_runtime.mlp_neuron_dictionary import (
    compile_swiglu_dictionary,
    mlp_neuron_dictionary_budget,
)


class ExactSwiGLU(nn.Module):
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        super().__init__()
        neurons, hidden = gate.shape
        self.gate_proj = nn.Linear(hidden, neurons, bias=False)
        self.up_proj = nn.Linear(hidden, neurons, bias=False)
        self.down_proj = nn.Linear(neurons, hidden, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate)
            self.up_proj.weight.copy_(up)
            self.down_proj.weight.copy_(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_target_128_prototype_mlp_budget_leaves_architecture_headroom() -> None:
    target, baseline = default_specs()
    budget = mlp_neuron_dictionary_budget(
        target=target,
        baseline=baseline,
        prototypes_per_layer=128,
        factor_bits=8,
    )
    assert budget.factor_gib < 0.75
    assert budget.remaining_traffic_gib_per_token > 2.0
    assert budget.memory_pass
    assert budget.partial_traffic_pass
    assert budget.partial_latency_pass


def test_repeated_neuron_functions_compile_exactly() -> None:
    generator = torch.Generator().manual_seed(211)
    hidden = 8
    prototypes = 4
    repeats = 3
    gate_base = torch.randn(prototypes, hidden, generator=generator)
    up_base = torch.randn(prototypes, hidden, generator=generator)
    assignments = torch.arange(prototypes).repeat_interleave(repeats)
    gate = gate_base[assignments]
    up = up_base[assignments]
    down = torch.randn(hidden, prototypes * repeats, generator=generator)
    exact = ExactSwiGLU(gate, up, down)

    compiled, stats = compile_swiglu_dictionary(
        gate_proj=exact.gate_proj,
        up_proj=exact.up_proj,
        down_proj=exact.down_proj,
        prototypes=prototypes,
        projection_dim=16,
        iterations=6,
        factor_bits=16,
        seed=223,
    )
    x = torch.randn(3, 5, hidden, generator=generator)
    expected = exact(x)
    actual = compiled(x)
    assert stats.minimum_cluster_size == repeats
    assert stats.maximum_cluster_size == repeats
    assert stats.gate_up_relative_l2_error < 1e-6
    assert torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)


def test_neuron_permutation_does_not_change_compiled_function() -> None:
    generator = torch.Generator().manual_seed(227)
    hidden = 6
    prototypes = 3
    repeats = 4
    gate_base = torch.randn(prototypes, hidden, generator=generator)
    up_base = torch.randn(prototypes, hidden, generator=generator)
    assignments = torch.arange(prototypes).repeat_interleave(repeats)
    gate = gate_base[assignments]
    up = up_base[assignments]
    down = torch.randn(hidden, prototypes * repeats, generator=generator)
    permutation = torch.randperm(prototypes * repeats, generator=generator)

    original = ExactSwiGLU(gate, up, down)
    permuted = ExactSwiGLU(
        gate[permutation],
        up[permutation],
        down[:, permutation],
    )
    compiled_original, _ = compile_swiglu_dictionary(
        gate_proj=original.gate_proj,
        up_proj=original.up_proj,
        down_proj=original.down_proj,
        prototypes=prototypes,
        projection_dim=12,
        iterations=6,
        factor_bits=16,
        seed=229,
    )
    compiled_permuted, _ = compile_swiglu_dictionary(
        gate_proj=permuted.gate_proj,
        up_proj=permuted.up_proj,
        down_proj=permuted.down_proj,
        prototypes=prototypes,
        projection_dim=12,
        iterations=6,
        factor_bits=16,
        seed=229,
    )
    x = torch.randn(4, hidden, generator=generator)
    assert torch.allclose(original(x), permuted(x), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        compiled_original(x),
        compiled_permuted(x),
        atol=5e-4,
        rtol=5e-4,
    )
