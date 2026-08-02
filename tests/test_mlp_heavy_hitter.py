from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.mlp_heavy_hitter import (
    OracleHeavyHitterSwiGLU,
    aggregate_heavy_hitter_stats,
    mlp_heavy_hitter_budget,
)


def _linear(weight: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def test_full_fraction_matches_exact_swiglu() -> None:
    generator = torch.Generator().manual_seed(27001)
    hidden = 7
    intermediate = 11
    gate = torch.randn(intermediate, hidden, generator=generator)
    up = torch.randn(intermediate, hidden, generator=generator)
    down = torch.randn(hidden, intermediate, generator=generator)
    module = OracleHeavyHitterSwiGLU(
        gate_proj=_linear(gate),
        up_proj=_linear(up),
        down_proj=_linear(down),
        act_fn=F.silu,
        selected_fraction=1.0,
    )
    x = torch.randn(2, 3, hidden, generator=generator)
    expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    stats = module.statistics()
    assert stats["selected_neurons"] == intermediate
    assert stats["mean_output_relative_l2_error"] < 1e-6


def test_oracle_selects_dominant_down_contribution() -> None:
    hidden = 3
    intermediate = 4
    gate = torch.zeros(intermediate, hidden)
    up = torch.zeros(intermediate, hidden)
    gate[:, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    up[:, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    down = torch.zeros(hidden, intermediate)
    down[:, 2] = torch.tensor([10.0, 0.0, 0.0])
    down[:, 0] = torch.tensor([0.1, 0.0, 0.0])
    down[:, 1] = torch.tensor([0.2, 0.0, 0.0])
    down[:, 3] = torch.tensor([0.3, 0.0, 0.0])
    module = OracleHeavyHitterSwiGLU(
        gate_proj=_linear(gate),
        up_proj=_linear(up),
        down_proj=_linear(down),
        act_fn=F.silu,
        selected_fraction=0.25,
    )
    output = module(torch.tensor([[1.0, 0.0, 0.0]]))
    assert output[0, 0] > 8.0
    assert bool(module.ever_selected[2].item())
    assert int(module.ever_selected.sum().item()) == 1


def test_aggregate_statistics_weight_vectors() -> None:
    generator = torch.Generator().manual_seed(27007)
    modules: list[OracleHeavyHitterSwiGLU] = []
    for _ in range(2):
        gate = torch.randn(8, 4, generator=generator)
        up = torch.randn(8, 4, generator=generator)
        down = torch.randn(4, 8, generator=generator)
        module = OracleHeavyHitterSwiGLU(
            gate_proj=_linear(gate),
            up_proj=_linear(up),
            down_proj=_linear(down),
            act_fn=F.silu,
            selected_fraction=0.25,
        )
        module(torch.randn(3, 4, generator=generator))
        modules.append(module)
    aggregate = aggregate_heavy_hitter_stats(modules)
    assert aggregate.modules == 2
    assert aggregate.activation_vectors == 6
    assert aggregate.selected_neurons_per_vector == 2
    assert 0 <= aggregate.mean_score_coverage <= 1


def test_405b_quarter_percent_exact_mlp_budget() -> None:
    target = ModelSpec(
        parameters=405_849_243_648,
        layers=126,
        hidden_size=16_384,
        intermediate_size=53_248,
        attention_heads=128,
        kv_heads=8,
        vocab_size=128_256,
        context_tokens=8_192,
        weight_bits=16,
        kv_bits=4,
    )
    budget = mlp_heavy_hitter_budget(
        target=target,
        selected_fraction=0.0025,
        source_bits=16,
        partial_traffic_limit_gib=1.6,
    )
    assert budget.selected_neurons_per_layer == 134
    assert budget.selected_weight_gib_per_token < 1.6
    assert budget.partial_traffic_pass
