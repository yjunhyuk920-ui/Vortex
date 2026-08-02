from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.mlp_heavy_hitter import OracleHeavyHitterSwiGLU
from vortex_runtime.nonlinear_heavy_hitter import (
    LayerDamagePoint,
    normalize_damage_curves,
    replace_llama_mlp_with_count_allocation,
    solve_nonlinear_allocation,
    uniform_neuron_allocation,
)


def _point(count: int, damage: float) -> LayerDamagePoint:
    return LayerDamagePoint(
        selected_neurons=count,
        damage=damage,
        top1_rate=0.0,
        top32_rate=0.0,
        output_error=damage,
    )


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = F.silu


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinyMLP()


class _TinyRoot(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer() for _ in range(layers)])


class _TinyLlama(nn.Module):
    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.model = _TinyRoot(layers)


def test_dynamic_program_prefers_sensitive_layer() -> None:
    curves = [
        [_point(1, 10.0), _point(2, 2.0), _point(4, 0.5)],
        [_point(1, 3.0), _point(2, 2.5), _point(4, 2.0)],
    ]
    allocation = solve_nonlinear_allocation(curves, total_budget=5)
    assert allocation.layer_counts == (4, 1)
    assert allocation.used_neurons == 5
    assert abs(allocation.predicted_total_damage - 3.5) < 1e-9


def test_allocator_can_leave_budget_unused_when_damage_is_equal() -> None:
    curves = [
        [_point(1, 1.0), _point(4, 1.0)],
        [_point(1, 2.0), _point(4, 2.0)],
    ]
    allocation = solve_nonlinear_allocation(curves, total_budget=8)
    assert allocation.used_neurons == 8
    assert allocation.layer_counts == (4, 4)


def test_normalization_builds_feasible_monotone_lower_envelope() -> None:
    curves = [[_point(1, 3.0), _point(2, 4.0), _point(4, 1.0)]]
    normalized = normalize_damage_curves(curves)[0]
    assert [point.damage for point in normalized] == [3.0, 3.0, 1.0]


def test_uniform_allocation_preserves_exact_total() -> None:
    counts = uniform_neuron_allocation(
        layers=4,
        intermediate_neurons=10,
        total_neurons=11,
    )
    assert counts == (3, 3, 3, 2)
    assert sum(counts) == 11


def test_branch_local_replacement_uses_requested_original_neuron_counts() -> None:
    model = _TinyLlama(layers=3)
    replacements = replace_llama_mlp_with_count_allocation(
        model,
        layer_counts=(1, 3, 8),
    )
    assert len(replacements) == 3
    assert all(isinstance(module, OracleHeavyHitterSwiGLU) for module in replacements)
    assert [module.selected_neurons for module in replacements] == [1, 3, 8]
    assert all(layer.mlp is replacement for layer, replacement in zip(model.model.layers, replacements))


def test_replacement_rejects_layer_count_mismatch() -> None:
    model = _TinyLlama(layers=2)
    try:
        replace_llama_mlp_with_count_allocation(model, layer_counts=(1,))
    except ValueError as error:
        assert "one neuron count" in str(error)
    else:
        raise AssertionError("expected a layer-count mismatch to fail")


def test_infeasible_budget_is_rejected() -> None:
    curves = [
        [_point(2, 1.0), _point(4, 0.5)],
        [_point(2, 1.0), _point(4, 0.5)],
    ]
    try:
        solve_nonlinear_allocation(curves, total_budget=3)
    except ValueError as error:
        assert "minimum" in str(error)
    else:
        raise AssertionError("expected infeasible allocation to fail")
