from __future__ import annotations

import torch

from vortex_runtime.adjoint_heavy_hitter import (
    allocate_global_neuron_budget,
    uniform_neuron_allocation,
)


def test_global_allocator_prefers_high_utility_layers() -> None:
    scores = [
        torch.tensor([10.0, 9.0, 8.0, 0.1]),
        torch.tensor([2.0, 1.0, 0.5, 0.2]),
        torch.tensor([0.4, 0.3, 0.2, 0.1]),
    ]
    allocation = allocate_global_neuron_budget(
        scores,
        total_neurons=6,
        minimum_per_layer=1,
    )
    # Reserve the largest item in every layer: 10, 2, 0.4. The remaining
    # unit-cost global winners are 9, 8 and 1, yielding counts 3, 2 and 1.
    assert allocation.layer_counts == (3, 2, 1)
    assert allocation.total_neurons == 6
    assert allocation.selected_score_fraction > 0.8


def test_allocator_matches_global_unit_cost_knapsack() -> None:
    generator = torch.Generator().manual_seed(29001)
    scores = [torch.rand(7, generator=generator) for _ in range(4)]
    allocation = allocate_global_neuron_budget(
        scores,
        total_neurons=10,
        minimum_per_layer=0,
    )
    flattened = torch.cat(scores)
    expected = float(torch.topk(flattened, k=10).values.sum().item())
    total = float(flattened.sum().item())
    assert abs(allocation.selected_score_fraction - expected / total) < 1e-6
    assert sum(allocation.layer_counts) == 10


def test_uniform_allocation_preserves_exact_total() -> None:
    counts = uniform_neuron_allocation(
        layers=4,
        intermediate_neurons=10,
        total_neurons=11,
    )
    assert counts == (3, 3, 3, 2)
    assert sum(counts) == 11


def test_minimum_reservation_rejects_too_small_budget() -> None:
    scores = [torch.ones(5), torch.ones(5), torch.ones(5)]
    try:
        allocate_global_neuron_budget(
            scores,
            total_neurons=2,
            minimum_per_layer=1,
        )
    except ValueError as error:
        assert "feasible" in str(error)
    else:
        raise AssertionError("expected infeasible minimum reservation to fail")
