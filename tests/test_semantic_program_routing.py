from __future__ import annotations

import math

import torch

from vortex_runtime.semantic_program_routing import (
    assign_semantic_states,
    block_perpendicular_ratio,
    build_block_bases,
    deterministic_projection,
    project_and_normalize_signatures,
    routing_run_statistics,
    semantic_program_budget,
    spherical_state_centroids,
    summarize_ratios,
)


def test_rank_two_405b_program_budget_matches_certificate() -> None:
    budget = semantic_program_budget(block_size=1024, rank=2)
    assert budget.blocks_per_hidden_vector == 16
    assert math.isclose(budget.active_program_gib, 0.682525634765625)
    assert math.isclose(budget.minimum_mean_run_length, 1.7063140869140625)
    assert budget.active_program_pass


def test_program_size_increases_with_rank() -> None:
    rank_two = semantic_program_budget(block_size=1024, rank=2)
    rank_four = semantic_program_budget(block_size=1024, rank=4)
    rank_eight = semantic_program_budget(block_size=1024, rank=8)
    assert rank_two.active_program_gib < rank_four.active_program_gib
    assert rank_four.active_program_gib < rank_eight.active_program_gib
    assert rank_eight.active_program_gib < 2.0


def test_projection_and_spherical_router_separate_two_groups() -> None:
    hidden = [
        torch.tensor([1.0, 0.1, 0.0, 0.0]),
        torch.tensor([0.9, -0.1, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.1, 0.0]),
        torch.tensor([-0.9, 0.0, -0.1, 0.0]),
    ]
    projection = torch.eye(4)
    signatures = project_and_normalize_signatures(hidden, projection)
    centroids, labels = spherical_state_centroids(signatures, states=2)
    reassigned, similarity = assign_semantic_states(signatures, centroids)
    assert torch.equal(labels, reassigned)
    assert labels[0].item() == labels[1].item()
    assert labels[2].item() == labels[3].item()
    assert labels[0].item() != labels[2].item()
    assert bool((similarity > 0.9).all().item())


def test_deterministic_projection_is_orthonormal() -> None:
    projection = deterministic_projection(16, 5, seed=7)
    gram = projection.T @ projection
    assert torch.allclose(gram, torch.eye(5), atol=1e-5, rtol=0.0)


def test_basis_aligned_vectors_have_negligible_perpendicular_ratio() -> None:
    vectors = [
        torch.tensor([1.0, 0.0, 2.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 2.0]),
    ]
    bases = build_block_bases(vectors, block_size=2, rank=2)
    ratio = block_perpendicular_ratio(
        torch.tensor([2.0, -3.0, 4.0, -6.0]),
        bases,
        block_size=2,
    )
    assert ratio < 1e-5


def test_higher_rank_cannot_worsen_training_vector_coverage() -> None:
    vectors = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    ]
    rank_one = build_block_bases(vectors, block_size=4, rank=1)
    rank_two = build_block_bases(vectors, block_size=4, rank=2)
    for vector in vectors:
        low = block_perpendicular_ratio(vector, rank_one, block_size=4)
        high = block_perpendicular_ratio(vector, rank_two, block_size=4)
        assert high <= low + 1e-6


def test_routing_run_statistics_charge_initial_loads_and_switches() -> None:
    stats = routing_run_statistics(
        [[0, 0, 1, 1, 1], [2, 2, 2]],
        active_program_gib=0.5,
    )
    assert stats.tokens == 8
    assert stats.program_loads == 3
    assert stats.transition_switches == 1
    assert math.isclose(stats.projected_switch_traffic_gib_per_token, 0.1875)
    assert math.isclose(stats.mean_run_length, 8 / 3)
    assert stats.maximum_run_length == 3


def test_ratio_summary_reports_upper_tail() -> None:
    summary = summarize_ratios([0.0, 0.1, 0.2, 0.9])
    assert summary.count == 4
    assert math.isclose(summary.mean, 0.3)
    assert summary.p95 > 0.7
    assert summary.maximum == 0.9
