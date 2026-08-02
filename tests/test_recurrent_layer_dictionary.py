from __future__ import annotations

from vortex_runtime.feasibility import default_specs
from vortex_runtime.recurrent_layer_dictionary import (
    cyclic_representative_assignment,
    nearest_representative_assignment,
    recurrent_draft_budget,
    recurrent_layer_schedule,
)


def test_nearest_assignment_preserves_depth_with_three_unique_layers() -> None:
    assignment = nearest_representative_assignment(
        total_layers=8,
        representative_indices=(0, 4, 7),
    )
    assert assignment == (0, 0, 0, 4, 4, 4, 7, 7)
    assert len(assignment) == 8
    assert set(assignment) == {0, 4, 7}


def test_cyclic_assignment_reuses_dictionary() -> None:
    assert cyclic_representative_assignment(
        total_layers=8,
        representative_indices=(0, 4, 7),
    ) == (0, 4, 7, 0, 4, 7, 0, 4)


def test_uniform_nearest_schedule_for_tinyllama_depth() -> None:
    schedule = recurrent_layer_schedule(
        total_layers=22,
        unique_layers=3,
        representative_strategy="uniform",
        assignment_strategy="nearest",
    )
    assert schedule.representative_indices == (0, 10, 21)
    assert schedule.total_positions == 22
    assert schedule.unique_layers == 3
    assert len(schedule.assignment) == 22
    assert schedule.assignment[0] == 0
    assert schedule.assignment[-1] == 21


def test_three_layer_recurrent_draft_closes_memory_and_compute_proxy() -> None:
    target, baseline = default_specs()
    budget = recurrent_draft_budget(
        target=target,
        baseline=baseline,
        unique_layers=3,
        weight_bits=4,
        tie_word_embeddings=False,
        workspace_gib=1.0,
        memory_limit_gib=8.0,
        effective_tops=160.0,
    )
    assert budget.memory_pass
    assert budget.compute_pass
    assert budget.pass_all
    assert budget.memory.total_gib <= 8.0
    assert budget.compute_seconds_per_token <= budget.allowed_seconds_per_token
