from __future__ import annotations

from vortex_runtime.feasibility import default_specs
from vortex_runtime.recurrent_lowrank_budget import (
    maximum_feasible_residual_rank,
    recurrent_low_rank_budget,
)


def test_positive_lowrank_residual_headroom_exists() -> None:
    target, baseline = default_specs()
    maximum_rank = maximum_feasible_residual_rank(
        target=target,
        baseline=baseline,
        unique_layers=3,
        dictionary_bits=4,
        residual_bits=8,
        metadata_bits=16,
        workspace_gib=1.0,
        memory_limit_gib=8.0,
        effective_tops=160.0,
        maximum_rank=64,
    )
    assert maximum_rank > 0
    feasible = recurrent_low_rank_budget(
        target=target,
        baseline=baseline,
        unique_layers=3,
        rank=maximum_rank,
        residual_bits=8,
    )
    assert feasible.pass_all
    assert feasible.total_gib <= 8.0


def test_next_rank_breaks_at_least_one_gate() -> None:
    target, baseline = default_specs()
    maximum_rank = maximum_feasible_residual_rank(
        target=target,
        baseline=baseline,
        unique_layers=3,
        residual_bits=8,
        maximum_rank=64,
    )
    next_point = recurrent_low_rank_budget(
        target=target,
        baseline=baseline,
        unique_layers=3,
        rank=maximum_rank + 1,
        residual_bits=8,
    )
    assert not next_point.pass_all


def test_rank_one_residual_compute_is_small() -> None:
    target, baseline = default_specs()
    point = recurrent_low_rank_budget(
        target=target,
        baseline=baseline,
        unique_layers=3,
        rank=1,
        residual_bits=8,
    )
    assert point.extra_residual_flops_per_token < 0.1e9
    assert point.compute_pass
