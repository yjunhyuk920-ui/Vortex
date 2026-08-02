from vortex_runtime.rank_frontier import (
    maximum_feasible_rank,
    rank_budget_point,
)


def test_rank_72_is_inside_fixed_405b_envelope() -> None:
    point = rank_budget_point(72)
    assert point.memory_pass is True
    assert point.traffic_pass is True
    assert point.compute_pass is True
    assert point.pass_all is True


def test_rank_80_exceeds_hot_traffic_envelope() -> None:
    point = rank_budget_point(80)
    assert point.memory_pass is True
    assert point.compute_pass is True
    assert point.traffic_pass is False
    assert point.pass_all is False


def test_maximum_aligned_feasible_rank_is_72() -> None:
    assert maximum_feasible_rank(step=8, maximum_rank=256) == 72
