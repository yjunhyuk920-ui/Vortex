from vortex_runtime.rank_frontier import (
    maximum_feasible_rank,
    rank_budget_point,
)


def test_rank_128_is_inside_fixed_405b_envelope() -> None:
    point = rank_budget_point(128)
    assert point.memory_pass is True
    assert point.traffic_pass is True
    assert point.compute_pass is True
    assert point.pass_all is True


def test_rank_136_exceeds_8gib_memory_envelope() -> None:
    point = rank_budget_point(136)
    assert point.memory_pass is False
    assert point.pass_all is False


def test_maximum_aligned_feasible_rank_is_128() -> None:
    assert maximum_feasible_rank(step=8, maximum_rank=256) == 128
