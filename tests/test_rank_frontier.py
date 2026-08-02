from vortex_runtime.rank_frontier import (
    maximum_feasible_rank,
    rank_budget_point,
)


def test_rank_72_is_inside_fixed_405b_8bit_envelope() -> None:
    point = rank_budget_point(72, capsule_bits=8)
    assert point.memory_pass is True
    assert point.traffic_pass is True
    assert point.compute_pass is True
    assert point.pass_all is True


def test_rank_80_exceeds_8bit_hot_traffic_envelope() -> None:
    point = rank_budget_point(80, capsule_bits=8)
    assert point.memory_pass is True
    assert point.compute_pass is True
    assert point.traffic_pass is False
    assert point.pass_all is False


def test_maximum_aligned_feasible_rank_by_precision() -> None:
    assert maximum_feasible_rank(
        capsule_bits=8,
        step=8,
        maximum_rank=256,
    ) == 72
    assert maximum_feasible_rank(
        capsule_bits=6,
        step=8,
        maximum_rank=256,
    ) == 96
    assert maximum_feasible_rank(
        capsule_bits=4,
        step=8,
        maximum_rank=256,
    ) == 136


def test_rank_144_exceeds_compute_even_at_4bit() -> None:
    point = rank_budget_point(144, capsule_bits=4)
    assert point.traffic_pass is True
    assert point.compute_pass is False
    assert point.pass_all is False
