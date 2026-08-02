from vortex_runtime.rank_frontier import rank_budget_point
from vortex_runtime.rolling_refresh_budget import (
    full_model_refresh_cost,
    managed_o_down_refresh_cost,
)


def test_refresh_cost_falls_with_longer_interval() -> None:
    hot = rank_budget_point(72, capsule_bits=8)
    short = managed_o_down_refresh_cost(refresh_interval=8, hot_budget=hot)
    long = managed_o_down_refresh_cost(refresh_interval=64, hot_budget=hot)

    assert short.projected_traffic_gib_per_token > long.projected_traffic_gib_per_token
    assert short.projected_compute_gflop_per_token > long.projected_compute_gflop_per_token
    assert short.minimum_integer_interval == long.minimum_integer_interval


def test_full_model_anchor_is_more_expensive_than_o_down_lower_bound() -> None:
    hot = rank_budget_point(72, capsule_bits=8)
    managed = managed_o_down_refresh_cost(refresh_interval=128, hot_budget=hot)
    full = full_model_refresh_cost(refresh_interval=128, hot_budget=hot)

    assert full.weight_bytes_per_anchor > managed.weight_bytes_per_anchor
    assert full.flops_per_anchor > managed.flops_per_anchor
    assert full.minimum_integer_interval > managed.minimum_integer_interval


def test_minimum_interval_boundary_matches_pass_state() -> None:
    hot = rank_budget_point(72, capsule_bits=8)
    probe = managed_o_down_refresh_cost(refresh_interval=1, hot_budget=hot)
    required = probe.minimum_integer_interval

    below = managed_o_down_refresh_cost(
        refresh_interval=max(1, required - 1),
        hot_budget=hot,
    )
    at = managed_o_down_refresh_cost(
        refresh_interval=required,
        hot_budget=hot,
    )

    if required > 1:
        assert not below.pass_all
    assert at.pass_all


def test_refresh_report_is_machine_readable() -> None:
    hot = rank_budget_point(96, capsule_bits=6)
    report = managed_o_down_refresh_cost(refresh_interval=64, hot_budget=hot).to_dict()

    assert report["scope"] == "o_down_exact_anchor_lower_bound"
    assert isinstance(report["minimum_integer_interval"], int)
    assert "projected_traffic_gib_per_token" in report
    assert "projected_compute_gflop_per_token" in report
