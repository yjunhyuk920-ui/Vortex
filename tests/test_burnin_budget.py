from vortex_runtime.burnin_budget import (
    amortization_frontier,
    burnin_amortization,
    minimum_amortization_horizon,
)


def test_zero_burnin_preserves_hot_cost() -> None:
    result = burnin_amortization(
        exact_burnin_tokens=0,
        horizon_tokens=256,
        hot_traffic_gib_per_token=1.5,
        cold_exact_traffic_gib_per_token=100.0,
        traffic_limit_gib_per_token=2.0,
        hot_compute_gflop_per_token=4.0,
        cold_exact_compute_gflop_per_token=50.0,
        compute_limit_gflop_per_token=10.0,
    )
    assert result.projected_traffic_gib_per_token == 1.5
    assert result.projected_compute_gflop_per_token == 4.0
    assert result.pass_all is True


def test_longer_horizon_reduces_startup_cost() -> None:
    frontier = amortization_frontier(
        exact_burnin_tokens=4,
        horizons=(256, 4096),
        hot_traffic_gib_per_token=1.5,
        cold_exact_traffic_gib_per_token=100.0,
        traffic_limit_gib_per_token=2.0,
        hot_compute_gflop_per_token=4.0,
        cold_exact_compute_gflop_per_token=50.0,
        compute_limit_gflop_per_token=10.0,
    )
    assert frontier[1].projected_traffic_gib_per_token < (
        frontier[0].projected_traffic_gib_per_token
    )
    assert frontier[1].projected_compute_gflop_per_token < (
        frontier[0].projected_compute_gflop_per_token
    )


def test_minimum_horizon_rounds_up() -> None:
    assert minimum_amortization_horizon(
        exact_burnin_tokens=4,
        hot_cost_per_token=1.5,
        cold_exact_cost_per_token=100.0,
        cost_limit_per_token=2.0,
    ) == 800


def test_no_headroom_has_no_finite_horizon() -> None:
    assert minimum_amortization_horizon(
        exact_burnin_tokens=1,
        hot_cost_per_token=2.0,
        cold_exact_cost_per_token=100.0,
        cost_limit_per_token=2.0,
    ) is None
