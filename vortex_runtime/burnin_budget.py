from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, inf
from typing import Iterable


@dataclass(frozen=True)
class BurnInAmortization:
    exact_burnin_tokens: int
    horizon_tokens: int
    hot_traffic_gib_per_token: float
    cold_exact_traffic_gib_per_token: float
    traffic_limit_gib_per_token: float
    projected_traffic_gib_per_token: float
    hot_compute_gflop_per_token: float
    cold_exact_compute_gflop_per_token: float
    compute_limit_gflop_per_token: float
    projected_compute_gflop_per_token: float

    @property
    def traffic_pass(self) -> bool:
        return self.projected_traffic_gib_per_token <= self.traffic_limit_gib_per_token

    @property
    def compute_pass(self) -> bool:
        return self.projected_compute_gflop_per_token <= self.compute_limit_gflop_per_token

    @property
    def pass_all(self) -> bool:
        return self.traffic_pass and self.compute_pass

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            **asdict(self),
            "traffic_pass": self.traffic_pass,
            "compute_pass": self.compute_pass,
            "pass_all": self.pass_all,
        }


def burnin_amortization(
    *,
    exact_burnin_tokens: int,
    horizon_tokens: int,
    hot_traffic_gib_per_token: float,
    cold_exact_traffic_gib_per_token: float,
    traffic_limit_gib_per_token: float,
    hot_compute_gflop_per_token: float,
    cold_exact_compute_gflop_per_token: float,
    compute_limit_gflop_per_token: float,
) -> BurnInAmortization:
    if exact_burnin_tokens < 0:
        raise ValueError("exact burn-in token count must be non-negative")
    if horizon_tokens <= 0:
        raise ValueError("amortization horizon must be positive")
    if min(
        hot_traffic_gib_per_token,
        cold_exact_traffic_gib_per_token,
        traffic_limit_gib_per_token,
        hot_compute_gflop_per_token,
        cold_exact_compute_gflop_per_token,
        compute_limit_gflop_per_token,
    ) < 0:
        raise ValueError("budget values must be non-negative")

    traffic = (
        hot_traffic_gib_per_token
        + exact_burnin_tokens
        * cold_exact_traffic_gib_per_token
        / horizon_tokens
    )
    compute = (
        hot_compute_gflop_per_token
        + exact_burnin_tokens
        * cold_exact_compute_gflop_per_token
        / horizon_tokens
    )
    return BurnInAmortization(
        exact_burnin_tokens=exact_burnin_tokens,
        horizon_tokens=horizon_tokens,
        hot_traffic_gib_per_token=hot_traffic_gib_per_token,
        cold_exact_traffic_gib_per_token=cold_exact_traffic_gib_per_token,
        traffic_limit_gib_per_token=traffic_limit_gib_per_token,
        projected_traffic_gib_per_token=traffic,
        hot_compute_gflop_per_token=hot_compute_gflop_per_token,
        cold_exact_compute_gflop_per_token=cold_exact_compute_gflop_per_token,
        compute_limit_gflop_per_token=compute_limit_gflop_per_token,
        projected_compute_gflop_per_token=compute,
    )


def minimum_amortization_horizon(
    *,
    exact_burnin_tokens: int,
    hot_cost_per_token: float,
    cold_exact_cost_per_token: float,
    cost_limit_per_token: float,
) -> int | None:
    if exact_burnin_tokens < 0:
        raise ValueError("exact burn-in token count must be non-negative")
    if min(hot_cost_per_token, cold_exact_cost_per_token, cost_limit_per_token) < 0:
        raise ValueError("cost values must be non-negative")
    if exact_burnin_tokens == 0:
        return 1 if hot_cost_per_token <= cost_limit_per_token else None
    headroom = cost_limit_per_token - hot_cost_per_token
    if headroom <= 0:
        return None
    return max(
        1,
        ceil(exact_burnin_tokens * cold_exact_cost_per_token / headroom),
    )


def amortization_frontier(
    *,
    exact_burnin_tokens: int,
    horizons: Iterable[int],
    hot_traffic_gib_per_token: float,
    cold_exact_traffic_gib_per_token: float,
    traffic_limit_gib_per_token: float,
    hot_compute_gflop_per_token: float,
    cold_exact_compute_gflop_per_token: float,
    compute_limit_gflop_per_token: float,
) -> list[BurnInAmortization]:
    values = list(horizons)
    if not values:
        raise ValueError("at least one amortization horizon is required")
    return [
        burnin_amortization(
            exact_burnin_tokens=exact_burnin_tokens,
            horizon_tokens=horizon,
            hot_traffic_gib_per_token=hot_traffic_gib_per_token,
            cold_exact_traffic_gib_per_token=cold_exact_traffic_gib_per_token,
            traffic_limit_gib_per_token=traffic_limit_gib_per_token,
            hot_compute_gflop_per_token=hot_compute_gflop_per_token,
            cold_exact_compute_gflop_per_token=cold_exact_compute_gflop_per_token,
            compute_limit_gflop_per_token=compute_limit_gflop_per_token,
        )
        for horizon in values
    ]
