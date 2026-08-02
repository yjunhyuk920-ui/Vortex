from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from vortex_runtime.feasibility import GIB, default_specs
from vortex_runtime.rank_frontier import RankBudgetPoint


@dataclass(frozen=True)
class RefreshCost:
    scope: str
    weight_bytes_per_anchor: float
    flops_per_anchor: float
    refresh_interval: int
    hot_traffic_gib_per_token: float
    projected_traffic_gib_per_token: float
    traffic_limit_gib_per_token: float
    hot_compute_gflop_per_token: float
    projected_compute_gflop_per_token: float
    compute_limit_gflop_per_token: float
    minimum_interval_from_traffic: float
    minimum_interval_from_compute: float

    @property
    def traffic_pass(self) -> bool:
        return self.projected_traffic_gib_per_token <= self.traffic_limit_gib_per_token

    @property
    def compute_pass(self) -> bool:
        return self.projected_compute_gflop_per_token <= self.compute_limit_gflop_per_token

    @property
    def pass_all(self) -> bool:
        return self.traffic_pass and self.compute_pass

    @property
    def minimum_integer_interval(self) -> int:
        return math.ceil(
            max(
                self.minimum_interval_from_traffic,
                self.minimum_interval_from_compute,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "traffic_pass": self.traffic_pass,
            "compute_pass": self.compute_pass,
            "pass_all": self.pass_all,
            "minimum_integer_interval": self.minimum_integer_interval,
        }


def _refresh_cost(
    *,
    scope: str,
    weight_bytes_per_anchor: float,
    flops_per_anchor: float,
    refresh_interval: int,
    hot_budget: RankBudgetPoint,
) -> RefreshCost:
    if refresh_interval <= 0:
        raise ValueError("refresh_interval must be positive")
    if weight_bytes_per_anchor < 0 or flops_per_anchor < 0:
        raise ValueError("refresh costs must be non-negative")

    traffic_headroom_bytes = (
        hot_budget.traffic_limit_gib_per_token
        - hot_budget.hot_traffic_gib_per_token
    ) * GIB
    compute_headroom_flops = (
        hot_budget.compute_limit_gflop_per_token
        - hot_budget.hot_compute_gflop_per_token
    ) * 1e9
    minimum_traffic = (
        weight_bytes_per_anchor / traffic_headroom_bytes
        if traffic_headroom_bytes > 0
        else math.inf
    )
    minimum_compute = (
        flops_per_anchor / compute_headroom_flops
        if compute_headroom_flops > 0
        else math.inf
    )
    projected_traffic = (
        hot_budget.hot_traffic_gib_per_token
        + weight_bytes_per_anchor / refresh_interval / GIB
    )
    projected_compute = (
        hot_budget.hot_compute_gflop_per_token
        + flops_per_anchor / refresh_interval / 1e9
    )
    return RefreshCost(
        scope=scope,
        weight_bytes_per_anchor=float(weight_bytes_per_anchor),
        flops_per_anchor=float(flops_per_anchor),
        refresh_interval=int(refresh_interval),
        hot_traffic_gib_per_token=hot_budget.hot_traffic_gib_per_token,
        projected_traffic_gib_per_token=float(projected_traffic),
        traffic_limit_gib_per_token=hot_budget.traffic_limit_gib_per_token,
        hot_compute_gflop_per_token=hot_budget.hot_compute_gflop_per_token,
        projected_compute_gflop_per_token=float(projected_compute),
        compute_limit_gflop_per_token=hot_budget.compute_limit_gflop_per_token,
        minimum_interval_from_traffic=float(minimum_traffic),
        minimum_interval_from_compute=float(minimum_compute),
    )


def managed_o_down_refresh_cost(
    *,
    refresh_interval: int,
    hot_budget: RankBudgetPoint,
) -> RefreshCost:
    """Lower-bound one exact O/down anchor on the 405B target.

    One causal anchor observes exact O and down projection outputs for one token
    in every layer and may append at most one new response-basis direction per
    managed module.  This charges the original 4-bit O/down matrices and their
    dense arithmetic once per anchor.  Basis maintenance and capsule writes are
    deliberately omitted, so this is an optimistic lower bound.
    """

    target, _baseline = default_specs()
    elements = target.layers * (
        target.hidden_size * target.hidden_size
        + target.hidden_size * target.intermediate_size
    )
    return _refresh_cost(
        scope="o_down_exact_anchor_lower_bound",
        weight_bytes_per_anchor=elements * target.weight_bits / 8,
        flops_per_anchor=2.0 * elements,
        refresh_interval=refresh_interval,
        hot_budget=hot_budget,
    )


def full_model_refresh_cost(
    *,
    refresh_interval: int,
    hot_budget: RankBudgetPoint,
) -> RefreshCost:
    """Charge one exact full-model decode anchor every refresh interval."""

    target, _baseline = default_specs()
    return _refresh_cost(
        scope="full_model_exact_anchor",
        weight_bytes_per_anchor=target.weight_bytes + target.kv_bytes,
        flops_per_anchor=(
            target.dense_linear_flops_per_token
            + target.dense_attention_flops_per_token
        ),
        refresh_interval=refresh_interval,
        hot_budget=hot_budget,
    )
