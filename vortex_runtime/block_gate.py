from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BlockSharedGate:
    """Combined traffic and compute accounting for a shared repair tile set.

    Weight bytes may be streamed once and amortized over a committed block.
    Exact tile arithmetic is still executed for every token, so repair compute
    scales with the selected parameter fraction and is not divided by the block
    length.
    """

    committed_tokens: int
    selected_weight_bytes: int
    full_model_weight_bytes: int
    minimum_traffic_efficiency: float
    hot_gflop_per_token: float
    full_exact_repair_gflop_per_token: float
    compute_limit_gflop_per_token: float

    def __post_init__(self) -> None:
        if self.committed_tokens < 0:
            raise ValueError("committed_tokens must be non-negative")
        if self.selected_weight_bytes < 0:
            raise ValueError("selected_weight_bytes must be non-negative")
        if self.full_model_weight_bytes <= 0:
            raise ValueError("full_model_weight_bytes must be positive")
        if self.minimum_traffic_efficiency <= 0:
            raise ValueError("minimum_traffic_efficiency must be positive")
        if min(
            self.hot_gflop_per_token,
            self.full_exact_repair_gflop_per_token,
            self.compute_limit_gflop_per_token,
        ) < 0:
            raise ValueError("compute values must be non-negative")

    @property
    def repair_fraction(self) -> float:
        return self.selected_weight_bytes / self.full_model_weight_bytes

    @property
    def traffic_efficiency(self) -> float | None:
        if self.selected_weight_bytes == 0:
            return None
        return self.committed_tokens / self.repair_fraction

    @property
    def exact_repair_gflop_per_token(self) -> float:
        return self.repair_fraction * self.full_exact_repair_gflop_per_token

    @property
    def projected_total_gflop_per_token(self) -> float:
        return self.hot_gflop_per_token + self.exact_repair_gflop_per_token

    @property
    def traffic_pass(self) -> bool:
        if self.selected_weight_bytes == 0:
            return True
        assert self.traffic_efficiency is not None
        return self.traffic_efficiency >= self.minimum_traffic_efficiency

    @property
    def compute_pass(self) -> bool:
        return (
            self.projected_total_gflop_per_token
            <= self.compute_limit_gflop_per_token
        )

    @property
    def pass_all(self) -> bool:
        return (
            self.committed_tokens > 0
            and self.traffic_pass
            and self.compute_pass
        )

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            **asdict(self),
            "repair_fraction": self.repair_fraction,
            "traffic_efficiency": self.traffic_efficiency,
            "exact_repair_gflop_per_token": self.exact_repair_gflop_per_token,
            "projected_total_gflop_per_token": (
                self.projected_total_gflop_per_token
            ),
            "traffic_pass": self.traffic_pass,
            "compute_pass": self.compute_pass,
            "nonempty_commit_pass": self.committed_tokens > 0,
            "pass_all": self.pass_all,
        }


def maximum_selected_bytes_for_compute(
    *,
    full_model_weight_bytes: int,
    hot_gflop_per_token: float,
    full_exact_repair_gflop_per_token: float,
    compute_limit_gflop_per_token: float,
) -> float:
    if full_model_weight_bytes <= 0:
        raise ValueError("full_model_weight_bytes must be positive")
    if full_exact_repair_gflop_per_token <= 0:
        raise ValueError("full exact repair compute must be positive")
    remaining = compute_limit_gflop_per_token - hot_gflop_per_token
    if remaining <= 0:
        return 0.0
    fraction = min(1.0, remaining / full_exact_repair_gflop_per_token)
    return fraction * full_model_weight_bytes


def maximum_selected_bytes_for_combined_gate(
    *,
    committed_tokens: int,
    full_model_weight_bytes: int,
    minimum_traffic_efficiency: float,
    hot_gflop_per_token: float,
    full_exact_repair_gflop_per_token: float,
    compute_limit_gflop_per_token: float,
) -> float:
    if committed_tokens < 0:
        raise ValueError("committed_tokens must be non-negative")
    traffic_limit = (
        committed_tokens
        * full_model_weight_bytes
        / minimum_traffic_efficiency
    )
    compute_limit = maximum_selected_bytes_for_compute(
        full_model_weight_bytes=full_model_weight_bytes,
        hot_gflop_per_token=hot_gflop_per_token,
        full_exact_repair_gflop_per_token=full_exact_repair_gflop_per_token,
        compute_limit_gflop_per_token=compute_limit_gflop_per_token,
    )
    return min(traffic_limit, compute_limit)
