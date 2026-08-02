from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vortex_runtime.feasibility import (
    ObservedMechanism,
    WaveCandidate,
    architecture_gate0_report,
    default_specs,
)


@dataclass(frozen=True)
class RankBudgetPoint:
    rank: int
    capsule_bits: float
    memory_gib: float
    memory_limit_gib: float
    hot_traffic_gib_per_token: float
    traffic_limit_gib_per_token: float
    hot_compute_gflop_per_token: float
    compute_limit_gflop_per_token: float

    @property
    def memory_pass(self) -> bool:
        return self.memory_gib <= self.memory_limit_gib

    @property
    def traffic_pass(self) -> bool:
        return self.hot_traffic_gib_per_token <= self.traffic_limit_gib_per_token

    @property
    def compute_pass(self) -> bool:
        return self.hot_compute_gflop_per_token <= self.compute_limit_gflop_per_token

    @property
    def pass_all(self) -> bool:
        return self.memory_pass and self.traffic_pass and self.compute_pass

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "rank": self.rank,
            "capsule_bits": self.capsule_bits,
            "memory_gib": self.memory_gib,
            "memory_limit_gib": self.memory_limit_gib,
            "memory_pass": self.memory_pass,
            "hot_traffic_gib_per_token": self.hot_traffic_gib_per_token,
            "traffic_limit_gib_per_token": self.traffic_limit_gib_per_token,
            "traffic_pass": self.traffic_pass,
            "hot_compute_gflop_per_token": self.hot_compute_gflop_per_token,
            "compute_limit_gflop_per_token": self.compute_limit_gflop_per_token,
            "compute_pass": self.compute_pass,
            "pass_all": self.pass_all,
        }


@dataclass(frozen=True)
class MixedRankBudgetPoint:
    global_rank: int
    session_rank: int
    global_bits: int
    session_bits: int
    total_rank: int
    weighted_bit_rank: int
    effective_capsule_bits: float
    uniform_equivalent: RankBudgetPoint

    @property
    def pass_all(self) -> bool:
        return self.uniform_equivalent.pass_all

    def to_dict(self) -> dict[str, int | float | bool | dict[str, object]]:
        return {
            "global_rank": self.global_rank,
            "session_rank": self.session_rank,
            "global_bits": self.global_bits,
            "session_bits": self.session_bits,
            "total_rank": self.total_rank,
            "weighted_bit_rank": self.weighted_bit_rank,
            "effective_capsule_bits": self.effective_capsule_bits,
            "memory_gib": self.uniform_equivalent.memory_gib,
            "memory_limit_gib": self.uniform_equivalent.memory_limit_gib,
            "memory_pass": self.uniform_equivalent.memory_pass,
            "hot_traffic_gib_per_token": (
                self.uniform_equivalent.hot_traffic_gib_per_token
            ),
            "traffic_limit_gib_per_token": (
                self.uniform_equivalent.traffic_limit_gib_per_token
            ),
            "traffic_pass": self.uniform_equivalent.traffic_pass,
            "hot_compute_gflop_per_token": (
                self.uniform_equivalent.hot_compute_gflop_per_token
            ),
            "compute_limit_gflop_per_token": (
                self.uniform_equivalent.compute_limit_gflop_per_token
            ),
            "compute_pass": self.uniform_equivalent.compute_pass,
            "pass_all": self.pass_all,
        }


def rank_budget_point(
    rank: int,
    capsule_bits: float = 8,
) -> RankBudgetPoint:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if capsule_bits <= 0:
        raise ValueError("capsule_bits must be positive")
    target, baseline = default_specs()
    report = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(
            linear_rank=rank,
            capsule_bits=capsule_bits,
            repair_fraction=1e-12,
            committed_tokens_per_repair=1.0,
        ),
        observed=ObservedMechanism(
            committed_tokens_per_repair=1.0,
            repair_fraction=1e-12,
            source="precision-rank frontier hot-path budget",
        ),
    )
    return RankBudgetPoint(
        rank=rank,
        capsule_bits=float(capsule_bits),
        memory_gib=float(report["memory"]["total_gib"]),
        memory_limit_gib=float(report["memory"]["limit_gib"]),
        hot_traffic_gib_per_token=float(report["traffic"]["hot_gib_per_token"]),
        traffic_limit_gib_per_token=float(report["traffic"]["limit_gib_per_token"]),
        hot_compute_gflop_per_token=float(
            report["compute"]["hot_total_gflop_per_token"]
        ),
        compute_limit_gflop_per_token=float(
            report["compute"]["limit_gflop_per_token"]
        ),
    )


def mixed_rank_budget_point(
    *,
    global_rank: int,
    session_rank: int,
    global_bits: int,
    session_bits: int,
) -> MixedRankBudgetPoint:
    if global_rank < 0 or session_rank < 0:
        raise ValueError("mixed ranks must be non-negative")
    if global_rank + session_rank <= 0:
        raise ValueError("mixed total rank must be positive")
    if global_bits <= 0 or session_bits <= 0:
        raise ValueError("mixed precision must be positive")

    total_rank = global_rank + session_rank
    weighted_bit_rank = (
        global_rank * global_bits + session_rank * session_bits
    )
    effective_bits = weighted_bit_rank / total_rank
    equivalent = rank_budget_point(
        total_rank,
        capsule_bits=effective_bits,
    )
    return MixedRankBudgetPoint(
        global_rank=global_rank,
        session_rank=session_rank,
        global_bits=global_bits,
        session_bits=session_bits,
        total_rank=total_rank,
        weighted_bit_rank=weighted_bit_rank,
        effective_capsule_bits=effective_bits,
        uniform_equivalent=equivalent,
    )


def rank_frontier(
    ranks: Iterable[int],
    *,
    capsule_bits: float = 8,
) -> list[RankBudgetPoint]:
    points = [
        rank_budget_point(int(rank), capsule_bits=capsule_bits)
        for rank in ranks
    ]
    if not points:
        raise ValueError("at least one rank is required")
    return points


def maximum_feasible_rank(
    *,
    capsule_bits: float = 8,
    step: int = 8,
    maximum_rank: int = 512,
) -> int:
    if capsule_bits <= 0:
        raise ValueError("capsule_bits must be positive")
    if step <= 0 or maximum_rank <= 0:
        raise ValueError("step and maximum_rank must be positive")
    feasible = [
        point.rank
        for point in rank_frontier(
            range(step, maximum_rank + 1, step),
            capsule_bits=capsule_bits,
        )
        if point.pass_all
    ]
    if not feasible:
        raise RuntimeError("no positive rank fits the fixed Gate 0 envelope")
    return max(feasible)
