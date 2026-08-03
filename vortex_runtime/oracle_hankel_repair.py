from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from vortex_runtime.hankel_decision_program import (
    HankelDecisionProgram,
    decision_logits,
    predict_next_reduced,
    reconstruct_hidden,
    reduce_hidden,
)

GIB = 1024**3
TARGET_PARAMETERS = 405_849_243_648
TARGET_REPAIR_TRAFFIC_GIB = TARGET_PARAMETERS * 4 / 8 / GIB
TARGET_REPAIR_COMPUTE_GFLOP = 2 * TARGET_PARAMETERS / 1e9


@dataclass(frozen=True)
class OracleRepairStatistics:
    tokens: int
    repairs: int
    accepted_predictions: int
    accepted_fraction: float
    mean_repair_interval: float
    minimum_repair_interval: int
    maximum_repair_interval: int
    p50_repair_interval: float
    p90_repair_interval: float
    repair_positions: tuple[int, ...]
    nonfinite_repairs: int
    mismatch_repairs: int
    emitted_exact_rate: float
    projected_repair_traffic_gib_per_token: float
    projected_repair_compute_gflop_per_token: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["repair_positions"] = list(self.repair_positions)
        return payload


@dataclass
class OracleRepairRollout:
    emitted_tokens: torch.Tensor
    reduced_states: torch.Tensor
    hidden_states: torch.Tensor
    statistics: OracleRepairStatistics


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _intervals_from_repairs(tokens: int, repair_positions: list[int]) -> list[int]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if not repair_positions:
        return [tokens]
    intervals: list[int] = []
    previous = 0
    for position in repair_positions:
        if position <= previous or position > tokens:
            raise ValueError("repair positions must be increasing 1-based indices")
        intervals.append(position - previous)
        previous = position
    if previous < tokens:
        intervals.append(tokens - previous)
    return intervals


def oracle_repair_hankel_rollout(
    program: HankelDecisionProgram,
    *,
    initial_history: list[torch.Tensor],
    exact_control_tokens: torch.Tensor,
    exact_target_tokens: torch.Tensor,
    exact_hidden_states: torch.Tensor,
) -> OracleRepairRollout:
    """Use an impossible exact-token oracle to minimize recurrence repairs.

    Exact control tokens are supplied because every emitted token is repaired to
    the target when necessary. A state is repaired only when the projected
    program would emit a wrong token or any recurrence/logit value is non-finite.
    """

    controls = exact_control_tokens.detach().to("cpu", torch.long).reshape(-1)
    targets = exact_target_tokens.detach().to("cpu", torch.long).reshape(-1)
    exact_hidden = exact_hidden_states.detach().to("cpu", torch.float32)
    if controls.numel() == 0 or controls.numel() != targets.numel():
        raise ValueError("one exact control is required per exact target")
    if exact_hidden.ndim != 2 or exact_hidden.shape[0] != targets.numel():
        raise ValueError("one exact hidden state is required per target token")
    if len(initial_history) < program.order:
        raise ValueError("initial history is shorter than recurrence order")

    history = [
        item.detach().to("cpu", torch.float32).reshape(-1)
        for item in initial_history[: program.order]
    ]
    emitted: list[int] = []
    reduced_states: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    repair_positions: list[int] = []
    nonfinite_repairs = 0
    mismatch_repairs = 0

    for index in range(targets.numel()):
        target = int(targets[index].item())
        predicted = predict_next_reduced(
            program,
            history,
            int(controls[index].item()),
        )
        finite = _finite_tensor(predicted)
        if finite:
            logits = decision_logits(program, predicted)
            finite = _finite_tensor(logits)
        else:
            logits = torch.empty(0)
        predicted_token = int(torch.argmax(logits).item()) if finite else -1
        repair = not finite or predicted_token != target
        if repair:
            repaired = reduce_hidden(program, exact_hidden[index])
            if not _finite_tensor(repaired):
                raise RuntimeError("exact projected repair state is non-finite")
            next_state = repaired
            repair_positions.append(index + 1)
            if finite:
                mismatch_repairs += 1
            else:
                nonfinite_repairs += 1
        else:
            next_state = predicted

        emitted.append(target)
        reduced_states.append(next_state)
        hidden_states.append(reconstruct_hidden(program, next_state))
        history = [next_state] + history[: program.order - 1]

    intervals = _intervals_from_repairs(targets.numel(), repair_positions)
    interval_tensor = torch.tensor(intervals, dtype=torch.float64)
    repairs = len(repair_positions)
    mean_interval = targets.numel() / max(repairs, 1)
    emitted_tensor = torch.tensor(emitted, dtype=torch.long)
    emitted_exact_rate = float(torch.eq(emitted_tensor, targets).float().mean().item())
    statistics = OracleRepairStatistics(
        tokens=targets.numel(),
        repairs=repairs,
        accepted_predictions=targets.numel() - repairs,
        accepted_fraction=(targets.numel() - repairs) / targets.numel(),
        mean_repair_interval=mean_interval,
        minimum_repair_interval=min(intervals),
        maximum_repair_interval=max(intervals),
        p50_repair_interval=float(torch.quantile(interval_tensor, 0.50).item()),
        p90_repair_interval=float(torch.quantile(interval_tensor, 0.90).item()),
        repair_positions=tuple(repair_positions),
        nonfinite_repairs=nonfinite_repairs,
        mismatch_repairs=mismatch_repairs,
        emitted_exact_rate=emitted_exact_rate,
        projected_repair_traffic_gib_per_token=(
            repairs * TARGET_REPAIR_TRAFFIC_GIB / targets.numel()
        ),
        projected_repair_compute_gflop_per_token=(
            repairs * TARGET_REPAIR_COMPUTE_GFLOP / targets.numel()
        ),
    )
    return OracleRepairRollout(
        emitted_tokens=emitted_tensor,
        reduced_states=torch.stack(reduced_states, dim=0),
        hidden_states=torch.stack(hidden_states, dim=0),
        statistics=statistics,
    )


def repair_envelope_passes(
    statistics: OracleRepairStatistics,
    *,
    program_hot_compute_gflop_per_token: float,
    program_build_compute_gflop: float,
    horizon_tokens: int,
    required_mean_interval: float = 247.0,
    traffic_limit_gib_per_token: float = 2.4,
    compute_limit_gflop_per_token: float = 9.6,
) -> tuple[bool, float]:
    if horizon_tokens <= 0:
        raise ValueError("horizon must be positive")
    amortized_compute = (
        program_hot_compute_gflop_per_token
        + program_build_compute_gflop / horizon_tokens
        + statistics.projected_repair_compute_gflop_per_token
    )
    passes = bool(
        statistics.repairs <= 1
        and statistics.mean_repair_interval >= required_mean_interval
        and statistics.emitted_exact_rate == 1.0
        and statistics.projected_repair_traffic_gib_per_token
        <= traffic_limit_gib_per_token
        and amortized_compute <= compute_limit_gflop_per_token
        and math.isfinite(amortized_compute)
    )
    return passes, amortized_compute
