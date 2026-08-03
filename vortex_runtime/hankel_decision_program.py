from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

GIB = 1024**3
LiftKind = Literal["linear", "quadratic", "bilinear", "full"]


@dataclass(frozen=True)
class HankelProgramBudget:
    hidden_size: int
    vocabulary_size: int
    state_rank: int
    control_rank: int
    order: int
    lift: str
    feature_size: int
    state_basis_gib: float
    mean_gib: float
    control_table_gib: float
    projected_lm_gib: float
    recurrence_gib: float
    history_gib: float
    total_program_gib: float
    hot_compute_gflop_per_token: float
    build_compute_gflop: float
    compute_limit_gflop_per_token: float
    minimum_build_reuse_tokens: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class HankelFitDiagnostics:
    samples: int
    requested_state_rank: int
    effective_state_rank: int
    requested_control_rank: int
    effective_control_rank: int
    order: int
    lift: str
    feature_size: int
    ridge: float
    training_relative_l2_error: float
    training_cosine_error: float
    regularized_condition_number: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass
class HankelDecisionProgram:
    mean_hidden: torch.Tensor
    state_basis: torch.Tensor
    control_basis: torch.Tensor
    control_table: torch.Tensor
    projected_lm: torch.Tensor
    projected_lm_bias: torch.Tensor
    theta: torch.Tensor
    order: int
    lift: LiftKind
    diagnostics: HankelFitDiagnostics

    @property
    def state_rank(self) -> int:
        return int(self.state_basis.shape[1])

    @property
    def control_rank(self) -> int:
        return int(self.control_basis.shape[1])

    @property
    def feature_size(self) -> int:
        return int(self.theta.shape[0])


@dataclass
class HankelRollout:
    token_ids: torch.Tensor
    reduced_states: torch.Tensor
    hidden_states: torch.Tensor
    logits: torch.Tensor


def _validate_lift(lift: str) -> LiftKind:
    if lift not in {"linear", "quadratic", "bilinear", "full"}:
        raise ValueError("lift must be linear, quadratic, bilinear, or full")
    return lift  # type: ignore[return-value]


def hankel_feature_size(
    *,
    state_rank: int,
    control_rank: int,
    order: int,
    lift: str,
) -> int:
    _validate_lift(lift)
    if min(state_rank, control_rank, order) <= 0:
        raise ValueError("ranks and order must be positive")
    size = order * state_rank + control_rank + 1
    if lift in {"quadratic", "full"}:
        size += state_rank
    if lift in {"bilinear", "full"}:
        size += state_rank
    return size


def hankel_program_budget(
    *,
    hidden_size: int = 16_384,
    vocabulary_size: int = 128_256,
    state_rank: int,
    control_rank: int,
    order: int,
    lift: str,
    basis_bits: int = 8,
    table_bits: int = 8,
    table_scale_bits: int = 16,
    mean_bits: int = 16,
    recurrence_bits: int = 16,
    history_bits: int = 16,
    compute_limit_gflop_per_token: float = 9.6,
) -> HankelProgramBudget:
    feature_size = hankel_feature_size(
        state_rank=state_rank,
        control_rank=control_rank,
        order=order,
        lift=lift,
    )
    if min(
        hidden_size,
        vocabulary_size,
        basis_bits,
        table_bits,
        table_scale_bits,
        mean_bits,
        recurrence_bits,
        history_bits,
    ) <= 0:
        raise ValueError("dimensions and precisions must be positive")
    if compute_limit_gflop_per_token <= 0:
        raise ValueError("compute limit must be positive")

    state_basis = hidden_size * state_rank * basis_bits / 8 / GIB
    mean = hidden_size * mean_bits / 8 / GIB
    # One row scale for each projected token vector.
    control_table = vocabulary_size * (
        control_rank * table_bits / 8 + table_scale_bits / 8
    ) / GIB
    projected_lm = vocabulary_size * (
        state_rank * table_bits / 8 + table_scale_bits / 8
    ) / GIB
    recurrence = feature_size * state_rank * recurrence_bits / 8 / GIB
    history = order * state_rank * history_bits / 8 / GIB
    total = state_basis + mean + control_table + projected_lm + recurrence + history
    hot_compute = (
        2 * vocabulary_size * state_rank
        + 2 * feature_size * state_rank
    ) / 1e9
    build_compute = 2 * vocabulary_size * hidden_size * (
        state_rank + control_rank
    ) / 1e9
    available = compute_limit_gflop_per_token - hot_compute
    reuse = float("inf") if available <= 0 else build_compute / available
    return HankelProgramBudget(
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        state_rank=state_rank,
        control_rank=control_rank,
        order=order,
        lift=lift,
        feature_size=feature_size,
        state_basis_gib=state_basis,
        mean_gib=mean,
        control_table_gib=control_table,
        projected_lm_gib=projected_lm,
        recurrence_gib=recurrence,
        history_gib=history,
        total_program_gib=total,
        hot_compute_gflop_per_token=hot_compute,
        build_compute_gflop=build_compute,
        compute_limit_gflop_per_token=compute_limit_gflop_per_token,
        minimum_build_reuse_tokens=reuse,
    )


def _orthonormal_basis(samples: torch.Tensor, rank: int) -> torch.Tensor:
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must be a non-empty matrix")
    if rank <= 0:
        raise ValueError("rank must be positive")
    usable = min(rank, samples.shape[0], samples.shape[1])
    if float(torch.linalg.vector_norm(samples).item()) <= 1e-12:
        return torch.zeros(samples.shape[1], usable, dtype=torch.float32)
    _, _, vh = torch.linalg.svd(samples, full_matrices=False)
    return vh[:usable].T.to(torch.float32).contiguous()


def _align_control(control: torch.Tensor, state_rank: int) -> torch.Tensor:
    if control.numel() == state_rank:
        return control
    if control.numel() > state_rank:
        return control[:state_rank]
    padding = torch.zeros(
        state_rank - control.numel(),
        dtype=control.dtype,
        device=control.device,
    )
    return torch.cat((control, padding), dim=0)


def build_hankel_feature(
    history: list[torch.Tensor],
    control: torch.Tensor,
    *,
    order: int,
    lift: str,
) -> torch.Tensor:
    lift_kind = _validate_lift(lift)
    if len(history) < order:
        raise ValueError("history is shorter than recurrence order")
    state_rank = int(history[0].numel())
    if any(item.numel() != state_rank for item in history[:order]):
        raise ValueError("history states must share one rank")
    current = history[0].reshape(-1)
    parts = [item.reshape(-1) for item in history[:order]]
    parts.append(control.reshape(-1))
    if lift_kind in {"quadratic", "full"}:
        parts.append(current * current)
    if lift_kind in {"bilinear", "full"}:
        parts.append(current * _align_control(control.reshape(-1), state_rank))
    parts.append(torch.ones(1, dtype=current.dtype, device=current.device))
    return torch.cat(parts, dim=0)


def fit_hankel_decision_program(
    *,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    state_rank: int,
    control_rank: int,
    order: int,
    lift: str,
    ridge: float = 1e-4,
) -> HankelDecisionProgram:
    """Compile a prompt trajectory into a controlled reduced-state recurrence."""

    lift_kind = _validate_lift(lift)
    hidden = hidden_states.detach().to("cpu", torch.float32)
    tokens = token_ids.detach().to("cpu", torch.long).reshape(-1)
    embedding = embedding_weight.detach().to("cpu", torch.float32)
    lm_weight = lm_head_weight.detach().to("cpu", torch.float32)
    if hidden.ndim != 2 or tokens.numel() != hidden.shape[0]:
        raise ValueError("one token id is required per prompt hidden state")
    if hidden.shape[0] <= order:
        raise ValueError("prompt trajectory is too short for recurrence order")
    if embedding.ndim != 2 or embedding.shape[1] != hidden.shape[1]:
        raise ValueError("embedding table must match hidden size")
    if lm_weight.ndim != 2 or lm_weight.shape[1] != hidden.shape[1]:
        raise ValueError("LM head must match hidden size")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    mean_hidden = hidden.mean(dim=0)
    centered = hidden - mean_hidden
    state_basis = _orthonormal_basis(centered, state_rank)
    reduced = centered @ state_basis

    prompt_embeddings = embedding[tokens]
    control_basis = _orthonormal_basis(prompt_embeddings, control_rank)
    control_table = embedding @ control_basis

    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in range(order - 1, hidden.shape[0] - 1):
        history = [reduced[index - lag] for lag in range(order)]
        next_control = control_table[tokens[index + 1]]
        features.append(
            build_hankel_feature(
                history,
                next_control,
                order=order,
                lift=lift_kind,
            )
        )
        targets.append(reduced[index + 1])
    design = torch.stack(features, dim=0)
    target = torch.stack(targets, dim=0)
    gram = design.T @ design
    scale = float(torch.trace(gram).item()) / max(gram.shape[0], 1)
    regularization = ridge * max(scale, 1e-12)
    regularized = gram + regularization * torch.eye(
        gram.shape[0], dtype=gram.dtype
    )
    rhs = design.T @ target
    try:
        theta = torch.linalg.solve(regularized, rhs)
    except RuntimeError:
        theta = torch.linalg.pinv(regularized) @ rhs

    fitted = design @ theta
    residual = fitted - target
    relative_error = float(
        torch.linalg.vector_norm(residual).item()
        / max(float(torch.linalg.vector_norm(target).item()), 1e-12)
    )
    fitted_flat = fitted.reshape(-1)
    target_flat = target.reshape(-1)
    cosine = float(
        torch.dot(fitted_flat, target_flat).item()
        / max(
            float(torch.linalg.vector_norm(fitted_flat).item())
            * float(torch.linalg.vector_norm(target_flat).item()),
            1e-12,
        )
    )
    condition = float(torch.linalg.cond(regularized).item())

    projected_lm = lm_weight @ state_basis
    projected_bias = lm_weight @ mean_hidden
    if lm_head_bias is not None:
        projected_bias = projected_bias + lm_head_bias.detach().to(
            "cpu", torch.float32
        )
    diagnostics = HankelFitDiagnostics(
        samples=design.shape[0],
        requested_state_rank=state_rank,
        effective_state_rank=state_basis.shape[1],
        requested_control_rank=control_rank,
        effective_control_rank=control_basis.shape[1],
        order=order,
        lift=lift_kind,
        feature_size=design.shape[1],
        ridge=ridge,
        training_relative_l2_error=relative_error,
        training_cosine_error=1.0 - cosine,
        regularized_condition_number=condition,
    )
    return HankelDecisionProgram(
        mean_hidden=mean_hidden.contiguous(),
        state_basis=state_basis,
        control_basis=control_basis,
        control_table=control_table.contiguous(),
        projected_lm=projected_lm.contiguous(),
        projected_lm_bias=projected_bias.contiguous(),
        theta=theta.contiguous(),
        order=order,
        lift=lift_kind,
        diagnostics=diagnostics,
    )


def reduce_hidden(program: HankelDecisionProgram, hidden: torch.Tensor) -> torch.Tensor:
    source = hidden.detach().to("cpu", torch.float32).reshape(-1)
    return (program.state_basis.T @ (source - program.mean_hidden)).contiguous()


def reconstruct_hidden(program: HankelDecisionProgram, reduced: torch.Tensor) -> torch.Tensor:
    return (
        program.mean_hidden
        + program.state_basis @ reduced.to("cpu", torch.float32).reshape(-1)
    ).contiguous()


def decision_logits(program: HankelDecisionProgram, reduced: torch.Tensor) -> torch.Tensor:
    return (
        program.projected_lm @ reduced.to("cpu", torch.float32).reshape(-1)
        + program.projected_lm_bias
    ).contiguous()


def predict_next_reduced(
    program: HankelDecisionProgram,
    history: list[torch.Tensor],
    token_id: int,
) -> torch.Tensor:
    if token_id < 0 or token_id >= program.control_table.shape[0]:
        raise ValueError("token id is outside control table")
    feature = build_hankel_feature(
        history,
        program.control_table[token_id],
        order=program.order,
        lift=program.lift,
    )
    return (feature @ program.theta).contiguous()


def rollout_hankel_program(
    program: HankelDecisionProgram,
    *,
    initial_history: list[torch.Tensor],
    first_control_token: int,
    steps: int,
    forced_control_tokens: list[int] | None = None,
) -> HankelRollout:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if len(initial_history) < program.order:
        raise ValueError("initial history is shorter than program order")
    if forced_control_tokens is not None and len(forced_control_tokens) < steps:
        raise ValueError("forced control sequence is shorter than steps")

    history = [item.detach().to("cpu", torch.float32).reshape(-1) for item in initial_history]
    control_token = int(first_control_token)
    tokens: list[int] = []
    reduced_states: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    logits_values: list[torch.Tensor] = []
    for step in range(steps):
        if forced_control_tokens is not None:
            control_token = int(forced_control_tokens[step])
        next_reduced = predict_next_reduced(
            program,
            history,
            control_token,
        )
        logits = decision_logits(program, next_reduced)
        predicted_token = int(torch.argmax(logits).item())
        tokens.append(predicted_token)
        reduced_states.append(next_reduced)
        hidden_states.append(reconstruct_hidden(program, next_reduced))
        logits_values.append(logits)
        history = [next_reduced] + history[: program.order - 1]
        if forced_control_tokens is None:
            control_token = predicted_token

    return HankelRollout(
        token_ids=torch.tensor(tokens, dtype=torch.long),
        reduced_states=torch.stack(reduced_states, dim=0),
        hidden_states=torch.stack(hidden_states, dim=0),
        logits=torch.stack(logits_values, dim=0),
    )
