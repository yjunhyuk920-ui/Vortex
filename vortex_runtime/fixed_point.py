"""Model-independent continuous block fixed-point solvers for EXP-049.

One callback evaluation is one synchronous target block pass. Hard proposals are
returned by the callback so numerical convergence and exact-token prefix length
remain separate measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np


class FixedPointError(ValueError):
    """Raised when a solver or map violates the fail-closed contract."""


@dataclass(frozen=True)
class MapEvaluation:
    projected_state: np.ndarray
    hard_tokens: tuple[int, ...]
    projection_read_bytes: int = 0
    projection_ops: int = 0

    def validate(self, expected_shape: tuple[int, ...]) -> None:
        state = np.asarray(self.projected_state)
        if state.shape != expected_shape:
            raise FixedPointError(
                f"map returned shape {state.shape}, expected {expected_shape}"
            )
        if not np.issubdtype(state.dtype, np.floating):
            raise FixedPointError("projected_state must have floating dtype")
        if not np.all(np.isfinite(state)):
            raise FixedPointError("map returned NaN or Inf")
        if len(self.hard_tokens) != expected_shape[0]:
            raise FixedPointError("hard token count must equal block width")
        if any(
            not isinstance(token, (int, np.integer)) or int(token) < 0
            for token in self.hard_tokens
        ):
            raise FixedPointError("hard tokens must be non-negative integers")
        if self.projection_read_bytes < 0 or self.projection_ops < 0:
            raise FixedPointError("projection accounting must be non-negative")


BlockMap = Callable[[np.ndarray], MapEvaluation]


@dataclass(frozen=True)
class SolverSnapshot:
    iteration: int
    hard_tokens: tuple[int, ...]
    residual_l2: float
    residual_linf: float
    state_l2: float
    coefficient_abs_max: float
    anderson_condition: float | None
    numerical_fallbacks: int
    projection_read_bytes_total: int
    projection_ops_total: int


@dataclass(frozen=True)
class SolverResult:
    method: str
    snapshots: tuple[SolverSnapshot, ...]
    final_state: np.ndarray
    target_solver_full_streams: int
    projection_read_bytes: int
    projection_ops: int
    numerical_fallbacks: int
    anderson_history_bytes_peak: int
    future_information_used: bool

    def snapshot_at(self, iteration: int) -> SolverSnapshot:
        for snapshot in self.snapshots:
            if snapshot.iteration == iteration:
                return snapshot
        raise FixedPointError(f"no snapshot recorded at iteration {iteration}")


def _validate_initial(
    initial_state: np.ndarray, iterations: int, damping: float
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] <= 0 or state.shape[1] <= 0:
        raise FixedPointError("initial_state must have shape [block, hidden]")
    if not np.all(np.isfinite(state)):
        raise FixedPointError("initial_state contains NaN or Inf")
    if iterations <= 0:
        raise FixedPointError("iterations must be positive")
    if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise FixedPointError("damping must be finite and in (0, 1]")
    return state.copy()


def _record_steps(iterations: int, requested: Iterable[int]) -> set[int]:
    steps = {int(step) for step in requested}
    if any(step <= 0 or step > iterations for step in steps):
        raise FixedPointError("record_steps must lie in [1, iterations]")
    steps.add(iterations)
    return steps


def run_damped_picard(
    initial_state: np.ndarray,
    *,
    map_fn: BlockMap,
    iterations: int,
    damping: float,
    record_steps: Sequence[int] = (1, 2, 4),
    future_information_used: bool = False,
) -> SolverResult:
    state = _validate_initial(initial_state, iterations, damping)
    steps = _record_steps(iterations, record_steps)
    snapshots: list[SolverSnapshot] = []
    read_bytes = 0
    projection_ops = 0

    for iteration in range(1, iterations + 1):
        evaluation = map_fn(state)
        evaluation.validate(state.shape)
        projected = np.asarray(evaluation.projected_state, dtype=np.float64)
        residual = projected - state
        next_state = state + damping * residual
        if not np.all(np.isfinite(next_state)):
            raise FixedPointError("Picard update produced NaN or Inf")
        read_bytes += int(evaluation.projection_read_bytes)
        projection_ops += int(evaluation.projection_ops)
        if iteration in steps:
            snapshots.append(
                SolverSnapshot(
                    iteration=iteration,
                    hard_tokens=tuple(int(token) for token in evaluation.hard_tokens),
                    residual_l2=float(np.linalg.norm(residual)),
                    residual_linf=float(np.max(np.abs(residual))),
                    state_l2=float(np.linalg.norm(next_state)),
                    coefficient_abs_max=1.0,
                    anderson_condition=None,
                    numerical_fallbacks=0,
                    projection_read_bytes_total=read_bytes,
                    projection_ops_total=projection_ops,
                )
            )
        state = next_state

    return SolverResult(
        method="damped_picard",
        snapshots=tuple(snapshots),
        final_state=state,
        target_solver_full_streams=iterations,
        projection_read_bytes=read_bytes,
        projection_ops=projection_ops,
        numerical_fallbacks=0,
        anderson_history_bytes_peak=0,
        future_information_used=bool(future_information_used),
    )


def _anderson_coefficients(
    residuals: Sequence[np.ndarray],
    *,
    regularization: float,
    coefficient_clip: float,
    condition_limit: float,
) -> tuple[np.ndarray, float]:
    history = len(residuals)
    if history <= 1:
        return np.ones(1, dtype=np.float64), 1.0
    matrix = np.stack([residual.reshape(-1) for residual in residuals], axis=1)
    gram = matrix.T @ matrix
    scale = max(1.0, float(np.trace(gram)) / history)
    system = gram + regularization * scale * np.eye(history, dtype=np.float64)
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError("ill-conditioned Anderson system")
    kkt = np.empty((history + 1, history + 1), dtype=np.float64)
    kkt[:history, :history] = system
    kkt[:history, history] = 1.0
    kkt[history, :history] = 1.0
    kkt[history, history] = 0.0
    rhs = np.zeros(history + 1, dtype=np.float64)
    rhs[history] = 1.0
    coefficients = np.linalg.solve(kkt, rhs)[:history]
    if not np.all(np.isfinite(coefficients)):
        raise np.linalg.LinAlgError("non-finite Anderson coefficients")
    coefficients = np.clip(coefficients, -coefficient_clip, coefficient_clip)
    total = float(coefficients.sum())
    if not math.isfinite(total) or abs(total) < 1e-12:
        raise np.linalg.LinAlgError("Anderson coefficients cannot be normalized")
    coefficients /= total
    if not np.all(np.isfinite(coefficients)):
        raise np.linalg.LinAlgError("normalized Anderson coefficients are non-finite")
    return coefficients, condition


def run_anderson(
    initial_state: np.ndarray,
    *,
    map_fn: BlockMap,
    iterations: int,
    history_size: int,
    damping: float,
    regularization: float = 1e-8,
    coefficient_clip: float = 10.0,
    condition_limit: float = 1e12,
    record_steps: Sequence[int] = (1, 2, 4),
    future_information_used: bool = False,
) -> SolverResult:
    state = _validate_initial(initial_state, iterations, damping)
    if history_size <= 0:
        raise FixedPointError("history_size must be positive")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise FixedPointError("regularization must be finite and non-negative")
    if not math.isfinite(coefficient_clip) or coefficient_clip < 1.0:
        raise FixedPointError("coefficient_clip must be finite and >=1")
    if not math.isfinite(condition_limit) or condition_limit <= 1.0:
        raise FixedPointError("condition_limit must be finite and >1")

    steps = _record_steps(iterations, record_steps)
    states: list[np.ndarray] = []
    projected_states: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    snapshots: list[SolverSnapshot] = []
    read_bytes = 0
    projection_ops = 0
    fallbacks = 0
    history_bytes_peak = 0

    for iteration in range(1, iterations + 1):
        evaluation = map_fn(state)
        evaluation.validate(state.shape)
        projected = np.asarray(evaluation.projected_state, dtype=np.float64)
        residual = projected - state
        states.append(state.copy())
        projected_states.append(projected.copy())
        residuals.append(residual.copy())
        if len(states) > history_size:
            states.pop(0)
            projected_states.pop(0)
            residuals.pop(0)
        history_bytes_peak = max(
            history_bytes_peak,
            sum(item.nbytes for item in states + projected_states + residuals),
        )

        coefficient_abs_max = 1.0
        condition: float | None = None
        try:
            coefficients, condition = _anderson_coefficients(
                residuals,
                regularization=regularization,
                coefficient_clip=coefficient_clip,
                condition_limit=condition_limit,
            )
            coefficient_abs_max = float(np.max(np.abs(coefficients)))
            mixed_state = sum(
                coefficient * candidate
                for coefficient, candidate in zip(coefficients, states)
            )
            mixed_projected = sum(
                coefficient * candidate
                for coefficient, candidate in zip(coefficients, projected_states)
            )
            next_state = (
                (1.0 - damping) * mixed_state + damping * mixed_projected
            )
            if not np.all(np.isfinite(next_state)):
                raise np.linalg.LinAlgError("Anderson update produced NaN or Inf")
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            fallbacks += 1
            next_state = state + damping * residual
            coefficient_abs_max = 1.0
            condition = None
            if not np.all(np.isfinite(next_state)):
                raise FixedPointError("Anderson fallback produced NaN or Inf")

        read_bytes += int(evaluation.projection_read_bytes)
        projection_ops += int(evaluation.projection_ops)
        if iteration in steps:
            snapshots.append(
                SolverSnapshot(
                    iteration=iteration,
                    hard_tokens=tuple(int(token) for token in evaluation.hard_tokens),
                    residual_l2=float(np.linalg.norm(residual)),
                    residual_linf=float(np.max(np.abs(residual))),
                    state_l2=float(np.linalg.norm(next_state)),
                    coefficient_abs_max=coefficient_abs_max,
                    anderson_condition=condition,
                    numerical_fallbacks=fallbacks,
                    projection_read_bytes_total=read_bytes,
                    projection_ops_total=projection_ops,
                )
            )
        state = next_state

    return SolverResult(
        method=f"anderson_m{history_size}",
        snapshots=tuple(snapshots),
        final_state=state,
        target_solver_full_streams=iterations,
        projection_read_bytes=read_bytes,
        projection_ops=projection_ops,
        numerical_fallbacks=fallbacks,
        anderson_history_bytes_peak=history_bytes_peak,
        future_information_used=bool(future_information_used),
    )


def matching_prefix(proposal: Sequence[int], reference: Sequence[int]) -> int:
    count = 0
    for proposed, exact in zip(proposal, reference):
        if int(proposed) != int(exact):
            break
        count += 1
    return count


class GatedTriangularChain:
    """Finite causal oracle revealing at most one new exact position per round.

    Position zero reveals its hidden token. Position i>0 reveals its hidden
    token only when the submitted predecessor hardens to the exact hidden token
    at i-1; otherwise it emits a fixed decoy. Hidden suffixes therefore remain
    indistinguishable until their predecessor is resolved.
    """

    def __init__(
        self,
        exact_tokens: Sequence[int],
        *,
        vocab_size: int,
        decoy_token: int = 0,
    ) -> None:
        self.exact_tokens = tuple(int(token) for token in exact_tokens)
        if not self.exact_tokens:
            raise FixedPointError("exact_tokens must not be empty")
        if vocab_size <= 1:
            raise FixedPointError("vocab_size must exceed one")
        if any(token < 0 or token >= vocab_size for token in self.exact_tokens):
            raise FixedPointError("exact token is outside vocabulary")
        if decoy_token < 0 or decoy_token >= vocab_size:
            raise FixedPointError("decoy token is outside vocabulary")
        if decoy_token in self.exact_tokens:
            raise FixedPointError("decoy token must differ from every exact token")
        self.vocab_size = int(vocab_size)
        self.decoy_token = int(decoy_token)
        self.embedding = np.eye(self.vocab_size, dtype=np.float64)

    @property
    def block_size(self) -> int:
        return len(self.exact_tokens)

    def zero_state(self) -> np.ndarray:
        return np.zeros((self.block_size, self.vocab_size), dtype=np.float64)

    def map(self, state: np.ndarray) -> MapEvaluation:
        candidate = np.asarray(state, dtype=np.float64)
        if candidate.shape != (self.block_size, self.vocab_size):
            raise FixedPointError("triangular state shape mismatch")
        if not np.all(np.isfinite(candidate)):
            raise FixedPointError("triangular state contains NaN or Inf")
        predecessor_tokens = np.argmax(candidate, axis=1)
        output_tokens = [self.exact_tokens[0]]
        for index in range(1, self.block_size):
            if int(predecessor_tokens[index - 1]) == self.exact_tokens[index - 1]:
                output_tokens.append(self.exact_tokens[index])
            else:
                output_tokens.append(self.decoy_token)
        projected = self.embedding[np.asarray(output_tokens, dtype=np.int64)]
        return MapEvaluation(
            projected_state=projected,
            hard_tokens=tuple(output_tokens),
            projection_read_bytes=int(projected.nbytes),
            projection_ops=self.block_size,
        )


def triangular_transcript_indistinguishable(
    first: GatedTriangularChain,
    second: GatedTriangularChain,
    state: np.ndarray,
    *,
    unresolved_position: int,
) -> bool:
    if first.block_size != second.block_size or first.vocab_size != second.vocab_size:
        raise FixedPointError("chain dimensions must match")
    if not 1 <= unresolved_position < first.block_size:
        raise FixedPointError("unresolved_position must be inside the block")
    if first.exact_tokens[:unresolved_position] != second.exact_tokens[:unresolved_position]:
        raise FixedPointError("chains must share the resolved hidden prefix")
    candidate = np.asarray(state, dtype=np.float64)
    predecessor = int(np.argmax(candidate[unresolved_position - 1]))
    if predecessor in {
        first.exact_tokens[unresolved_position - 1],
        second.exact_tokens[unresolved_position - 1],
    }:
        raise FixedPointError("predecessor must remain unresolved")
    first_eval = first.map(candidate)
    second_eval = second.map(candidate)
    return (
        first_eval.hard_tokens[unresolved_position:]
        == second_eval.hard_tokens[unresolved_position:]
    )
