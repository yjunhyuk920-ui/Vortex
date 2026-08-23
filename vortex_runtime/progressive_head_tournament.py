from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import numpy as np


Q4_PAYLOAD_BYTES_PER_WEIGHT = 0.5
Q4_SCALE_ZERO_BYTES_PER_GROUP = 4
Q4_GROUP_SIZE = 128
ROW_STATE_BYTES_PER_ACTIVE_STAGE = 36  # int64 score R/W + uint64 residual R/W + row id/flag
ROW_NORM_BYTES = 8


def ceil_sqrt_int(values: np.ndarray) -> np.ndarray:
    """Exact ceil(sqrt(n)) for non-negative int64 values."""
    arr = np.asarray(values, dtype=np.int64)
    if np.any(arr < 0):
        raise ValueError("sqrt input must be non-negative")
    # Vectorized float seed, repaired exactly around the candidate.
    roots = np.floor(np.sqrt(arr.astype(np.float64))).astype(np.int64)
    while True:
        mask = (roots + 1) > 0
        too_low = mask & ((roots + 1) <= arr // (roots + 1))
        if not np.any(too_low):
            break
        roots[too_low] += 1
    while True:
        positive = roots > 0
        divisor = np.where(positive, roots, np.int64(1))
        too_high = positive & (roots > arr // divisor)
        if not np.any(too_high):
            break
        roots[too_high] -= 1
    exact = roots * roots == arr
    return roots + (~exact).astype(np.int64)


@dataclass(frozen=True)
class TournamentLedger:
    vocab_size: int
    dimension: int
    block_size: int
    stage_count: int
    winner: int
    reference_winner: int
    exact: bool
    stopped_early: bool
    stages_executed: int
    active_fraction_by_stage: tuple[float, ...]
    weight_element_fraction: float
    q4_payload_bytes: float
    q4_scale_zero_bytes: float
    row_state_bytes: float
    row_norm_bytes: float
    activation_and_order_bytes: float
    total_head_query_bytes: float
    full_q4_head_bytes: float
    head_query_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def full_q4_head_bytes(vocab_size: int, dimension: int) -> float:
    groups_per_row = math.ceil(dimension / Q4_GROUP_SIZE)
    return (
        vocab_size * dimension * Q4_PAYLOAD_BYTES_PER_WEIGHT
        + vocab_size * groups_per_row * Q4_SCALE_ZERO_BYTES_PER_GROUP
    )


def _argmax_lowest(values: np.ndarray) -> int:
    maximum = values.max()
    return int(np.flatnonzero(values == maximum)[0])


def activation_ordered_tournament_runtime(
    weights_q4: np.ndarray,
    activation_q8: np.ndarray,
    block_size: int,
) -> tuple[int, dict[str, object]]:
    """Execute the query-dependent tournament without a dense reference pass."""
    weights = np.asarray(weights_q4, dtype=np.int8)
    activation = np.asarray(activation_q8, dtype=np.int8)
    if weights.ndim != 2 or activation.ndim != 1 or weights.shape[1] != activation.shape[0]:
        raise ValueError("expected weights[V,D] and activation[D]")
    if np.any(weights < -8) or np.any(weights > 7):
        raise ValueError("weights must be signed Q4 in [-8, 7]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    vocab_size, dimension = weights.shape
    order = np.lexsort((np.arange(dimension, dtype=np.int64), -np.abs(activation.astype(np.int16))))
    stages = math.ceil(dimension / block_size)

    active = np.ones(vocab_size, dtype=bool)
    partial = np.zeros(vocab_size, dtype=np.int64)
    residual_w_sq = np.sum(weights.astype(np.int64) ** 2, axis=1, dtype=np.int64)
    residual_x_sq = int(np.sum(activation.astype(np.int64) ** 2, dtype=np.int64))

    active_fractions: list[float] = []
    q4_payload_bytes = 0.0
    q4_scale_zero_bytes = 0.0
    row_state_bytes = 0.0
    seen_groups: set[int] = set()
    weight_elements_read = 0
    stopped_early = False
    stages_executed = 0

    for stage in range(stages):
        start = stage * block_size
        stop = min((stage + 1) * block_size, dimension)
        dims = order[start:stop]
        active_ids = np.flatnonzero(active)
        active_count = int(active_ids.size)
        active_fractions.append(active_count / vocab_size)
        stages_executed += 1

        block = weights[np.ix_(active_ids, dims)].astype(np.int64)
        x_block = activation[dims].astype(np.int64)
        partial[active_ids] += block @ x_block
        residual_w_sq[active_ids] -= np.sum(block * block, axis=1, dtype=np.int64)
        residual_x_sq -= int(np.sum(x_block * x_block, dtype=np.int64))
        if residual_x_sq < 0 or np.any(residual_w_sq[active_ids] < 0):
            raise AssertionError("residual norm underflow")

        weight_elements_read += active_count * len(dims)
        q4_payload_bytes += active_count * len(dims) * Q4_PAYLOAD_BYTES_PER_WEIGHT
        groups = {int(dim) // Q4_GROUP_SIZE for dim in dims.tolist()}
        new_groups = groups - seen_groups
        q4_scale_zero_bytes += active_count * len(new_groups) * Q4_SCALE_ZERO_BYTES_PER_GROUP
        seen_groups.update(groups)
        row_state_bytes += active_count * ROW_STATE_BYTES_PER_ACTIVE_STAGE

        products = residual_w_sq[active_ids] * np.int64(residual_x_sq)
        radii = ceil_sqrt_int(products)
        lower = partial[active_ids] - radii
        upper = partial[active_ids] + radii
        best_lower = int(lower.max())
        keep_local = upper >= best_lower
        active[active_ids[~keep_local]] = False

        remaining = np.flatnonzero(active)
        if remaining.size == 1:
            stopped_early = True
            break
        rem_products = residual_w_sq[remaining] * np.int64(residual_x_sq)
        rem_radii = ceil_sqrt_int(rem_products)
        rem_lower = partial[remaining] - rem_radii
        rem_upper = partial[remaining] + rem_radii
        best_pos = int(np.argmax(rem_lower))
        candidate = int(remaining[best_pos])
        other_mask = remaining != candidate
        if np.all(rem_lower[best_pos] > rem_upper[other_mask]):
            stopped_early = True
            active[:] = False
            active[candidate] = True
            break

    final_active = np.flatnonzero(active)
    if stopped_early and stages_executed < stages:
        winner = int(final_active[0])
    else:
        active_scores = partial[final_active]
        local_winner = _argmax_lowest(active_scores)
        winner = int(final_active[local_winner])

    row_norm_bytes = vocab_size * ROW_NORM_BYTES
    activation_and_order_bytes = dimension * (1 + 4)
    total_head_query_bytes = (
        q4_payload_bytes
        + q4_scale_zero_bytes
        + row_state_bytes
        + row_norm_bytes
        + activation_and_order_bytes
    )
    full_bytes = full_q4_head_bytes(vocab_size, dimension)
    runtime = {
        "vocab_size": vocab_size,
        "dimension": dimension,
        "block_size": block_size,
        "stage_count": stages,
        "winner": winner,
        "stopped_early": stopped_early,
        "stages_executed": stages_executed,
        "active_fraction_by_stage": tuple(active_fractions),
        "weight_element_fraction": weight_elements_read / (vocab_size * dimension),
        "q4_payload_bytes": q4_payload_bytes,
        "q4_scale_zero_bytes": q4_scale_zero_bytes,
        "row_state_bytes": row_state_bytes,
        "row_norm_bytes": float(row_norm_bytes),
        "activation_and_order_bytes": float(activation_and_order_bytes),
        "total_head_query_bytes": total_head_query_bytes,
        "full_q4_head_bytes": full_bytes,
        "head_query_fraction": total_head_query_bytes / full_bytes,
    }
    return winner, runtime


def exact_activation_ordered_tournament(
    weights_q4: np.ndarray,
    activation_q8: np.ndarray,
    block_size: int,
) -> TournamentLedger:
    """Run the mechanism, then perform an independent dense reference check.

    The full dense dot below is validation-only and is excluded from the runtime
    ledger. The returned winner is produced by activation_ordered_tournament_runtime.
    """
    weights = np.asarray(weights_q4, dtype=np.int8)
    activation = np.asarray(activation_q8, dtype=np.int8)
    winner, runtime = activation_ordered_tournament_runtime(weights, activation, block_size)
    full_scores = weights.astype(np.int64) @ activation.astype(np.int64)
    reference_winner = _argmax_lowest(full_scores)
    return TournamentLedger(
        **runtime,
        reference_winner=reference_winner,
        exact=winner == reference_winner,
    )


def make_random_q4_matrix(rng: np.random.Generator, vocab_size: int, dimension: int) -> np.ndarray:
    # Symmetric dense Q4 population, representative of a non-low-rank hard control.
    return rng.integers(-8, 8, size=(vocab_size, dimension), dtype=np.int8)


def make_structured_positive_control(
    rng: np.random.Generator, vocab_size: int, dimension: int, decisive_dimensions: int
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = rng.integers(-2, 3, size=(vocab_size, dimension), dtype=np.int8)
    activation = np.zeros(dimension, dtype=np.int8)
    activation[:decisive_dimensions] = np.int8(127)
    winner = 0
    weights[winner, :decisive_dimensions] = np.int8(7)
    weights[1:, :decisive_dimensions] = np.int8(-8)
    return weights, activation, winner


def make_late_decision_control(
    rng: np.random.Generator, vocab_size: int, dimension: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((vocab_size, dimension), dtype=np.int8)
    activation = np.ones(dimension, dtype=np.int8)
    # Equal high-magnitude prefixes keep all rows plausible. Distinguishing signal
    # is deliberately placed in the smallest-|x| tie order's final dimensions.
    weights[:, : dimension - block_size] = np.int8(1)
    tail = rng.integers(-8, 8, size=(vocab_size, block_size), dtype=np.int8)
    weights[:, dimension - block_size :] = tail
    return weights, activation
