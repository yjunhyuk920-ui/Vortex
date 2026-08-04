"""Sound output-head demand lower bounds for activation-conditioned lazy execution.

The registered mechanism reveals full-vocabulary column stripes of a linear output
head.  For a known exact hidden vector and exact real-arithmetic winner, an unread
coordinate may contribute adversarially to every winner-versus-competitor margin.
The functions below derive a necessary number of revealed stripes.  They do not
claim a deployable scheduler or bitwise floating-point replay equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OutputHeadDemandLowerBound:
    vocabulary_size: int
    hidden_width: int
    tile_columns: int
    tile_count: int
    winner_index: int
    runner_up_index: int
    exact_real_margin: float
    necessary_tile_count: int
    necessary_column_count: int
    head_weight_fraction_lower_bound: float
    hardest_competitor_index: int
    domain_winner_mismatch_count: int
    nonfinite_count: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "hidden_width": self.hidden_width,
            "tile_columns": self.tile_columns,
            "tile_count": self.tile_count,
            "winner_index": self.winner_index,
            "runner_up_index": self.runner_up_index,
            "exact_real_margin": self.exact_real_margin,
            "necessary_tile_count": self.necessary_tile_count,
            "necessary_column_count": self.necessary_column_count,
            "head_weight_fraction_lower_bound": (
                self.head_weight_fraction_lower_bound
            ),
            "hardest_competitor_index": self.hardest_competitor_index,
            "domain_winner_mismatch_count": self.domain_winner_mismatch_count,
            "nonfinite_count": self.nonfinite_count,
        }


def _tile_widths(hidden_width: int, tile_columns: int) -> tuple[int, ...]:
    if hidden_width <= 0 or tile_columns <= 0:
        raise ValueError("hidden width and tile columns must be positive")
    return tuple(
        min(tile_columns, hidden_width - start)
        for start in range(0, hidden_width, tile_columns)
    )


def _minimum_columns_for_tile_count(
    widths: Sequence[int], tile_count: int
) -> int:
    if tile_count < 0 or tile_count > len(widths):
        raise ValueError("invalid tile count")
    return sum(sorted(int(value) for value in widths)[:tile_count])


def exact_real_logits(
    weight: np.ndarray,
    hidden: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.asarray(weight)
    vector = np.asarray(hidden)
    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("weight must be 2D and hidden must be 1D")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("weight and hidden dimensions do not match")
    logits = matrix.astype(np.float64, copy=False) @ vector.astype(
        np.float64, copy=False
    )
    if bias is not None:
        offset = np.asarray(bias, dtype=np.float64)
        if offset.shape != (matrix.shape[0],):
            raise ValueError("bias shape mismatch")
        logits = logits + offset
    return np.asarray(logits, dtype=np.float64)


def analyze_output_head_demand_lower_bound(
    weight: np.ndarray,
    hidden: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    tile_columns: int = 1,
    expected_winner: int | None = None,
    competitor_chunk_rows: int = 4096,
) -> OutputHeadDemandLowerBound:
    """Return a sound necessary stripe count for exact winner certification.

    For competitor ``i`` and hidden coordinate ``j`` define
    ``d_ij = (w_winner,j - w_i,j) * h_j``.  After revealing a set S,
    certification requires

        bias_diff + sum_{j in S} d_ij > sum_{j not in S} |d_ij|.

    Equivalently, revealed stripes must collect more than
    ``sum_j |d_ij| - bias_diff`` of gain ``2*max(d_ij, 0)``.  The independently
    optimal stripe count for every competitor is a necessary condition for any
    single subset that certifies all competitors.  Their maximum is therefore a
    rigorous lower bound, while still granting an impossible competitor-specific
    oracle ordering.
    """
    matrix = np.asarray(weight)
    vector = np.asarray(hidden)
    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("weight must be 2D and hidden must be 1D")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("weight and hidden dimensions do not match")
    if competitor_chunk_rows <= 0:
        raise ValueError("competitor chunk size must be positive")

    widths = _tile_widths(int(matrix.shape[1]), int(tile_columns))
    logits = exact_real_logits(matrix, vector, bias)
    nonfinite_count = int(np.size(logits) - np.count_nonzero(np.isfinite(logits)))
    nonfinite_count += int(
        np.size(matrix) - np.count_nonzero(np.isfinite(matrix))
    )
    nonfinite_count += int(
        np.size(vector) - np.count_nonzero(np.isfinite(vector))
    )
    if nonfinite_count:
        raise ValueError("non-finite output-head inputs or logits")

    order = np.argsort(logits)
    winner = int(order[-1])
    runner_up = int(order[-2]) if len(order) > 1 else winner
    mismatch = int(expected_winner is not None and winner != expected_winner)
    exact_margin = float(logits[winner] - logits[runner_up])
    if not exact_margin > 0.0:
        raise ValueError("winner margin must be strictly positive")

    bias64 = (
        np.zeros(matrix.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64)
    )
    winner_weight = np.asarray(matrix[winner], dtype=np.float64)
    hidden64 = np.asarray(vector, dtype=np.float64)
    winner_bias = float(bias64[winner])

    necessary_tiles = 0
    hardest = runner_up
    domain_violations = 0
    rows = int(matrix.shape[0])
    width = int(matrix.shape[1])
    padded_width = len(widths) * int(tile_columns)

    for start in range(0, rows, competitor_chunk_rows):
        stop = min(rows, start + competitor_chunk_rows)
        chunk = np.asarray(matrix[start:stop], dtype=np.float64)
        delta = (winner_weight[None, :] - chunk) * hidden64[None, :]
        bias_diff = winner_bias - bias64[start:stop]
        full_margin = bias_diff + np.sum(delta, axis=1)
        local_winner = winner - start
        if 0 <= local_winner < stop - start:
            full_margin[local_winner] = math.inf
        invalid = full_margin <= 0.0
        domain_violations += int(np.count_nonzero(invalid))

        threshold = np.sum(np.abs(delta), axis=1) - bias_diff
        coordinate_gain = 2.0 * np.maximum(delta, 0.0)
        if padded_width != width:
            coordinate_gain = np.pad(
                coordinate_gain,
                ((0, 0), (0, padded_width - width)),
                mode="constant",
            )
        tile_gain = coordinate_gain.reshape(
            coordinate_gain.shape[0], len(widths), int(tile_columns)
        ).sum(axis=2)
        tile_gain.sort(axis=1)
        tile_gain = tile_gain[:, ::-1]
        cumulative = np.cumsum(tile_gain, axis=1)
        met = cumulative > threshold[:, None]
        local_counts = np.argmax(met, axis=1) + 1
        impossible = ~np.any(met, axis=1)
        local_counts[threshold < 0.0] = 0
        local_counts[impossible] = len(widths) + 1
        if 0 <= local_winner < stop - start:
            local_counts[local_winner] = 0

        local_max_index = int(np.argmax(local_counts))
        local_max = int(local_counts[local_max_index])
        if local_max > necessary_tiles:
            necessary_tiles = local_max
            hardest = start + local_max_index

    domain_violations += int(necessary_tiles > len(widths))
    if domain_violations:
        necessary_tiles = len(widths)

    necessary_columns = _minimum_columns_for_tile_count(
        widths, necessary_tiles
    )
    return OutputHeadDemandLowerBound(
        vocabulary_size=int(matrix.shape[0]),
        hidden_width=width,
        tile_columns=int(tile_columns),
        tile_count=len(widths),
        winner_index=winner,
        runner_up_index=runner_up,
        exact_real_margin=exact_margin,
        necessary_tile_count=necessary_tiles,
        necessary_column_count=necessary_columns,
        head_weight_fraction_lower_bound=necessary_columns / width,
        hardest_competitor_index=hardest,
        domain_winner_mismatch_count=mismatch + domain_violations,
        nonfinite_count=nonfinite_count,
    )


def subset_certifies_winner(
    weight: np.ndarray,
    hidden: np.ndarray,
    winner: int,
    revealed_tiles: Sequence[int],
    *,
    bias: np.ndarray | None = None,
    tile_columns: int = 1,
) -> bool:
    matrix = np.asarray(weight, dtype=np.float64)
    vector = np.asarray(hidden, dtype=np.float64)
    widths = _tile_widths(matrix.shape[1], tile_columns)
    mask = np.zeros(matrix.shape[1], dtype=bool)
    for tile in revealed_tiles:
        if tile < 0 or tile >= len(widths):
            raise ValueError("invalid tile index")
        start = tile * tile_columns
        mask[start : start + widths[tile]] = True
    delta = (matrix[winner][None, :] - matrix) * vector[None, :]
    bias64 = (
        np.zeros(matrix.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64)
    )
    known = bias64[winner] - bias64 + np.sum(delta[:, mask], axis=1)
    unread = np.sum(np.abs(delta[:, ~mask]), axis=1)
    certified = known > unread
    certified[winner] = True
    return bool(np.all(certified))


def exact_minimum_tile_count_bruteforce(
    weight: np.ndarray,
    hidden: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    tile_columns: int = 1,
) -> int:
    """Small-control oracle only; raises when the tile population is too large."""
    logits = exact_real_logits(weight, hidden, bias)
    winner = int(np.argmax(logits))
    tile_count = len(_tile_widths(np.asarray(weight).shape[1], tile_columns))
    if tile_count > 20:
        raise ValueError("bruteforce control is limited to 20 tiles")
    for count in range(tile_count + 1):
        for subset in combinations(range(tile_count), count):
            if subset_certifies_winner(
                weight,
                hidden,
                winner,
                subset,
                bias=bias,
                tile_columns=tile_columns,
            ):
                return count
    return tile_count
