"""EXP-047R range-audit helpers.

These routines are analysis/reference machinery. They do not replace a real
Transformer operation and they do not constitute Phase-D evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from vortex_runtime.cptc import (
    CPTCConfig,
    CPTCError,
    alpha_spending,
    serfling_total_radius,
    sign_decision,
)


@dataclass(frozen=True)
class StratifiedCPTCResult:
    decision: int
    certified: bool
    fallback: bool
    population_size: int
    sampled_before_decision: int
    total_tiles_evaluated: int
    lower_bound: float
    upper_bound: float
    sampled_indices: tuple[int, ...]
    per_stratum_samples: tuple[int, ...]
    exact_total_after_fallback: float | None

    @property
    def evaluated_fraction(self) -> float:
        return self.total_tiles_evaluated / self.population_size


def _finite_sequence(values: Sequence[float], name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise CPTCError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in converted):
        raise CPTCError(f"{name} must contain only finite values")
    return converted


def exact_state_range(contributions: Sequence[float]) -> tuple[float, float]:
    """Return the non-deployable exact per-state range used by C1."""

    values = _finite_sequence(contributions, "contributions")
    return min(values), max(values)


def tile_bounds_from_weight_span(
    hidden: Sequence[float],
    per_dimension_weight_span: Sequence[float],
    *,
    tile_size: int,
) -> tuple[float, ...]:
    """Compute sound activation-conditioned pair-margin tile bounds.

    For output rows a and b, |W[a,j] - W[b,j]| is bounded by the checkpoint
    column span max_o W[o,j] - min_o W[o,j]. Therefore each tile contribution
    is bounded by sum_j |h_j| * span_j without reading the selected output rows.
    """

    hidden_values = _finite_sequence(hidden, "hidden")
    spans = _finite_sequence(per_dimension_weight_span, "per_dimension_weight_span")
    if len(hidden_values) != len(spans):
        raise CPTCError("hidden and per_dimension_weight_span must have equal length")
    if tile_size <= 0:
        raise CPTCError("tile_size must be positive")
    if any(span < 0.0 for span in spans):
        raise CPTCError("weight spans must be non-negative")

    bounds: list[float] = []
    for start in range(0, len(hidden_values), tile_size):
        stop = min(start + tile_size, len(hidden_values))
        bound = math.fsum(
            abs(hidden_values[index]) * spans[index] for index in range(start, stop)
        )
        if not math.isfinite(bound):
            raise CPTCError("computed tile bound is non-finite")
        bounds.append(bound)
    return tuple(bounds)


def pair_margin_tile_contributions(
    hidden: Sequence[float],
    top_weight: Sequence[float],
    competitor_weight: Sequence[float],
    *,
    tile_size: int,
) -> tuple[float, ...]:
    """Materialize exact pair-margin tile contributions for offline audit."""

    hidden_values = _finite_sequence(hidden, "hidden")
    top_values = _finite_sequence(top_weight, "top_weight")
    competitor_values = _finite_sequence(competitor_weight, "competitor_weight")
    if not (len(hidden_values) == len(top_values) == len(competitor_values)):
        raise CPTCError("hidden and weight rows must have equal length")
    if tile_size <= 0:
        raise CPTCError("tile_size must be positive")

    contributions: list[float] = []
    for start in range(0, len(hidden_values), tile_size):
        stop = min(start + tile_size, len(hidden_values))
        value = math.fsum(
            (top_values[index] - competitor_values[index]) * hidden_values[index]
            for index in range(start, stop)
        )
        if not math.isfinite(value):
            raise CPTCError("computed tile contribution is non-finite")
        contributions.append(value)
    return tuple(contributions)


def global_symmetric_range(tile_bounds: Sequence[float]) -> tuple[float, float]:
    bounds = _finite_sequence(tile_bounds, "tile_bounds")
    if any(bound < 0.0 for bound in bounds):
        raise CPTCError("tile bounds must be non-negative")
    maximum = max(bounds)
    return -maximum, maximum


def quantile_strata(
    tile_bounds: Sequence[float], stratum_count: int
) -> tuple[tuple[int, ...], ...]:
    """Partition indices by bound magnitude with deterministic balanced buckets."""

    bounds = _finite_sequence(tile_bounds, "tile_bounds")
    if any(bound < 0.0 for bound in bounds):
        raise CPTCError("tile bounds must be non-negative")
    if stratum_count <= 0:
        raise CPTCError("stratum_count must be positive")
    count = min(stratum_count, len(bounds))
    ranked = sorted(range(len(bounds)), key=lambda index: (bounds[index], index))
    strata: list[tuple[int, ...]] = []
    for bucket in range(count):
        start = bucket * len(ranked) // count
        stop = (bucket + 1) * len(ranked) // count
        members = tuple(ranked[start:stop])
        if members:
            strata.append(members)
    return tuple(strata)


def stratum_step_delta(delta_total: float, stratum_index: int, sample_count: int) -> float:
    """Union-account delta across both strata and adaptive sample counts."""

    if stratum_index < 0:
        raise CPTCError("stratum_index must be non-negative")
    stratum_budget = alpha_spending(delta_total, stratum_index + 1)
    return alpha_spending(stratum_budget, sample_count)


def _validate_strata(
    strata: Sequence[Sequence[int]], population_size: int
) -> tuple[tuple[int, ...], ...]:
    if not strata:
        raise CPTCError("at least one stratum is required")
    normalized = tuple(tuple(int(index) for index in stratum) for stratum in strata)
    flat = [index for stratum in normalized for index in stratum]
    if any(not stratum for stratum in normalized):
        raise CPTCError("strata must not contain empty groups")
    if sorted(flat) != list(range(population_size)):
        raise CPTCError("strata must partition every population index exactly once")
    return normalized


def certify_stratified_sum_sign(
    contributions: Sequence[float],
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    strata: Sequence[Sequence[int]],
    base_margin: float = 0.0,
    config: CPTCConfig | None = None,
) -> StratifiedCPTCResult:
    """Certify a finite sum using range-stratified Serfling intervals.

    This reference implementation validates every declared bound because it is
    an offline audit. A deployable runtime must instead trust checksummed bound
    metadata and reject/fallback on metadata failure.
    """

    values = _finite_sequence(contributions, "contributions")
    lows = _finite_sequence(lower_bounds, "lower_bounds")
    highs = _finite_sequence(upper_bounds, "upper_bounds")
    if len(lows) != len(values) or len(highs) != len(values):
        raise CPTCError("bounds and contributions must have equal length")
    if not math.isfinite(base_margin):
        raise CPTCError("base_margin must be finite")
    for value, low, high in zip(values, lows, highs):
        if low > high:
            raise CPTCError("lower bound cannot exceed upper bound")
        if value < low or value > high:
            raise CPTCError("declared bounds do not cover every contribution")

    cfg = config or CPTCConfig(max_sample_fraction=1.0)
    cfg.validate(len(values))
    normalized = _validate_strata(strata, len(values))

    limit = min(
        len(values),
        max(cfg.min_samples, math.ceil(len(values) * cfg.max_sample_fraction)),
    )
    orders: list[list[int]] = []
    for stratum_index, members in enumerate(normalized):
        order = list(members)
        random.Random(cfg.seed + 1_000_003 * (stratum_index + 1)).shuffle(order)
        orders.append(order)

    positions = [0] * len(normalized)
    sample_sums = [0.0] * len(normalized)
    sampled_indices: list[int] = []
    total_sampled = 0
    lower_total = -math.inf
    upper_total = math.inf

    while total_sampled < limit:
        candidates: list[tuple[float, int]] = []
        for stratum_index, members in enumerate(normalized):
            remaining = len(members) - positions[stratum_index]
            if remaining <= 0:
                continue
            stratum_low = min(lows[index] for index in members)
            stratum_high = max(highs[index] for index in members)
            priority = remaining * (stratum_high - stratum_low)
            candidates.append((priority, -stratum_index))
        if not candidates:
            break
        _, negative_index = max(candidates)
        stratum_index = -negative_index
        index = orders[stratum_index][positions[stratum_index]]
        positions[stratum_index] += 1
        sample_sums[stratum_index] += values[index]
        sampled_indices.append(index)
        total_sampled += 1

        lower_total = base_margin
        upper_total = base_margin
        for current_index, members in enumerate(normalized):
            size = len(members)
            sampled = positions[current_index]
            stratum_low = min(lows[index] for index in members)
            stratum_high = max(highs[index] for index in members)
            deterministic_low = size * stratum_low
            deterministic_high = size * stratum_high
            if sampled == 0:
                lower_total += deterministic_low
                upper_total += deterministic_high
                continue
            if sampled == size:
                exact = sample_sums[current_index]
                lower_total += exact
                upper_total += exact
                continue
            delta_at_step = stratum_step_delta(cfg.delta, current_index, sampled)
            radius = serfling_total_radius(
                population_size=size,
                sample_count=sampled,
                value_min=stratum_low,
                value_max=stratum_high,
                delta_at_step=delta_at_step,
            )
            estimate = size * sample_sums[current_index] / sampled
            lower_total += max(deterministic_low, estimate - radius)
            upper_total += min(deterministic_high, estimate + radius)

        if total_sampled < len(values):
            if lower_total > 0.0:
                return StratifiedCPTCResult(
                    decision=1,
                    certified=True,
                    fallback=False,
                    population_size=len(values),
                    sampled_before_decision=total_sampled,
                    total_tiles_evaluated=total_sampled,
                    lower_bound=lower_total,
                    upper_bound=upper_total,
                    sampled_indices=tuple(sampled_indices),
                    per_stratum_samples=tuple(positions),
                    exact_total_after_fallback=None,
                )
            if upper_total < 0.0:
                return StratifiedCPTCResult(
                    decision=-1,
                    certified=True,
                    fallback=False,
                    population_size=len(values),
                    sampled_before_decision=total_sampled,
                    total_tiles_evaluated=total_sampled,
                    lower_bound=lower_total,
                    upper_bound=upper_total,
                    sampled_indices=tuple(sampled_indices),
                    per_stratum_samples=tuple(positions),
                    exact_total_after_fallback=None,
                )

    exact_total = base_margin + math.fsum(values)
    return StratifiedCPTCResult(
        decision=sign_decision(exact_total),
        certified=False,
        fallback=True,
        population_size=len(values),
        sampled_before_decision=len(sampled_indices),
        total_tiles_evaluated=len(values),
        lower_bound=lower_total,
        upper_bound=upper_total,
        sampled_indices=tuple(sampled_indices),
        per_stratum_samples=tuple(positions),
        exact_total_after_fallback=exact_total,
    )
