"""Causal Probabilistic Tile Certificate (CPTC).

This module is a Phase-B reference/optimized primitive.  It certifies the sign
of a finite sum while revealing tile contributions in a causal random order.
The certificate uses fixed-time Serfling bounds with an alpha-spending union
bound over all adaptive stopping times.  If the interval does not exclude zero,
the implementation evaluates every remaining tile and returns the exact sign.

It is not a complete Transformer runtime and contains no Phase-D measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


class CPTCError(ValueError):
    """Raised for invalid configuration or invalid numeric input."""


@dataclass(frozen=True)
class CPTCConfig:
    """Configuration for one finite-population sign certificate."""

    delta: float = 1e-6
    min_samples: int = 2
    max_sample_fraction: float = 0.25
    seed: int = 0

    def validate(self, population_size: int) -> None:
        if population_size <= 0:
            raise CPTCError("population_size must be positive")
        if not (0.0 < self.delta < 1.0):
            raise CPTCError("delta must be in (0, 1)")
        if self.min_samples <= 0:
            raise CPTCError("min_samples must be positive")
        if not (0.0 < self.max_sample_fraction <= 1.0):
            raise CPTCError("max_sample_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CPTCResult:
    """Result of a causal certificate or exact fallback."""

    decision: int
    certified: bool
    fallback: bool
    population_size: int
    sampled_before_decision: int
    total_tiles_evaluated: int
    sampled_sum: float
    estimated_total: float
    lower_bound: float
    upper_bound: float
    delta_spent_at_stop: float | None
    sampled_indices: tuple[int, ...]
    permutation_seed: int
    exact_total_after_fallback: float | None

    @property
    def sampled_fraction_before_decision(self) -> float:
        return self.sampled_before_decision / self.population_size

    @property
    def evaluated_fraction(self) -> float:
        return self.total_tiles_evaluated / self.population_size


@dataclass(frozen=True)
class ReferenceResult:
    total: float
    decision: int


def sign_decision(value: float, *, atol: float = 0.0) -> int:
    """Return -1, 0, or +1 under a declared absolute tie tolerance."""

    if not math.isfinite(value):
        raise CPTCError("decision value must be finite")
    if atol < 0 or not math.isfinite(atol):
        raise CPTCError("atol must be finite and non-negative")
    if value > atol:
        return 1
    if value < -atol:
        return -1
    return 0


def exact_reference(contributions: Iterable[float], *, base_margin: float = 0.0) -> ReferenceResult:
    """Independent full-sum reference used by Phase-B validation."""

    if not math.isfinite(base_margin):
        raise CPTCError("base_margin must be finite")
    total = base_margin
    count = 0
    for value in contributions:
        value = float(value)
        if not math.isfinite(value):
            raise CPTCError("contributions must be finite")
        total += value
        count += 1
    if count == 0:
        raise CPTCError("at least one contribution is required")
    return ReferenceResult(total=total, decision=sign_decision(total))


def alpha_spending(delta_total: float, sample_count: int) -> float:
    """Allocate failure probability across all possible stopping times.

    The sequence delta_n = delta_total * 6 / (pi^2 n^2) sums to at most
    delta_total, so a union bound makes fixed-n valid intervals anytime-valid.
    """

    if not (0.0 < delta_total < 1.0):
        raise CPTCError("delta_total must be in (0, 1)")
    if sample_count <= 0:
        raise CPTCError("sample_count must be positive")
    return delta_total * 6.0 / (math.pi * math.pi * sample_count * sample_count)


def serfling_total_radius(
    *,
    population_size: int,
    sample_count: int,
    value_min: float,
    value_max: float,
    delta_at_step: float,
) -> float:
    """Two-sided Serfling radius for a population total.

    For a simple random sample without replacement from N values in [a, b],
    this returns R such that |sum(population) - N * sample_mean| <= R with
    fixed-step probability at least 1 - delta_at_step.
    """

    if population_size <= 0:
        raise CPTCError("population_size must be positive")
    if not (1 <= sample_count <= population_size):
        raise CPTCError("sample_count must be in [1, population_size]")
    if not (math.isfinite(value_min) and math.isfinite(value_max)):
        raise CPTCError("bounds must be finite")
    if value_min > value_max:
        raise CPTCError("value_min cannot exceed value_max")
    if not (0.0 < delta_at_step < 1.0):
        raise CPTCError("delta_at_step must be in (0, 1)")
    if sample_count == population_size or value_min == value_max:
        return 0.0

    finite_population_correction = 1.0 - (sample_count - 1.0) / population_size
    log_term = math.log(2.0 / delta_at_step)
    mean_radius = (value_max - value_min) * math.sqrt(
        finite_population_correction * log_term / (2.0 * sample_count)
    )
    return population_size * mean_radius


def audit_declared_bounds(
    contributions: Sequence[float], *, value_min: float, value_max: float
) -> None:
    """Validation-only full audit of the declared range.

    A real optimized runtime must derive a sound range from checksummed metadata
    and the current activation.  Calling this helper consumes every value and is
    therefore forbidden from optimized-path accounting.
    """

    if not contributions:
        raise CPTCError("at least one contribution is required")
    if value_min > value_max:
        raise CPTCError("value_min cannot exceed value_max")
    for value in contributions:
        if not math.isfinite(float(value)):
            raise CPTCError("contributions must be finite")
        if value < value_min or value > value_max:
            raise CPTCError("declared bounds do not cover every contribution")


def certify_sum_sign(
    contributions: Sequence[float],
    *,
    value_min: float,
    value_max: float,
    base_margin: float = 0.0,
    config: CPTCConfig | None = None,
) -> CPTCResult:
    """Certify the sign of a finite sum or fall back to an exact full sum.

    Only sampled values are used before certification.  Validation code should
    compute `exact_reference` separately; this function deliberately does not
    compute the exact total after an early certificate.
    """

    cfg = config or CPTCConfig()
    population_size = len(contributions)
    cfg.validate(population_size)
    if not math.isfinite(base_margin):
        raise CPTCError("base_margin must be finite")
    if not (math.isfinite(value_min) and math.isfinite(value_max)):
        raise CPTCError("bounds must be finite")
    if value_min > value_max:
        raise CPTCError("value_min cannot exceed value_max")

    order = list(range(population_size))
    random.Random(cfg.seed).shuffle(order)
    limit = min(
        population_size,
        max(cfg.min_samples, math.ceil(population_size * cfg.max_sample_fraction)),
    )

    sampled_sum = 0.0
    sampled_indices: list[int] = []
    estimated_total = base_margin
    lower_bound = -math.inf
    upper_bound = math.inf
    delta_at_stop: float | None = None

    for sample_count, index in enumerate(order[:limit], start=1):
        value = float(contributions[index])
        if not math.isfinite(value):
            raise CPTCError("encountered non-finite tile contribution")
        if value < value_min or value > value_max:
            raise CPTCError("encountered contribution outside declared bounds")
        sampled_indices.append(index)
        sampled_sum += value

        if sample_count < cfg.min_samples:
            continue

        delta_at_step = alpha_spending(cfg.delta, sample_count)
        radius = serfling_total_radius(
            population_size=population_size,
            sample_count=sample_count,
            value_min=value_min,
            value_max=value_max,
            delta_at_step=delta_at_step,
        )
        estimated_total = base_margin + population_size * sampled_sum / sample_count
        lower_bound = estimated_total - radius
        upper_bound = estimated_total + radius

        if lower_bound > 0.0:
            delta_at_stop = delta_at_step
            return CPTCResult(
                decision=1,
                certified=True,
                fallback=False,
                population_size=population_size,
                sampled_before_decision=sample_count,
                total_tiles_evaluated=sample_count,
                sampled_sum=sampled_sum,
                estimated_total=estimated_total,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                delta_spent_at_stop=delta_at_stop,
                sampled_indices=tuple(sampled_indices),
                permutation_seed=cfg.seed,
                exact_total_after_fallback=None,
            )
        if upper_bound < 0.0:
            delta_at_stop = delta_at_step
            return CPTCResult(
                decision=-1,
                certified=True,
                fallback=False,
                population_size=population_size,
                sampled_before_decision=sample_count,
                total_tiles_evaluated=sample_count,
                sampled_sum=sampled_sum,
                estimated_total=estimated_total,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                delta_spent_at_stop=delta_at_stop,
                sampled_indices=tuple(sampled_indices),
                permutation_seed=cfg.seed,
                exact_total_after_fallback=None,
            )

    # Exact fallback: evaluate all unseen tiles and return the full reference sum.
    unseen_sum = 0.0
    sampled_set = set(sampled_indices)
    for index, raw_value in enumerate(contributions):
        value = float(raw_value)
        if not math.isfinite(value):
            raise CPTCError("encountered non-finite tile contribution during fallback")
        if value < value_min or value > value_max:
            raise CPTCError("encountered contribution outside declared bounds during fallback")
        if index not in sampled_set:
            unseen_sum += value
    exact_total = base_margin + sampled_sum + unseen_sum
    decision = sign_decision(exact_total)
    return CPTCResult(
        decision=decision,
        certified=False,
        fallback=True,
        population_size=population_size,
        sampled_before_decision=len(sampled_indices),
        total_tiles_evaluated=population_size,
        sampled_sum=sampled_sum,
        estimated_total=estimated_total,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        delta_spent_at_stop=None,
        sampled_indices=tuple(sampled_indices),
        permutation_seed=cfg.seed,
        exact_total_after_fallback=exact_total,
    )
