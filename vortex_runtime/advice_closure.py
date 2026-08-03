"""Constraint-closure equations for exact target-specific advice (EXP-052)."""

from __future__ import annotations

from dataclasses import dataclass
import math


class AdviceClosureError(ValueError):
    """Raised when an exact-advice closure equation is malformed."""


def _fraction(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AdviceClosureError(f"{name} must lie in [0,1]")
    return number


def fully_accounted_target_fraction(
    *, hit_rate: float, amortized_build_fraction: float
) -> float:
    """Return build streams/query plus exact-target fallback streams/query."""

    hit = _fraction(hit_rate, "hit_rate")
    build = float(amortized_build_fraction)
    if not math.isfinite(build) or build < 0.0:
        raise AdviceClosureError(
            "amortized_build_fraction must be finite and non-negative"
        )
    return build + (1.0 - hit)


def required_hit_rate_for_fraction(
    *, allowed_fraction: float, amortized_build_fraction: float
) -> float:
    """Minimum exact hit rate required by the charged target-stream equation."""

    allowed = _fraction(allowed_fraction, "allowed_fraction")
    if allowed <= 0.0:
        raise AdviceClosureError("allowed_fraction must be positive")
    build = float(amortized_build_fraction)
    if not math.isfinite(build) or build < 0.0:
        raise AdviceClosureError(
            "amortized_build_fraction must be finite and non-negative"
        )
    return min(1.0, max(0.0, 1.0 + build - allowed))


def required_exact_repetitions(
    *,
    query_count: int,
    build_target_calls: int,
    hit_rate: float,
    allowed_fraction: float,
) -> int | None:
    """Return repetitions needed to amortize build cost, or None if impossible.

    One repetition means querying the complete evaluation set once. The online
    exact-target fallback fraction is charged on every repetition.
    """

    if query_count <= 0 or build_target_calls < 0:
        raise AdviceClosureError("invalid query/build counts")
    hit = _fraction(hit_rate, "hit_rate")
    allowed = _fraction(allowed_fraction, "allowed_fraction")
    remaining = allowed - (1.0 - hit)
    if remaining <= 0.0:
        return None
    if build_target_calls == 0:
        return 1
    return math.ceil(build_target_calls / (query_count * remaining))


@dataclass(frozen=True)
class BudgetCoverageAudit:
    state_count: int
    entry_bytes: int
    budget_bytes: int
    maximum_entries: int
    maximum_coverage_fraction: float
    minimum_fallback_fraction: float


def budget_coverage_audit(
    *, state_count: int, entry_bytes: int, budget_bytes: int
) -> BudgetCoverageAudit:
    if state_count <= 0 or entry_bytes <= 0 or budget_bytes < 0:
        raise AdviceClosureError("invalid state/storage budget")
    entries = min(state_count, budget_bytes // entry_bytes)
    coverage = entries / state_count
    return BudgetCoverageAudit(
        state_count=state_count,
        entry_bytes=entry_bytes,
        budget_bytes=budget_bytes,
        maximum_entries=entries,
        maximum_coverage_fraction=coverage,
        minimum_fallback_fraction=1.0 - coverage,
    )


def hot_index_capacity(*, budget_bytes: int, slot_bytes: int) -> int:
    if budget_bytes < 0 or slot_bytes <= 0:
        raise AdviceClosureError("invalid hot-index budget")
    return budget_bytes // slot_bytes


@dataclass(frozen=True)
class AdviceClosureVerdict:
    hit_rate: float
    fallback_fraction: float
    amortized_build_fraction: float
    fully_accounted_fraction: float
    allowed_fraction: float
    passes: bool


def evaluate_advice_closure(
    *, hit_rate: float, amortized_build_fraction: float, allowed_fraction: float
) -> AdviceClosureVerdict:
    hit = _fraction(hit_rate, "hit_rate")
    allowed = _fraction(allowed_fraction, "allowed_fraction")
    if allowed <= 0.0:
        raise AdviceClosureError("allowed_fraction must be positive")
    total = fully_accounted_target_fraction(
        hit_rate=hit, amortized_build_fraction=amortized_build_fraction
    )
    return AdviceClosureVerdict(
        hit_rate=hit,
        fallback_fraction=1.0 - hit,
        amortized_build_fraction=float(amortized_build_fraction),
        fully_accounted_fraction=total,
        allowed_fraction=allowed,
        passes=total <= allowed,
    )
