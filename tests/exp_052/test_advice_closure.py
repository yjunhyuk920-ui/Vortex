from __future__ import annotations

import pytest

from vortex_runtime.advice_closure import (
    AdviceClosureError,
    budget_coverage_audit,
    evaluate_advice_closure,
    fully_accounted_target_fraction,
    hot_index_capacity,
    required_exact_repetitions,
    required_hit_rate_for_fraction,
)


ALLOWED = 0.011851851851851851


def test_infinite_reuse_still_requires_98_8148_percent_hits() -> None:
    required = required_hit_rate_for_fraction(
        allowed_fraction=ALLOWED,
        amortized_build_fraction=0.0,
    )
    assert required == pytest.approx(1.0 - ALLOWED)


def test_85_reuses_require_nearly_perfect_coverage() -> None:
    required = required_hit_rate_for_fraction(
        allowed_fraction=ALLOWED,
        amortized_build_fraction=1.0 / 85.0,
    )
    assert required == pytest.approx(0.9999128540305011)
    assert fully_accounted_target_fraction(
        hit_rate=1.0,
        amortized_build_fraction=1.0 / 85.0,
    ) == pytest.approx(1.0 / 85.0)


def test_repetition_solver_fails_when_fallback_floor_exceeds_budget() -> None:
    assert (
        required_exact_repetitions(
            query_count=64,
            build_target_calls=320,
            hit_rate=0.0,
            allowed_fraction=ALLOWED,
        )
        is None
    )


def test_repetition_solver_returns_85_for_one_build_per_exact_state() -> None:
    repetitions = required_exact_repetitions(
        query_count=320,
        build_target_calls=320,
        hit_rate=1.0,
        allowed_fraction=ALLOWED,
    )
    assert repetitions == 85


def test_one_tib_cannot_cover_large_independent_state_family() -> None:
    audit = budget_coverage_audit(
        state_count=2**48,
        entry_bytes=128,
        budget_bytes=2**40,
    )
    assert audit.maximum_entries == 2**33
    assert audit.maximum_coverage_fraction == pytest.approx(2**-15)
    assert audit.minimum_fallback_fraction > ALLOWED


def test_hot_index_capacity_is_explicit() -> None:
    assert hot_index_capacity(budget_bytes=8 * 2**30, slot_bytes=48) == (
        8 * 2**30
    ) // 48


def test_closure_verdict() -> None:
    passing = evaluate_advice_closure(
        hit_rate=1.0,
        amortized_build_fraction=1.0 / 85.0,
        allowed_fraction=ALLOWED,
    )
    failing = evaluate_advice_closure(
        hit_rate=0.99,
        amortized_build_fraction=0.01,
        allowed_fraction=ALLOWED,
    )
    assert passing.passes
    assert not failing.passes


def test_invalid_equations_fail_closed() -> None:
    with pytest.raises(AdviceClosureError):
        required_hit_rate_for_fraction(
            allowed_fraction=0.0,
            amortized_build_fraction=0.0,
        )
    with pytest.raises(AdviceClosureError):
        budget_coverage_audit(state_count=0, entry_bytes=1, budget_bytes=1)
    with pytest.raises(AdviceClosureError):
        hot_index_capacity(budget_bytes=1, slot_bytes=0)
