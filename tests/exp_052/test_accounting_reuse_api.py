from __future__ import annotations

import pytest

from vortex_runtime.exact_advice import AdviceAccounting, ExactAdviceError


def test_amortized_target_fraction_charges_build_and_fallback() -> None:
    accounting = AdviceAccounting(
        query_count=100,
        advice_hits=90,
        target_fallback_calls=10,
        build_target_calls=100,
        advice_bytes=4096,
        lookup_probes=100,
    )
    assert accounting.amortized_target_forward_fraction(1) == pytest.approx(1.1)
    assert accounting.amortized_target_forward_fraction(100) == pytest.approx(0.11)


def test_perfect_one_build_per_state_reaches_one_over_85() -> None:
    accounting = AdviceAccounting(
        query_count=320,
        advice_hits=320,
        target_fallback_calls=0,
        build_target_calls=320,
        advice_bytes=1,
        lookup_probes=320,
    )
    assert accounting.amortized_target_forward_fraction(85) == pytest.approx(1 / 85)


def test_nonpositive_repetitions_fail_closed() -> None:
    accounting = AdviceAccounting(
        query_count=1,
        advice_hits=1,
        target_fallback_calls=0,
        build_target_calls=1,
        advice_bytes=1,
        lookup_probes=1,
    )
    with pytest.raises(ExactAdviceError):
        accounting.amortized_target_forward_fraction(0)
