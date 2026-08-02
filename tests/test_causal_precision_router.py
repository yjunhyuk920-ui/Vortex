from __future__ import annotations

import pytest

from vortex_runtime.causal_precision_router import (
    PrecisionStageObservation,
    select_stable_precision_stage,
)


def observation(stage: int, token: int, margin: float, fraction: float):
    return PrecisionStageObservation(
        stage=stage,
        token=token,
        margin=margin,
        cumulative_layer_fraction=fraction,
    )


def test_router_stops_at_first_stable_high_margin_stage() -> None:
    decision = select_stable_precision_stage(
        [
            observation(0, 10, 0.1, 0.0),
            observation(1, 11, 0.5, 0.08),
            observation(2, 11, 0.7, 0.16),
            observation(3, 12, 1.0, 0.28),
        ],
        margin_threshold=0.4,
    )
    assert decision.accepted
    assert decision.selected_stage == 2
    assert decision.selected_token == 11
    assert decision.cumulative_layer_fraction == 0.16


def test_router_rejects_stable_low_margin_token() -> None:
    decision = select_stable_precision_stage(
        [
            observation(0, 10, 0.1, 0.0),
            observation(1, 10, 0.2, 0.1),
            observation(2, 11, 0.3, 0.2),
        ],
        margin_threshold=0.4,
    )
    assert not decision.accepted
    assert decision.selected_token is None
    assert "fallback" in decision.reason


def test_router_requires_consecutive_monotonic_stages() -> None:
    with pytest.raises(ValueError):
        select_stable_precision_stage(
            [observation(0, 1, 1.0, 0.2), observation(2, 1, 1.0, 0.1)]
        )
