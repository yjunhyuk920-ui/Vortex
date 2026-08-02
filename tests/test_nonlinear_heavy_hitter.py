from __future__ import annotations

from vortex_runtime.nonlinear_heavy_hitter import (
    LayerDamagePoint,
    normalize_damage_curves,
    solve_nonlinear_allocation,
)


def _point(count: int, damage: float) -> LayerDamagePoint:
    return LayerDamagePoint(
        selected_neurons=count,
        damage=damage,
        top1_rate=0.0,
        top32_rate=0.0,
        output_error=damage,
    )


def test_dynamic_program_prefers_sensitive_layer() -> None:
    curves = [
        [_point(1, 10.0), _point(2, 2.0), _point(4, 0.5)],
        [_point(1, 3.0), _point(2, 2.5), _point(4, 2.0)],
    ]
    allocation = solve_nonlinear_allocation(curves, total_budget=5)
    assert allocation.layer_counts == (4, 1)
    assert allocation.used_neurons == 5
    assert abs(allocation.predicted_total_damage - 3.5) < 1e-9


def test_allocator_can_leave_budget_unused_when_damage_is_equal() -> None:
    curves = [
        [_point(1, 1.0), _point(4, 1.0)],
        [_point(1, 2.0), _point(4, 2.0)],
    ]
    allocation = solve_nonlinear_allocation(curves, total_budget=8)
    # Tie-breaking prefers the larger measured allocation, demonstrating that
    # unused budget is not mandatory when the lower envelope is flat.
    assert allocation.used_neurons == 8
    assert allocation.layer_counts == (4, 4)


def test_normalization_builds_monotone_lower_envelope() -> None:
    curves = [[_point(1, 3.0), _point(2, 4.0), _point(4, 1.0)]]
    normalized = normalize_damage_curves(curves)[0]
    assert [point.damage for point in normalized] == [1.0, 1.0, 1.0]


def test_infeasible_budget_is_rejected() -> None:
    curves = [
        [_point(2, 1.0), _point(4, 0.5)],
        [_point(2, 1.0), _point(4, 0.5)],
    ]
    try:
        solve_nonlinear_allocation(curves, total_budget=3)
    except ValueError as error:
        assert "minimum" in str(error)
    else:
        raise AssertionError("expected infeasible allocation to fail")
