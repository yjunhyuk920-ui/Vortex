from __future__ import annotations

import math
import random

import pytest

from vortex_runtime.cptc import (
    CPTCConfig,
    CPTCError,
    alpha_spending,
    audit_declared_bounds,
    certify_sum_sign,
    exact_reference,
    serfling_total_radius,
)


def test_alpha_spending_is_within_total_budget() -> None:
    delta = 1e-5
    spent = sum(alpha_spending(delta, n) for n in range(1, 100_000))
    assert spent < delta
    assert spent > delta * 0.999


def test_serfling_radius_collapses_for_full_population() -> None:
    assert (
        serfling_total_radius(
            population_size=32,
            sample_count=32,
            value_min=-1.0,
            value_max=1.0,
            delta_at_step=1e-8,
        )
        == 0.0
    )


def test_positive_cancellation_control_certifies_under_quarter_tiles() -> None:
    values = [1.0] * 384 + [-0.2] * 128
    audit_declared_bounds(values, value_min=-1.0, value_max=1.0)
    reference = exact_reference(values)
    result = certify_sum_sign(
        values,
        value_min=-1.0,
        value_max=1.0,
        config=CPTCConfig(delta=1e-6, max_sample_fraction=0.25, seed=17),
    )
    assert reference.decision == 1
    assert result.certified
    assert not result.fallback
    assert result.decision == reference.decision
    assert result.sampled_fraction_before_decision <= 0.25
    assert result.total_tiles_evaluated == result.sampled_before_decision


def test_negative_control_certifies() -> None:
    values = [-value for value in ([1.0] * 384 + [-0.2] * 128)]
    reference = exact_reference(values)
    result = certify_sum_sign(
        values,
        value_min=-1.0,
        value_max=1.0,
        config=CPTCConfig(delta=1e-6, max_sample_fraction=0.25, seed=17),
    )
    assert reference.decision == -1
    assert result.certified
    assert result.decision == reference.decision


def test_zero_margin_adversary_falls_back_exactly() -> None:
    values = [1.0, -1.0] * 128
    reference = exact_reference(values)
    result = certify_sum_sign(
        values,
        value_min=-1.0,
        value_max=1.0,
        config=CPTCConfig(delta=1e-8, max_sample_fraction=0.25, seed=3),
    )
    assert reference.total == 0.0
    assert reference.decision == 0
    assert result.fallback
    assert not result.certified
    assert result.total_tiles_evaluated == len(values)
    assert result.exact_total_after_fallback == reference.total
    assert result.decision == reference.decision


def test_misleading_sample_prefix_does_not_silently_commit() -> None:
    population_size = 512
    config = CPTCConfig(delta=1e-10, max_sample_fraction=0.25, seed=41)
    order = list(range(population_size))
    random.Random(config.seed).shuffle(order)
    values = [-1.0] * population_size
    # Make the entire optimized sample prefix mildly positive.  The declared
    # range remains wide enough that a valid certificate must not close.
    for index in order[: math.ceil(population_size * config.max_sample_fraction)]:
        values[index] = 0.2
    reference = exact_reference(values)
    result = certify_sum_sign(
        values,
        value_min=-1.0,
        value_max=1.0,
        config=config,
    )
    assert reference.decision == -1
    assert result.fallback
    assert result.decision == reference.decision
    assert result.total_tiles_evaluated == population_size


def test_randomized_property_reference_or_certified_agreement() -> None:
    generator = random.Random(12345)
    certified_count = 0
    for case_index in range(200):
        population_size = generator.choice([32, 64, 128, 256])
        values = [generator.uniform(-1.0, 1.0) for _ in range(population_size)]
        base_margin = generator.uniform(-5.0, 5.0)
        reference = exact_reference(values, base_margin=base_margin)
        result = certify_sum_sign(
            values,
            value_min=-1.0,
            value_max=1.0,
            base_margin=base_margin,
            config=CPTCConfig(
                delta=1e-12,
                max_sample_fraction=generator.choice([0.125, 0.25, 0.5]),
                seed=case_index,
            ),
        )
        assert result.decision == reference.decision
        if result.certified:
            certified_count += 1
            assert result.lower_bound > 0.0 or result.upper_bound < 0.0
        else:
            assert result.fallback
            assert result.exact_total_after_fallback == pytest.approx(reference.total)
            assert result.total_tiles_evaluated == population_size
    assert certified_count >= 0  # Explicitly reportable without requiring lucky cases.


def test_fixed_seed_is_deterministic() -> None:
    values = [0.8] * 90 + [-0.4] * 38
    config = CPTCConfig(delta=1e-7, max_sample_fraction=0.5, seed=99)
    first = certify_sum_sign(values, value_min=-1.0, value_max=1.0, config=config)
    second = certify_sum_sign(values, value_min=-1.0, value_max=1.0, config=config)
    assert first == second


def test_declared_bound_audit_rejects_invalid_metadata() -> None:
    with pytest.raises(CPTCError):
        audit_declared_bounds([0.0, 2.0], value_min=-1.0, value_max=1.0)


def test_numerical_and_configuration_faults_are_rejected() -> None:
    with pytest.raises(CPTCError):
        exact_reference([1.0, float("nan")])
    with pytest.raises(CPTCError):
        certify_sum_sign(
            [1.0, 2.0],
            value_min=-2.0,
            value_max=2.0,
            config=CPTCConfig(delta=0.0),
        )
    with pytest.raises(CPTCError):
        certify_sum_sign([1.0], value_min=2.0, value_max=1.0)
    with pytest.raises(CPTCError):
        certify_sum_sign([float("inf")], value_min=-1.0, value_max=1.0)
