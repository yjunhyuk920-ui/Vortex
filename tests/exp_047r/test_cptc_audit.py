from __future__ import annotations

import math
import random

import pytest

from vortex_runtime.cptc import CPTCConfig, CPTCError, certify_sum_sign, exact_reference
from vortex_runtime.cptc_audit import (
    certify_stratified_sum_sign,
    exact_state_range,
    global_symmetric_range,
    pair_margin_tile_contributions,
    quantile_strata,
    stratum_step_delta,
    tile_bounds_from_weight_span,
)


def test_pair_contributions_and_checkpoint_span_bounds_are_sound() -> None:
    hidden = [2.0, -1.0, 0.5, 3.0]
    top = [1.0, 2.0, -1.0, 0.25]
    competitor = [-1.0, 0.0, 1.0, -0.75]
    column_spans = [2.0, 3.0, 4.0, 1.0]
    contributions = pair_margin_tile_contributions(
        hidden, top, competitor, tile_size=2
    )
    bounds = tile_bounds_from_weight_span(hidden, column_spans, tile_size=2)
    assert contributions == pytest.approx((2.0, 2.0))
    assert all(abs(value) <= bound for value, bound in zip(contributions, bounds))


def test_quantile_strata_partition_every_index_once() -> None:
    strata = quantile_strata([3.0, 1.0, 4.0, 2.0, 8.0], 3)
    flattened = [index for stratum in strata for index in stratum]
    assert sorted(flattened) == list(range(5))
    assert len(flattened) == len(set(flattened))


def test_two_dimensional_alpha_spending_stays_within_budget() -> None:
    delta = 1e-5
    spent = sum(
        stratum_step_delta(delta, stratum, sample)
        for stratum in range(200)
        for sample in range(1, 2_000)
    )
    assert spent < delta
    assert spent > delta * 0.99


def test_stratified_certificate_matches_exact_reference() -> None:
    values = [1.0] * 96 + [-0.1] * 32
    bounds = [1.0] * len(values)
    result = certify_stratified_sum_sign(
        values,
        lower_bounds=[-bound for bound in bounds],
        upper_bounds=bounds,
        strata=quantile_strata(bounds, 8),
        config=CPTCConfig(delta=1e-6, max_sample_fraction=1.0, seed=11),
    )
    assert result.certified
    assert not result.fallback
    assert result.decision == exact_reference(values).decision
    assert result.total_tiles_evaluated < len(values)


def test_zero_margin_stratified_case_falls_back_exactly() -> None:
    values = [1.0, -1.0] * 64
    bounds = [1.0] * len(values)
    result = certify_stratified_sum_sign(
        values,
        lower_bounds=[-1.0] * len(values),
        upper_bounds=[1.0] * len(values),
        strata=quantile_strata(bounds, 8),
        config=CPTCConfig(delta=1e-10, max_sample_fraction=0.25, seed=3),
    )
    assert result.fallback
    assert not result.certified
    assert result.total_tiles_evaluated == len(values)
    assert result.exact_total_after_fallback == 0.0
    assert result.decision == 0


def test_misleading_stratified_prefix_does_not_wrongly_accept() -> None:
    population = 256
    config = CPTCConfig(delta=1e-12, max_sample_fraction=0.25, seed=29)
    strata = quantile_strata([1.0] * population, 8)
    values = [-1.0] * population
    # Reconstruct the implementation's per-stratum sample prefix and make it
    # mildly positive. A valid interval must remain open and exact fallback wins.
    for stratum_index, members in enumerate(strata):
        order = list(members)
        random.Random(config.seed + 1_000_003 * (stratum_index + 1)).shuffle(order)
        for index in order[: max(1, len(order) // 4)]:
            values[index] = 0.2
    result = certify_stratified_sum_sign(
        values,
        lower_bounds=[-1.0] * population,
        upper_bounds=[1.0] * population,
        strata=strata,
        config=config,
    )
    assert exact_reference(values).decision == -1
    assert result.fallback
    assert result.decision == -1


def test_oracle_range_never_wider_than_sound_global_range() -> None:
    values = [-0.7, 0.2, 0.1, 0.4, -0.1, 0.3] * 16
    sound_bounds = [1.0] * len(values)
    c0_low, c0_high = global_symmetric_range(sound_bounds)
    c1_low, c1_high = exact_state_range(values)
    assert c1_low >= c0_low
    assert c1_high <= c0_high

    config = CPTCConfig(delta=1e-7, max_sample_fraction=1.0, seed=17)
    c0 = certify_sum_sign(
        values, value_min=c0_low, value_max=c0_high, config=config
    )
    c1 = certify_sum_sign(
        values, value_min=c1_low, value_max=c1_high, config=config
    )
    assert c0.decision == c1.decision == exact_reference(values).decision
    assert c1.total_tiles_evaluated <= c0.total_tiles_evaluated


def test_deterministic_replay() -> None:
    values = [0.8] * 60 + [-0.4] * 36
    bounds = [1.0 + index / 1_000 for index in range(len(values))]
    kwargs = dict(
        lower_bounds=[-bound for bound in bounds],
        upper_bounds=bounds,
        strata=quantile_strata(bounds, 7),
        config=CPTCConfig(delta=1e-8, max_sample_fraction=1.0, seed=91),
    )
    assert certify_stratified_sum_sign(values, **kwargs) == certify_stratified_sum_sign(
        values, **kwargs
    )


def test_invalid_bounds_and_partitions_fail_closed() -> None:
    with pytest.raises(CPTCError):
        tile_bounds_from_weight_span([1.0], [-1.0], tile_size=1)
    with pytest.raises(CPTCError):
        certify_stratified_sum_sign(
            [2.0],
            lower_bounds=[-1.0],
            upper_bounds=[1.0],
            strata=[(0,)],
        )
    with pytest.raises(CPTCError):
        certify_stratified_sum_sign(
            [0.0, 0.0],
            lower_bounds=[-1.0, -1.0],
            upper_bounds=[1.0, 1.0],
            strata=[(0,), (0,)],
        )
    with pytest.raises(CPTCError):
        exact_state_range([math.nan])
