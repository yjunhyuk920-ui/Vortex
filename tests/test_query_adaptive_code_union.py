from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from vortex_runtime.causal_bilinear_query_restriction import (
    CHECKPOINT_SERVICE_TOKENS,
    P50_TARGET_FRACTION,
)
from vortex_runtime.query_adaptive_code_union import (
    MAXIMUM_LEAF_DIMENSION,
    REJECT_DECISION,
    build_amortized_fraction,
    calibration_union_contains_only_calibration_span,
    derive_frontier,
    exp084_full_build_span_diagnostic,
    explicit_union_state_bytes,
    independent_query_hit_upper_bound,
    independent_stream_fraction,
    maximum_total_cached_directions,
    minimum_required_full_fallback_coverage,
    online_independent_stream_lower_bound,
    packed_leaf_dimensions,
    query_fraction_without_build,
)


ROOT = Path(__file__).resolve().parents[1]


def _span(basis: tuple[int, ...]) -> set[int]:
    words = {0}
    for row in basis:
        words |= {word ^ row for word in tuple(words)}
    return words


def test_union_is_strictly_more_expressive_than_one_leaf_but_pays_dimensions() -> None:
    first = _span((0b0001,))
    second = _span((0b0010,))
    assert 0b0001 in first and 0b0010 not in first
    assert 0b0010 in second and 0b0001 not in second
    assert independent_query_hit_upper_bound(2, 2) == 2
    assert independent_query_hit_upper_bound(1, 2) == 1


def test_every_trace_built_leaf_remains_inside_the_global_calibration_span() -> None:
    calibration = _span((0b0011, 0b1100))
    leaves = (_span((0b0011,)), _span((0b1100,)))
    assert calibration_union_contains_only_calibration_span(calibration, leaves)
    assert 0b0101 not in calibration
    assert all(0b0101 not in leaf for leaf in leaves)


def test_registered_query_leaf_dimension_and_build_ceiling_are_exact() -> None:
    assert MAXIMUM_LEAF_DIMENSION == 23
    assert maximum_total_cached_directions(1) == 87_958
    assert maximum_total_cached_directions(23) == 2_600
    query = query_fraction_without_build(23, resource="traffic")
    assert query + build_amortized_fraction(2_600) <= P50_TARGET_FRACTION
    assert query + build_amortized_fraction(2_601) > P50_TARGET_FRACTION


def test_packed_leaf_layout_and_persistent_state_charge_every_direction() -> None:
    leaves = packed_leaf_dimensions(2_600, 23)
    assert len(leaves) == 114
    assert leaves.count(23) == 113
    assert leaves[-1] == 1
    assert sum(leaves) == 2_600
    assert explicit_union_state_bytes(leaves) == 204_096_339_160


def test_build_ceiling_eliminates_full_dense_fallback_headroom() -> None:
    required = minimum_required_full_fallback_coverage(1, 87_958)
    assert required == Fraction(35_485_652_494_216_457, 35_485_655_040_000_000)
    assert float(required) > 0.9999999


def test_independent_stream_cannot_amortize_new_exact_answer_directions() -> None:
    total = maximum_total_cached_directions(1)
    coverage = Fraction(total, CHECKPOINT_SERVICE_TOKENS)
    assert coverage == Fraction(43_979, 10_000_000)
    traffic = independent_stream_fraction(
        1, total, resource="traffic"
    )
    assert traffic > 1
    assert traffic / P50_TARGET_FRACTION > 85
    assert online_independent_stream_lower_bound(1, resource="traffic") > 1


def test_registered_table_selects_dimension_one_only_as_a_favorable_ceiling() -> None:
    result = derive_frontier()
    ceiling = result["strongest_direction_count_ceiling"]
    assert ceiling["leaf_dimension"] == 1
    assert ceiling["maximum_total_cached_directions"] == 87_958
    assert ceiling["explicit_union_state_tib"] > 6.25
    assert result["decision"] == REJECT_DECISION
    assert (
        result["claim_boundary"]["implicit_nonlinear_checkpoint_data_structure"]
        == "NOT RULED OUT"
    )
    assert result["claim_boundary"]["core_candidate"] == "NONE"


def test_frozen_exp084_full_build_span_replay_rejects_every_partition() -> None:
    diagnostic = exp084_full_build_span_diagnostic(ROOT / "results" / "exp_084a")
    assert diagnostic["model_forward_calls"] == 0
    assert diagnostic["build_rows"] == 24
    assert diagnostic["evaluation_rows"] == 5
    assert diagnostic["build_modular_ranks"] == {
        "65497": 24,
        "65519": 24,
        "65521": 24,
    }
    assert diagnostic["full_build_union_exact_hits"] == 0
    assert diagnostic["full_build_union_certified_misses"] == 5
    for row in diagnostic["rows"]:
        assert set(row["membership_ranks"].values()) == {25}
        assert all(row["rank_increases"].values())
        assert row["certified_outside_full_build_span"]


def test_invalid_union_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        maximum_total_cached_directions(0)
    with pytest.raises(ValueError):
        packed_leaf_dimensions(1, 0)
    with pytest.raises(ValueError):
        explicit_union_state_bytes((1, 0))
    with pytest.raises(ValueError):
        query_fraction_without_build(1, resource="latency")
