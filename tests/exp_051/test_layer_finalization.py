from __future__ import annotations

import math

import pytest

from vortex_runtime.layer_finalization import (
    LateDecisionResidualChain,
    LayerFinalizationError,
    LayerTraffic,
    analyze_layer_probe,
    first_match_depth,
    fixed_depths,
    post_first_match_wrong_depths,
    suffix_stable_depth,
    token_changes,
)


def traffic() -> LayerTraffic:
    return LayerTraffic(
        block_parameter_bytes=(10, 20, 30, 40),
        embedding_row_bytes=2,
        final_norm_bytes=3,
        lm_head_bytes=5,
    )


def test_first_match_and_suffix_stable_are_distinct() -> None:
    tokens = (7, 3, 7, 4, 7)
    assert first_match_depth(tokens) == 0
    assert suffix_stable_depth(tokens) == 4
    # After the first depth-0 match, only depths 1 and 3 differ from the
    # full-depth token. The final depth is correct by definition.
    assert post_first_match_wrong_depths(tokens) == 2
    assert token_changes(tokens) == 4


def test_suffix_stable_can_begin_at_depth_zero() -> None:
    tokens = (5, 5, 5, 5, 5)
    assert first_match_depth(tokens) == 0
    assert suffix_stable_depth(tokens) == 0
    assert token_changes(tokens) == 0


def test_layer_traffic_full_depth_is_one() -> None:
    value = traffic()
    assert value.full_logical_bytes == 110
    assert value.bytes_at_depth(0) == 10
    assert value.bytes_at_depth(2) == 40
    assert value.fraction_at_depth(4) == pytest.approx(1.0)
    assert value.lm_head_fraction == pytest.approx(5 / 110)


def test_analysis_reports_favorable_stable_fraction() -> None:
    result = analyze_layer_probe(
        intermediate_tokens=(9, 4, 9, 9, 9),
        margins=(1.0, 0.1, 0.2, 0.4, 0.8),
        traffic=traffic(),
    )
    assert result.first_match_depth == 0
    assert result.suffix_stable_depth == 2
    assert result.first_match_block_fraction == 0.0
    assert result.suffix_stable_block_fraction == 0.5
    assert result.suffix_stable_logical_byte_fraction == pytest.approx(40 / 110)
    assert result.post_first_match_wrong_depths == 1
    assert result.final_token == 9


def test_fixed_depths_are_pre_registered_fractions() -> None:
    assert fixed_depths(8) == (0, 1, 2, 4, 6, 8)
    assert fixed_depths(7) == (0, 1, 2, 4, 6, 7)


def test_late_decision_chain_flips_only_at_final_layer() -> None:
    chain = LateDecisionResidualChain(block_count=8)
    tokens, margins, states = chain.probe()
    assert len(tokens) == 9
    assert tokens[:-1] == (0,) * 8
    assert tokens[-1] == 1
    assert all(value > 0 for value in margins)
    assert states[0] == (2.0, 0.0)
    assert states[-1] == (-2.0, 3.0)
    assert first_match_depth(tokens) == 8
    assert suffix_stable_depth(tokens) == 8


def test_late_decision_analysis_requires_full_traffic() -> None:
    chain = LateDecisionResidualChain(block_count=4)
    tokens, margins, _ = chain.probe()
    result = analyze_layer_probe(
        intermediate_tokens=tokens,
        margins=margins,
        traffic=traffic(),
    )
    assert result.suffix_stable_depth == 4
    assert result.suffix_stable_block_fraction == 1.0
    assert result.suffix_stable_logical_byte_fraction == 1.0


def test_invalid_probe_contracts_fail_closed() -> None:
    with pytest.raises(LayerFinalizationError):
        analyze_layer_probe(
            intermediate_tokens=(1, 2),
            margins=(1.0, 2.0),
            traffic=traffic(),
        )
    with pytest.raises(LayerFinalizationError):
        analyze_layer_probe(
            intermediate_tokens=(1, 1, 1, 1, 1),
            margins=(1.0, 1.0, math.nan, 1.0, 1.0),
            traffic=traffic(),
        )
    with pytest.raises(LayerFinalizationError):
        fixed_depths(0)


def test_invalid_traffic_and_chain_fail_closed() -> None:
    with pytest.raises(LayerFinalizationError):
        LayerTraffic(
            block_parameter_bytes=(10, 0),
            embedding_row_bytes=1,
            final_norm_bytes=1,
            lm_head_bytes=1,
        ).validate()
    with pytest.raises(LayerFinalizationError):
        LateDecisionResidualChain(block_count=0).probe()
    with pytest.raises(LayerFinalizationError):
        LateDecisionResidualChain(block_count=2, early_token=1, final_token=1).probe()
