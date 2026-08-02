from __future__ import annotations

from vortex_runtime.feasibility import default_specs
from vortex_runtime.precision_consensus import (
    PrecisionConsensusRow,
    analyze_precision_consensus,
    progressive_refinement_budget,
)


def test_consensus_rule_flags_disagreement_and_low_margin() -> None:
    rows = [
        PrecisionConsensusRow(0, 10, 10, 10, 1.0),
        PrecisionConsensusRow(1, 11, 12, 11, 0.9),
        PrecisionConsensusRow(2, 13, 12, 12, 0.2),
        PrecisionConsensusRow(3, 14, 14, 14, 0.3),
    ]
    report = analyze_precision_consensus(rows, margin_threshold=0.4)
    assert report.tokens == 4
    assert report.accepted_tokens == 1
    assert report.accepted_exact_tokens == 1
    assert report.refinement_tokens == 3
    assert report.q4_errors_flagged == 2
    assert report.q6_errors_flagged == 1
    assert report.all_exact_errors_flagged


def test_wrong_high_margin_consensus_is_not_claimed_safe() -> None:
    rows = [PrecisionConsensusRow(0, 10, 11, 11, 2.0)]
    report = analyze_precision_consensus(rows, margin_threshold=0.4)
    assert report.accepted_tokens == 1
    assert report.accepted_error_tokens == 1
    assert not report.all_exact_errors_flagged


def test_layer_local_refinement_can_fit_ideal_envelope() -> None:
    target, baseline = default_specs()
    point = progressive_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        refinement_fraction=0.25,
        refined_layer_fraction=0.25,
    )
    assert point.consensus_weight_gib > 280
    assert point.residual_weight_gib > 20
    assert point.ideal_pass
    assert point.required_overlap_fraction <= 1.0
    assert point.maximum_refinement_fraction_at_layer_fraction > 0.25


def test_global_q8_refinement_of_quarter_tokens_misses_compute_gate() -> None:
    target, baseline = default_specs()
    point = progressive_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        refinement_fraction=0.25,
        refined_layer_fraction=1.0,
    )
    assert not point.ideal_pass
    assert point.maximum_refinement_fraction_at_layer_fraction < 0.25
