from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from vortex_runtime.causal_residual_atlas_gate import (
    derive_first_decode_gate,
    required_successes,
    select_favorable_page,
)


def test_first_decode_gate_freezes_smallest_population_and_page_count() -> None:
    gate = derive_first_decode_gate()
    assert gate.rank == 16
    assert gate.page_columns == 64
    assert gate.layer_indices == (11,)
    assert gate.teacher_position_index == 1
    assert gate.pages_per_projection == (("q_proj", 16), ("down_proj", 56))
    assert gate.token_states == 18
    assert gate.projection_branches == 36
    assert gate.oracle_page_candidates == 1_296


def test_registered_coverage_allows_no_failure_in_frozen_population() -> None:
    gate = derive_first_decode_gate()
    assert gate.minimum_coverage == pytest.approx(
        Decimal("0.998998405303215166463")
    )
    assert gate.required_token_successes == 18
    assert gate.maximum_token_failures == 0
    assert gate.required_family_successes == 3
    assert gate.one_token_failure_coverage == Decimal(17) / Decimal(18)
    assert gate.one_token_failure_coverage < gate.minimum_coverage


def test_required_successes_validates_inputs_and_rounds_up() -> None:
    assert required_successes(10, Decimal("0.8")) == 8
    assert required_successes(10, Decimal("0.81")) == 9
    with pytest.raises(ValueError):
        required_successes(0, Decimal("0.5"))
    with pytest.raises(ValueError):
        required_successes(1, Decimal("1.1"))


def test_page_oracle_prefers_top1_preservation_before_kl() -> None:
    baseline = np.asarray([4.0, 3.0, 0.0])
    candidates = np.asarray(
        [
            [3.0, 3.1, 0.0],
            [4.0, 1.0, 0.0],
        ]
    )
    choice = select_favorable_page(baseline, candidates)
    assert choice.page_index == 1
    assert choice.top1_match
    assert choice.target_top1 == 0
    assert choice.candidate_top1 == 0


def test_page_oracle_breaks_equal_candidates_by_page_index() -> None:
    baseline = np.asarray([3.0, 1.0])
    candidates = np.asarray([[2.0, 1.0], [2.0, 1.0]])
    choice = select_favorable_page(
        baseline,
        candidates,
        page_indices=(9, 4),
    )
    assert choice.page_index == 4
    assert choice.top1_match


def test_page_oracle_reports_failure_when_no_page_preserves_top1() -> None:
    choice = select_favorable_page(
        np.asarray([3.0, 2.0, 1.0]),
        np.asarray([[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]]),
    )
    assert not choice.top1_match
    assert choice.candidate_top1 in {1, 2}


def test_page_oracle_rejects_malformed_or_nonfinite_rows() -> None:
    with pytest.raises(ValueError):
        select_favorable_page(np.asarray([1.0]), np.asarray([[1.0]]))
    with pytest.raises(ValueError):
        select_favorable_page(
            np.asarray([1.0, 0.0]),
            np.asarray([[1.0, np.nan]]),
        )
    with pytest.raises(ValueError):
        select_favorable_page(
            np.asarray([1.0, 0.0]),
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            page_indices=(0, 0),
        )
