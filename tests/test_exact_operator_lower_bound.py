from __future__ import annotations

import math

import pytest

from vortex_runtime.exact_operator_lower_bound import (
    construct_skipped_coordinate_adversary,
    exact_operator_information_budget,
    exhaustive_single_skip_adversaries,
)


TARGET_PARAMETERS = 405_849_243_648


def test_q4_405b_information_and_compute_constants() -> None:
    budget = exact_operator_information_budget(
        parameter_count=TARGET_PARAMETERS,
        bits_per_parameter=4,
        resident_gib=8.0,
    )
    assert budget.exact_information_bits == 1_623_396_974_592
    assert math.isclose(
        budget.exact_information_gib,
        188.98828125,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        budget.minimum_external_information_gib,
        180.98828125,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        budget.dense_compute_gflop,
        811.698487296,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        budget.baseline_dense_compute_gflop,
        8.0,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        budget.compute_ratio_to_baseline,
        101.462310912,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert budget.exact_information_exceeds_resident
    assert budget.resident_fraction < 0.05


def test_information_bound_scales_with_precision() -> None:
    q4 = exact_operator_information_budget(
        parameter_count=TARGET_PARAMETERS,
        bits_per_parameter=4,
    )
    q8 = exact_operator_information_budget(
        parameter_count=TARGET_PARAMETERS,
        bits_per_parameter=8,
    )
    fp16 = exact_operator_information_budget(
        parameter_count=TARGET_PARAMETERS,
        bits_per_parameter=16,
    )
    assert math.isclose(q8.exact_information_gib, 2 * q4.exact_information_gib)
    assert math.isclose(fp16.exact_information_gib, 4 * q4.exact_information_gib)
    assert q4.dense_compute_gflop == q8.dense_compute_gflop
    assert q8.dense_compute_gflop == fp16.dense_compute_gflop


def test_one_skipped_coordinate_changes_output_and_top1() -> None:
    case = construct_skipped_coordinate_adversary(
        rows=4,
        columns=7,
        skipped_row=2,
        skipped_column=5,
    )
    assert case.observations_equal
    assert case.outputs_differ
    assert case.winner_flips
    assert case.baseline_winner == 3
    assert case.alternate_winner == 2
    assert case.changed_coordinates == 1
    assert case.changed_coordinate_is_uninspected
    assert case.passes


@pytest.mark.parametrize("rows,columns", [(2, 4), (3, 5), (4, 7), (8, 8)])
def test_every_single_skipped_coordinate_has_an_adversary(
    rows: int,
    columns: int,
) -> None:
    summary, cases = exhaustive_single_skip_adversaries(
        rows=rows,
        columns=columns,
    )
    assert len(cases) == rows * columns
    assert summary.total_coordinates == rows * columns
    assert summary.passing_coordinates == rows * columns
    assert summary.coverage == 1.0
    assert summary.all_observations_equal
    assert summary.all_outputs_differ
    assert summary.all_winners_flip
    assert summary.passes


def test_invalid_adversary_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        construct_skipped_coordinate_adversary(
            rows=1,
            columns=4,
            skipped_row=0,
            skipped_column=0,
        )
    with pytest.raises(ValueError):
        construct_skipped_coordinate_adversary(
            rows=2,
            columns=4,
            skipped_row=2,
            skipped_column=0,
        )
