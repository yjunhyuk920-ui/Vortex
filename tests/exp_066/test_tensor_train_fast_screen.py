from __future__ import annotations

import numpy as np

from vortex_runtime.tensor_train_fast_screen import (
    certify_fast_screen_plan,
    threshold_rank_witness,
)
from vortex_runtime.tensor_train_rank import TensorTrainModePlan


def paired_plan() -> TensorTrainModePlan:
    return TensorTrainModePlan(
        variant="fast_test",
        row_schedule=(2, 2, 2),
        column_schedule=(2, 2, 2),
        mode_pairs=((2, 2), (2, 2), (2, 2)),
    )


def test_threshold_witness_reports_exact_deficiency() -> None:
    matrix = np.outer(
        np.asarray([1, 2, 3, 4], dtype=np.int16),
        np.asarray([2, 1, 3, 5], dtype=np.int16),
    )
    witness = threshold_rank_witness(matrix, prime=251, required_rank=2)
    assert witness.rank_lower_bound == 1
    assert not witness.reached_required_rank
    assert witness.exact_modular_rank_if_below_required == 1
    assert witness.certified_minor_determinant != 0


def test_threshold_witness_stops_at_requested_rank() -> None:
    matrix = np.random.default_rng(660067).integers(
        -7, 8, size=(16, 16), dtype=np.int16
    )
    witness = threshold_rank_witness(matrix, prime=257, required_rank=3)
    assert witness.rank_lower_bound == 3
    assert witness.reached_required_rank
    assert len(witness.pivot_rows) == 3
    assert len(witness.pivot_columns) == 3
    assert witness.certified_minor_determinant != 0


def test_fast_screen_is_a_rigorous_censored_lower_bound() -> None:
    matrix = np.random.default_rng(660068).integers(
        -7, 8, size=(8, 8), dtype=np.int16
    )
    plan = certify_fast_screen_plan(
        matrix,
        paired_plan(),
        primes=(251, 257),
        rejection_operation_fraction=0.25,
        rejection_storage_fraction=0.25,
    )
    assert plan.witness_mismatches == 0
    assert plan.operation_fraction > 0.25
    assert plan.storage_fraction > 0.25
    assert plan.certified_cuts
    assert all(len(witnesses) == 2 for _, witnesses in plan.witnesses)
