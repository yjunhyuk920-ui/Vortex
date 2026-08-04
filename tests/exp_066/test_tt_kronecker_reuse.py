from __future__ import annotations

import math

import numpy as np

from vortex_runtime.tensor_train_rank import TensorTrainModePlan
from vortex_runtime.tt_kronecker_reuse import (
    derive_reused_tt_plan,
    index_exp065_plan_rows,
    tt_cut_factors,
    validate_cut_equivalence,
)


def test_every_internal_cut_matches_kronecker_convention() -> None:
    matrix = np.arange(8 * 12, dtype=np.int16).reshape(8, 12)
    plan = TensorTrainModePlan(
        variant="mixed",
        row_schedule=(2, 2, 2),
        column_schedule=(3, 2, 2),
        mode_pairs=((2, 3), (2, 2), (2, 2)),
    )
    for cut in range(1, len(plan.mode_pairs)):
        factors = tt_cut_factors(plan.mode_pairs, cut)
        assert factors[0] * factors[1] == matrix.shape[0]
        assert factors[2] * factors[3] == matrix.shape[1]
        assert validate_cut_equivalence(matrix, plan.mode_pairs, cut)


def test_frozen_plan_rows_become_bond_lower_bounds() -> None:
    plan = TensorTrainModePlan(
        variant="paired",
        row_schedule=(2, 2, 2),
        column_schedule=(2, 2, 2),
        mode_pairs=((2, 2), (2, 2), (2, 2)),
    )
    rows = [
        {
            "model_id": "model",
            "tensor_name": "weight",
            "factors": [2, 4, 2, 4],
            "prime_ranks": [4, 4],
            "rank_lower_bound": 4,
            "full_rearrangement_rank_proven": True,
            "witness_mismatches": 0,
        },
        {
            "model_id": "model",
            "tensor_name": "weight",
            "factors": [4, 2, 4, 2],
            "prime_ranks": [4, 4],
            "rank_lower_bound": 4,
            "full_rearrangement_rank_proven": True,
            "witness_mismatches": 0,
        },
    ]
    derived = derive_reused_tt_plan(
        model_id="model",
        tensor_name="weight",
        rows=8,
        columns=8,
        has_bias=False,
        mode_plan=plan,
        exp065_index=index_exp065_plan_rows(rows),
    )
    assert derived.bond_rank_lower_bounds == (4, 4)
    assert derived.mapped_cut_count == 2
    assert derived.unit_boundary_cut_count == 0
    assert derived.missing_nontrivial_cut_count == 0
    assert derived.source_witness_mismatches == 0
    assert derived.core_scalar_lower_bound == 96
    assert math.isclose(derived.operation_fraction, 1.625)


def test_unit_factor_cut_uses_favorable_rank_one_lower_bound() -> None:
    plan = TensorTrainModePlan(
        variant="row_then_column",
        row_schedule=(2, 2),
        column_schedule=(2, 2),
        mode_pairs=((2, 1), (2, 1), (1, 2), (1, 2)),
    )
    derived = derive_reused_tt_plan(
        model_id="model",
        tensor_name="weight",
        rows=4,
        columns=4,
        has_bias=False,
        mode_plan=plan,
        exp065_index={},
    )
    assert derived.bond_rank_lower_bounds == (1, 1, 1)
    assert derived.unit_boundary_cut_count == 3
    assert derived.missing_nontrivial_cut_count == 0
