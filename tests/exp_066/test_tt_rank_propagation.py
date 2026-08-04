from __future__ import annotations

from vortex_runtime.tensor_train_rank import TensorTrainModePlan
from vortex_runtime.tt_kronecker_reuse import derive_reused_tt_plan
from vortex_runtime.tt_rank_propagation import (
    is_full_matrix_or_transpose_cut,
    propagate_adjacent_bond_lower_bounds,
    strengthen_with_full_matrix_rank,
)


def test_full_matrix_cut_patterns() -> None:
    assert is_full_matrix_or_transpose_cut((1, 64, 64, 1))
    assert is_full_matrix_or_transpose_cut((64, 1, 1, 64))
    assert not is_full_matrix_or_transpose_cut((2, 32, 2, 32))


def test_adjacent_rank_bounds_propagate_both_directions() -> None:
    assert propagate_adjacent_bond_lower_bounds(
        (2, 2, 2, 2), (1, 8, 1)
    ) == (4, 8, 4)


def test_full_matrix_rank_strengthens_column_then_row_plan() -> None:
    plan = TensorTrainModePlan(
        variant="column_then_row",
        row_schedule=(2, 2, 2),
        column_schedule=(2, 2, 2),
        mode_pairs=((1, 2), (1, 2), (1, 2), (2, 1), (2, 1), (2, 1)),
    )
    base = derive_reused_tt_plan(
        model_id="model",
        tensor_name="weight",
        rows=8,
        columns=8,
        has_bias=False,
        mode_plan=plan,
        exp065_index={},
    )
    strengthened = strengthen_with_full_matrix_rank(
        base,
        matrix_rank_lower_bound=8,
        has_bias=False,
    )
    assert strengthened.bond_rank_lower_bounds == (2, 4, 8, 4, 2)
    assert strengthened.operation_fraction > base.operation_fraction
    assert any("EXP-058" in row["source"] for row in strengthened.cut_records)
