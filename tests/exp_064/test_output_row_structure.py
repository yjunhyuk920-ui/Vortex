from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.output_row_structure import (
    OutputRowStructureError,
    compile_output_row_plans,
    select_output_row_plan,
)


def compile(matrix: np.ndarray, *, bias: bool = True):
    return compile_output_row_plans(
        matrix,
        scales=np.linspace(0.1, 1.0, matrix.shape[0], dtype=np.float32),
        bits_per_weight=4,
        activation_bytes=4,
        has_bias=bias,
        prototype_counts=(1, 2, 4, 8),
        candidate_cap=32,
    )


def deployable(plans):
    selected = select_output_row_plan(plans)
    dense = next(plan for plan in plans if plan.mechanism == "dense")
    return (
        selected
        if selected.operation_fraction < 1.0
        and selected.query_byte_fraction < 1.0
        else dense
    )


def test_identical_rows_share_one_integer_dot() -> None:
    matrix = np.tile(np.arange(16, dtype=np.int16), (16, 1))
    selected = deployable(compile(matrix))
    assert selected.mechanism == "identical_rows"
    assert selected.prototype_count == 1
    assert selected.duplicate_row_count == 15
    assert selected.operation_fraction < 0.25


def test_sign_related_rows_share_canonical_dot() -> None:
    row = np.arange(16, dtype=np.int16)
    matrix = np.vstack([row, -row] * 8)
    selected = deployable(compile(matrix))
    assert selected.mechanism == "sign_canonical_rows"
    assert selected.prototype_count == 1
    assert selected.negative_row_count == 8
    assert selected.operation_fraction < 0.30


def test_sparse_delta_reconstructs_exactly() -> None:
    matrix = np.tile(np.arange(16, dtype=np.int16), (16, 1))
    matrix[np.arange(16), np.arange(16)] += 1
    plans = compile(matrix)
    sparse = [
        plan
        for plan in plans
        if plan.mechanism.startswith("prototype_sparse_delta")
    ]
    assert sparse
    assert all(plan.reconstruction_mismatches == 0 for plan in sparse)
    assert min(plan.residual_scalar_fraction for plan in sparse) < 0.15
    assert deployable(plans).mechanism.startswith("prototype_sparse_delta")


def test_one_nibble_change_prevents_false_identity() -> None:
    row = np.arange(16, dtype=np.int16)
    matrix = np.vstack([row, row.copy()])
    matrix[1, 7] += 1
    identical = next(
        plan for plan in compile(matrix) if plan.mechanism == "identical_rows"
    )
    assert identical.prototype_count == 2
    assert identical.duplicate_row_count == 0


def test_dense_random_fails_closed_to_dense() -> None:
    matrix = np.random.default_rng(640064).integers(
        -7, 8, size=(32, 32), dtype=np.int16
    )
    selected = deployable(compile(matrix))
    assert selected.mechanism == "dense"
    assert selected.operation_fraction == 1.0
    assert selected.query_byte_fraction == 1.0


def test_forced_unique_rows_have_no_exact_group_reuse() -> None:
    matrix = np.random.default_rng(640065).integers(
        -7, 8, size=(16, 16), dtype=np.int16
    )
    matrix[:, 0] = np.arange(-8, 8, dtype=np.int16)
    identical = next(
        plan for plan in compile(matrix) if plan.mechanism == "identical_rows"
    )
    sign = next(
        plan for plan in compile(matrix) if plan.mechanism == "sign_canonical_rows"
    )
    assert identical.prototype_count == 16
    assert sign.prototype_count == 16


def test_bias_and_scale_costs_are_never_removed() -> None:
    matrix = np.tile(np.arange(8, dtype=np.int16), (8, 1))
    with_bias = deployable(compile(matrix, bias=True))
    without_bias = deployable(compile(matrix, bias=False))
    assert with_bias.candidate_operations == without_bias.candidate_operations + 8
    assert with_bias.candidate_query_bytes == without_bias.candidate_query_bytes + 32


def test_compilation_is_deterministic() -> None:
    matrix = np.random.default_rng(1234).integers(
        -7, 8, size=(20, 24), dtype=np.int16
    )
    first = [plan.accounting() for plan in compile(matrix)]
    second = [plan.accounting() for plan in compile(matrix.copy())]
    assert first == second


def test_every_plan_reports_exact_reconstruction() -> None:
    matrix = np.random.default_rng(99).integers(
        -7, 8, size=(12, 10), dtype=np.int16
    )
    assert all(plan.reconstruction_mismatches == 0 for plan in compile(matrix))


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(OutputRowStructureError):
        compile_output_row_plans(
            np.ones(8, dtype=np.int16), scales=np.ones(8, dtype=np.float32)
        )
    with pytest.raises(OutputRowStructureError):
        compile_output_row_plans(
            np.ones((2, 2), dtype=np.float32), scales=np.ones(2, dtype=np.float32)
        )
    with pytest.raises(OutputRowStructureError):
        compile_output_row_plans(
            np.ones((2, 2), dtype=np.int16), scales=np.ones(1, dtype=np.float32)
        )
