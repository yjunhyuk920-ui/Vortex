from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.sparse_streaming import (
    SparseStreamingError,
    compile_bsr,
    compile_csr,
    compile_registered_sparse_formats,
    compile_row_runs,
    packed_bytes,
    select_favorable_sparse_format,
    unsigned_width,
)


def assert_all_reconstruct(matrix: np.ndarray) -> None:
    plans = compile_registered_sparse_formats(matrix, materialize=True)
    for plan in plans:
        assert np.array_equal(plan.reconstruct(), matrix), plan.kind


def test_rejects_invalid_matrix() -> None:
    with pytest.raises(SparseStreamingError):
        compile_csr(np.asarray([], dtype=np.int64))
    with pytest.raises(SparseStreamingError):
        compile_csr(np.ones((2, 2), dtype=np.float32))


def test_packed_bytes_and_unsigned_width() -> None:
    assert packed_bytes(3, 4) == 2
    assert packed_bytes(16, 4) == 8
    assert unsigned_width(0) == 1
    assert unsigned_width(255) == 1
    assert unsigned_width(256) == 2


def test_all_formats_reconstruct_exact_matrix() -> None:
    matrix = np.asarray(
        [[0, 1, 2, 0, 0], [3, 0, 4, 5, 0], [0, 0, 0, 0, 6]],
        dtype=np.int64,
    )
    assert_all_reconstruct(matrix)


def test_csr_charges_values_indexes_and_row_pointers() -> None:
    matrix = np.asarray([[0, 2, 0, 3], [4, 0, 0, 0]], dtype=np.int64)
    plan = compile_csr(matrix, materialize=True)
    assert plan.operation_terms == 3
    assert plan.metadata_bytes == 3 + 3
    assert plan.encoded_bytes == 2 + 6
    assert np.array_equal(plan.reconstruct(), matrix)


def test_row_runs_join_only_consecutive_nonzeros() -> None:
    matrix = np.asarray([[1, 2, 0, 3, 4, 5, 0]], dtype=np.int64)
    plan = compile_row_runs(matrix, materialize=True)
    assert len(plan.payload) == 2
    assert plan.operation_terms == 5
    assert np.array_equal(plan.reconstruct(), matrix)


def test_bsr_charges_internal_zeros_and_edge_padding() -> None:
    matrix = np.asarray([[1, 0, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int64)
    plan = compile_bsr(matrix, block_shape=(2, 2), materialize=True)
    assert plan.stored_value_slots == 8
    assert plan.operation_terms == 8
    assert plan.nonzero_scalar_count == 2
    assert np.array_equal(plan.reconstruct(), matrix)


def test_highly_sparse_control_selects_sparse_format_below_ten_percent() -> None:
    matrix = np.zeros((32, 32), dtype=np.int64)
    matrix[0, 0] = 1
    matrix[17, 19] = -2
    plans = compile_registered_sparse_formats(matrix, materialize=True)
    selected = select_favorable_sparse_format(plans)
    assert selected.operation_fraction < 0.10
    assert np.array_equal(selected.reconstruct(), matrix)


def test_dense_random_control_selects_dense() -> None:
    matrix = np.random.default_rng(600060).integers(1, 8, size=(16, 16), dtype=np.int64)
    plans = compile_registered_sparse_formats(matrix, materialize=True)
    selected = select_favorable_sparse_format(plans)
    assert selected.kind == "dense"
    assert selected.operation_fraction == 1.0
    assert selected.query_byte_fraction == 1.0


def test_isolated_zero_adversary_exposes_block_padding_waste() -> None:
    matrix = np.ones((16, 16), dtype=np.int64)
    matrix[::2, ::2] = 0
    plans = compile_registered_sparse_formats(matrix)
    bsr = next(plan for plan in plans if plan.kind == "bsr_8x8")
    csr = next(plan for plan in plans if plan.kind == "csr")
    assert bsr.operation_fraction == 1.0
    assert csr.operation_fraction == 0.75


def test_block_zero_control_favors_matching_bsr_over_dense_work() -> None:
    matrix = np.zeros((16, 16), dtype=np.int64)
    matrix[:8, :8] = 3
    plans = compile_registered_sparse_formats(matrix, materialize=True)
    bsr = next(plan for plan in plans if plan.kind == "bsr_8x8")
    assert bsr.operation_fraction == 0.25
    assert np.array_equal(bsr.reconstruct(), matrix)
