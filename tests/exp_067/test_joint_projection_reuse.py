from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.joint_projection_reuse import (
    JointProjectionReuseError,
    analyze_joint_rows,
    canonical_integer_row,
    exact_repeated_block_stats,
)


def test_canonical_integer_row_reconstructs_sign_and_gcd() -> None:
    for row, primitive, multiplier in (
        ([2, 4, -6], [1, 2, -3], 2),
        ([-2, -4, 6], [1, 2, -3], -2),
        ([0, -3, 6], [0, 1, -2], -3),
    ):
        result = canonical_integer_row(np.asarray(row, dtype=np.int8))
        assert result.multiplier == multiplier
        np.testing.assert_array_equal(result.primitive, primitive)
        np.testing.assert_array_equal(
            result.primitive * result.multiplier, row
        )


def test_joint_reuse_finds_duplicate_sign_and_integer_multiple() -> None:
    first = np.asarray(
        [[1, 2, 0], [2, 4, 0], [3, 0, 1]], dtype=np.int8
    )
    second = np.asarray(
        [[-1, -2, 0], [0, 0, 0], [3, 0, 1]], dtype=np.int8
    )
    plan = analyze_joint_rows((first, second))
    assert plan.total_rows == 6
    assert plan.zero_rows == 1
    assert plan.primitive_class_count == 2
    assert plan.reusable_rows == 3
    assert plan.maximum_class_size == 3
    assert plan.operation_fraction < 1.0


def test_one_nibble_mutation_breaks_a_class() -> None:
    a = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = a.copy()
    original = analyze_joint_rows((a, b))
    b[0, 0] += 1
    mutated = analyze_joint_rows((a, b))
    assert mutated.primitive_class_count > original.primitive_class_count
    assert mutated.reusable_rows < original.reusable_rows


def test_random_dense_rows_have_no_exact_reuse() -> None:
    rng = np.random.default_rng(67)
    a = rng.integers(-7, 8, size=(32, 16), dtype=np.int8)
    b = rng.integers(-7, 8, size=(32, 16), dtype=np.int8)
    plan = analyze_joint_rows((a, b))
    assert plan.reusable_row_fraction == 0.0
    assert plan.operation_fraction >= 1.0


def test_exact_repeated_blocks() -> None:
    block = np.arange(8, dtype=np.int8).reshape(2, 4)
    a = np.vstack((block, block))
    b = np.vstack((block, -block))
    stats = exact_repeated_block_stats((a, b), block_rows=2)
    assert stats["block_count"] == 4
    assert stats["unique_block_count"] == 2
    assert stats["reusable_block_count"] == 2


def test_malformed_requests_fail_closed() -> None:
    with pytest.raises(JointProjectionReuseError):
        analyze_joint_rows((np.ones((2, 3), dtype=np.int8),))
    with pytest.raises(JointProjectionReuseError):
        analyze_joint_rows(
            (
                np.ones((2, 3), dtype=np.int8),
                np.ones((2, 4), dtype=np.int8),
            )
        )
