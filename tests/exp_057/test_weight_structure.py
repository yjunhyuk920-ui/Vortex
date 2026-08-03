from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.weight_structure import (
    WeightStructureError,
    column_group_stats,
    deterministic_column_shuffle,
    deterministic_element_permutation,
    prototype_residual_stats,
    symmetric_row_quantize,
)


def test_q4_and_q8_quantization_are_deterministic_and_bounded() -> None:
    matrix = np.array(
        [
            [-3.5, -1.75, 0.0, 1.75, 3.5],
            [0.2, -0.4, 0.6, -0.8, 1.0],
        ],
        dtype=np.float32,
    )
    for bits, qmax in ((4, 7), (8, 127)):
        left = symmetric_row_quantize(matrix, bits=bits)
        right = symmetric_row_quantize(matrix, bits=bits)
        assert np.array_equal(left.values, right.values)
        assert np.array_equal(left.scales, right.scales)
        assert int(left.values.min()) >= -qmax
        assert int(left.values.max()) <= qmax
        assert left.clipped_value_count == 0
        assert left.maximum_absolute_error >= 0.0
        assert left.mean_absolute_error >= 0.0


def test_zero_rows_are_exactly_preserved() -> None:
    matrix = np.array(
        [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]], dtype=np.float32
    )
    quantized = symmetric_row_quantize(matrix, bits=4)
    assert quantized.zero_row_count == 1
    assert np.array_equal(quantized.values[0], np.zeros(3, dtype=np.int16))
    assert quantized.scales[0] == np.float32(1.0)


def test_identical_and_sign_canonical_groups_are_counted_exactly() -> None:
    matrix = np.array(
        [
            [1, 1, -1, -1, 0, 0],
            [2, 2, -2, -2, 0, 0],
            [3, 3, -3, -3, 0, 0],
        ],
        dtype=np.int16,
    )
    stats = column_group_stats(matrix, scalar_bits=8)
    assert stats["identical"]["group_count"] == 3
    assert stats["identical"]["largest_group"] == 2
    assert stats["sign_canonical"]["group_count"] == 2
    assert stats["sign_canonical"]["largest_group"] == 4
    assert stats["zero_column_count"] == 2
    assert stats["selected_grouping"] == "sign_canonical"


def test_column_shuffle_preserves_structural_counts_and_dictionary_density() -> None:
    matrix = np.array(
        [
            [1, 1, 2, 2, 1, 2, 1, 2],
            [3, 3, 4, 4, 3, 4, 3, 4],
            [-1, -1, -2, -2, -1, -2, -1, -2],
        ],
        dtype=np.int16,
    )
    shuffled = deterministic_column_shuffle(matrix, seed=57)
    original_groups = column_group_stats(matrix, scalar_bits=4)
    shuffled_groups = column_group_stats(shuffled, scalar_bits=4)
    for mode in ("identical", "sign_canonical"):
        for key in (
            "group_count",
            "largest_group",
            "repeated_column_coverage_fraction",
        ):
            assert shuffled_groups[mode][key] == original_groups[mode][key]

    original_dictionary = prototype_residual_stats(
        matrix, scalar_bits=4, candidate_cap=16
    )
    shuffled_dictionary = prototype_residual_stats(
        shuffled, scalar_bits=4, candidate_cap=16
    )
    assert (
        shuffled_dictionary["selected"]["residual_scalar_count"]
        == original_dictionary["selected"]["residual_scalar_count"]
    )
    assert (
        shuffled_dictionary["selected"]["operation_fraction"]
        == original_dictionary["selected"]["operation_fraction"]
    )


def test_element_permutation_destroys_aligned_column_repetition() -> None:
    matrix = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8],
        ],
        dtype=np.int16,
    )
    original = column_group_stats(matrix, scalar_bits=4)
    permuted = deterministic_element_permutation(matrix, seed=57057)
    adversarial = column_group_stats(permuted, scalar_bits=4)
    assert original["identical"]["group_count"] == 1
    assert adversarial["identical"]["group_count"] > 1
    assert (
        adversarial["identical"]["repeated_column_coverage_fraction"]
        < original["identical"]["repeated_column_coverage_fraction"]
    )


def test_prototype_residual_positive_control_beats_unique_adversary() -> None:
    repeated = np.array(
        [
            [2, 2, 2, 2, 2, 2, 3, 2],
            [-3, -3, -3, -3, -3, -3, -3, -2],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        dtype=np.int16,
    )
    unique = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, -7],
            [-7, -6, -5, -4, -3, -2, -1, 1],
            [7, 5, 3, 1, -1, -3, -5, -7],
            [-6, -4, -2, 2, 4, 6, -7, 7],
        ],
        dtype=np.int16,
    )
    structured = prototype_residual_stats(
        repeated, scalar_bits=4, candidate_cap=16
    )
    adversarial = prototype_residual_stats(
        unique, scalar_bits=4, candidate_cap=16
    )
    assert structured["selected"]["residual_scalar_count"] == 2
    assert (
        structured["selected"]["operation_fraction"]
        < adversarial["selected"]["operation_fraction"]
    )
    assert all(
        plan["reconstruction_mismatches"] == 0
        for plan in structured["plans"] + adversarial["plans"]
    )


def test_float_bit_pattern_grouping_is_exact() -> None:
    positive_zero = np.float32(0.0)
    negative_zero = np.float32(-0.0)
    matrix = np.array(
        [
            [1.0, 1.0, -1.0, positive_zero, negative_zero],
            [2.0, 2.0, -2.0, positive_zero, negative_zero],
        ],
        dtype=np.float32,
    )
    stats = column_group_stats(matrix, scalar_bits=32)
    assert stats["identical"]["largest_group"] == 2
    assert stats["sign_canonical"]["largest_group"] >= 3
    # +0.0 and -0.0 are distinct exact stored bit patterns in identical mode.
    assert stats["identical"]["group_count"] == 4


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(WeightStructureError):
        symmetric_row_quantize(np.array([1.0], dtype=np.float32), bits=4)
    with pytest.raises(WeightStructureError):
        symmetric_row_quantize(np.ones((2, 2), dtype=np.float32), bits=3)
    with pytest.raises(WeightStructureError):
        column_group_stats(np.array([1, 2], dtype=np.int16), scalar_bits=4)
    with pytest.raises(WeightStructureError):
        prototype_residual_stats(
            np.ones((2, 2), dtype=np.float32), scalar_bits=4
        )
