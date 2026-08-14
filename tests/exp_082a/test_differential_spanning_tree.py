from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.differential_spanning_tree import (
    DifferentialTreeError,
    best_orientation_bound,
    certified_orientation_bound,
    exact_block_pattern_profile,
    exact_block_distance,
    exact_hamming_distance,
    exact_mst_hamming_weight,
    nearest_block_distances,
    oriented_vectors,
)


def test_exact_block_distance_never_exceeds_hamming_distance() -> None:
    rng = np.random.default_rng(820001)
    matrix = rng.integers(-7, 8, size=(17, 29), dtype=np.int16)
    vectors = oriented_vectors(matrix, "rows")
    profile = exact_block_pattern_profile(vectors, block_size=5)
    for left in range(vectors.shape[0]):
        for right in range(vectors.shape[0]):
            block_distance = exact_block_distance(profile, left, right)
            assert block_distance <= exact_hamming_distance(
                vectors[left], vectors[right]
            )


@pytest.mark.parametrize("orientation", ["rows", "columns"])
def test_certified_bound_never_exceeds_exact_mst(orientation: str) -> None:
    rng = np.random.default_rng(820002)
    for rows, columns in ((2, 3), (4, 5), (7, 4), (8, 9)):
        matrix = rng.integers(-2, 3, size=(rows, columns), dtype=np.int16)
        bound = certified_orientation_bound(
            matrix,
            orientation=orientation,
            block_size=3,
            chunk_rows=2,
        )
        exact = exact_mst_hamming_weight(matrix, orientation=orientation)
        assert 0 <= bound.mst_hamming_lower_bound <= exact


def test_zero_and_duplicate_vertices_keep_bound_sound() -> None:
    matrix = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        dtype=np.int16,
    )
    bound = certified_orientation_bound(
        matrix, orientation="rows", block_size=2, chunk_rows=2
    )
    assert bound.nearest_minimum == 0
    assert bound.mst_hamming_lower_bound <= exact_mst_hamming_weight(
        matrix, orientation="rows"
    )


def test_universally_distinct_blocks_avoid_pairwise_payload() -> None:
    matrix = np.eye(8, dtype=np.int16)
    vectors = oriented_vectors(matrix, "rows")
    profile = exact_block_pattern_profile(vectors, block_size=8)
    assert profile.universally_distinct_blocks == 1
    assert profile.collision_block_count == 0
    nearest = nearest_block_distances(profile, chunk_rows=3)
    assert np.array_equal(nearest, np.ones(9, dtype=np.int32))


def test_best_orientation_grants_the_smaller_bound() -> None:
    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    result = best_orientation_bound(matrix, block_size=1, chunk_rows=2)
    assert result.selected_lower_bound == min(
        result.rows.mst_hamming_lower_bound,
        result.columns.mst_hamming_lower_bound,
    )
    assert 0 <= result.operation_fraction_lower_bound <= 1


def test_reference_is_deterministic() -> None:
    matrix = np.random.default_rng(820003).integers(
        -7, 8, size=(13, 11), dtype=np.int16
    )
    first = best_orientation_bound(matrix, block_size=4, chunk_rows=3)
    second = best_orientation_bound(matrix.copy(), block_size=4, chunk_rows=5)
    assert first == second


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(DifferentialTreeError):
        best_orientation_bound(np.ones(4, dtype=np.int16), block_size=2, chunk_rows=1)
    with pytest.raises(DifferentialTreeError):
        best_orientation_bound(
            np.ones((2, 2), dtype=np.float32), block_size=2, chunk_rows=1
        )
    with pytest.raises(DifferentialTreeError):
        certified_orientation_bound(
            np.ones((2, 2), dtype=np.int16),
            orientation="diagonal",  # type: ignore[arg-type]
            block_size=2,
            chunk_rows=1,
        )
    with pytest.raises(DifferentialTreeError):
        best_orientation_bound(
            np.ones((2, 2), dtype=np.int16), block_size=0, chunk_rows=1
        )
    with pytest.raises(DifferentialTreeError):
        best_orientation_bound(
            np.ones((2, 2), dtype=np.int16), block_size=1, chunk_rows=0
        )
