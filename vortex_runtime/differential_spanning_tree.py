"""Certified lower bounds for exact differential spanning-tree MatVec.

For a matrix with an added all-zero vertex, a row-tree evaluates every output
from exact sparse parent/child differences.  The dual column-tree rewrites the
same product through subtree sums.  This module implements only the registered
EXP-082A Stage-1 lower bound and small exact references; it deliberately does
not construct a model-scale tree after a decisive rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np


Orientation = Literal["rows", "columns"]


class DifferentialTreeError(ValueError):
    """Raised when a differential-tree structural input is malformed."""


def _integer_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise DifferentialTreeError("matrix must be nonempty and rank two")
    if matrix.dtype.kind not in "iu":
        raise DifferentialTreeError("matrix must use an integer dtype")
    return np.ascontiguousarray(matrix)


def oriented_vectors(matrix: Any, orientation: Orientation) -> np.ndarray:
    source = _integer_matrix(matrix)
    if orientation == "rows":
        vectors = source
    elif orientation == "columns":
        vectors = source.T
    else:
        raise DifferentialTreeError(f"unknown orientation: {orientation}")
    zero = np.zeros((1, vectors.shape[1]), dtype=vectors.dtype)
    return np.ascontiguousarray(np.concatenate((zero, vectors), axis=0))


def _exact_block_ids(block: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(block)
    row_bytes = contiguous.dtype.itemsize * contiguous.shape[1]
    opaque = contiguous.view(np.dtype((np.void, row_bytes))).reshape(-1)
    _, inverse = np.unique(opaque, return_inverse=True)
    return inverse.astype(np.int32, copy=False)


@dataclass(frozen=True)
class BlockPatternProfile:
    vertex_count: int
    coordinate_count: int
    block_size: int
    block_count: int
    universally_distinct_blocks: int
    collision_pattern_ids: np.ndarray

    @property
    def collision_block_count(self) -> int:
        return int(self.collision_pattern_ids.shape[1])


def exact_block_pattern_profile(
    vectors: Any, *, block_size: int
) -> BlockPatternProfile:
    source = _integer_matrix(vectors)
    if block_size <= 0:
        raise DifferentialTreeError("block size must be positive")
    collision_ids: list[np.ndarray] = []
    universal = 0
    for start in range(0, source.shape[1], block_size):
        identifiers = _exact_block_ids(source[:, start : start + block_size])
        if int(np.unique(identifiers).size) == source.shape[0]:
            universal += 1
        else:
            collision_ids.append(identifiers)
    if collision_ids:
        packed = np.stack(collision_ids, axis=1)
    else:
        packed = np.empty((source.shape[0], 0), dtype=np.int32)
    return BlockPatternProfile(
        vertex_count=int(source.shape[0]),
        coordinate_count=int(source.shape[1]),
        block_size=int(block_size),
        block_count=math.ceil(source.shape[1] / block_size),
        universally_distinct_blocks=universal,
        collision_pattern_ids=np.ascontiguousarray(packed),
    )


def nearest_block_distances(
    profile: BlockPatternProfile, *, chunk_rows: int
) -> np.ndarray:
    if profile.vertex_count < 2:
        raise DifferentialTreeError("a spanning tree needs at least two vertices")
    if chunk_rows <= 0:
        raise DifferentialTreeError("chunk rows must be positive")
    identifiers = profile.collision_pattern_ids
    nearest = np.empty(profile.vertex_count, dtype=np.int32)
    sentinel = profile.block_count + 1
    for start in range(0, profile.vertex_count, chunk_rows):
        stop = min(start + chunk_rows, profile.vertex_count)
        if identifiers.shape[1]:
            distances = np.count_nonzero(
                identifiers[start:stop, None, :] != identifiers[None, :, :],
                axis=2,
            ).astype(np.int32, copy=False)
            distances += profile.universally_distinct_blocks
        else:
            distances = np.full(
                (stop - start, profile.vertex_count),
                profile.universally_distinct_blocks,
                dtype=np.int32,
            )
        local = np.arange(stop - start)
        global_indices = np.arange(start, stop)
        distances[local, global_indices] = sentinel
        nearest[start:stop] = distances.min(axis=1)
    return nearest


def exact_block_distance(
    profile: BlockPatternProfile, left: int, right: int
) -> int:
    if not (0 <= left < profile.vertex_count and 0 <= right < profile.vertex_count):
        raise DifferentialTreeError("block-distance vertex is out of range")
    if left == right:
        return 0
    distance = profile.universally_distinct_blocks
    if profile.collision_pattern_ids.shape[1]:
        distance += int(
            np.count_nonzero(
                profile.collision_pattern_ids[left]
                != profile.collision_pattern_ids[right]
            )
        )
    return distance


@dataclass(frozen=True)
class OrientationBound:
    orientation: Orientation
    row_count: int
    column_count: int
    vertex_count: int
    coordinate_count: int
    block_size: int
    block_count: int
    universally_distinct_blocks: int
    collision_block_count: int
    nearest_minimum: int
    nearest_median: float
    nearest_maximum: int
    mst_hamming_lower_bound: int

    @property
    def dense_coefficients(self) -> int:
        return self.row_count * self.column_count

    @property
    def operation_fraction_lower_bound(self) -> float:
        return self.mst_hamming_lower_bound / self.dense_coefficients


def certified_orientation_bound(
    matrix: Any,
    *,
    orientation: Orientation,
    block_size: int,
    chunk_rows: int,
) -> OrientationBound:
    source = _integer_matrix(matrix)
    vectors = oriented_vectors(source, orientation)
    profile = exact_block_pattern_profile(vectors, block_size=block_size)
    nearest = nearest_block_distances(profile, chunk_rows=chunk_rows)
    # Every tree edge is incident to two vertices, while each incident edge is
    # at least that vertex's nearest-neighbor distance in the block metric.
    lower_bound = (int(nearest.sum(dtype=np.int64)) + 1) // 2
    return OrientationBound(
        orientation=orientation,
        row_count=int(source.shape[0]),
        column_count=int(source.shape[1]),
        vertex_count=profile.vertex_count,
        coordinate_count=profile.coordinate_count,
        block_size=profile.block_size,
        block_count=profile.block_count,
        universally_distinct_blocks=profile.universally_distinct_blocks,
        collision_block_count=profile.collision_block_count,
        nearest_minimum=int(nearest.min()),
        nearest_median=float(np.median(nearest)),
        nearest_maximum=int(nearest.max()),
        mst_hamming_lower_bound=lower_bound,
    )


@dataclass(frozen=True)
class BestOrientationBound:
    rows: OrientationBound
    columns: OrientationBound
    selected_orientation: Orientation
    selected_lower_bound: int

    @property
    def dense_coefficients(self) -> int:
        return self.rows.dense_coefficients

    @property
    def operation_fraction_lower_bound(self) -> float:
        return self.selected_lower_bound / self.dense_coefficients


def best_orientation_bound(
    matrix: Any, *, block_size: int, chunk_rows: int
) -> BestOrientationBound:
    rows = certified_orientation_bound(
        matrix,
        orientation="rows",
        block_size=block_size,
        chunk_rows=chunk_rows,
    )
    columns = certified_orientation_bound(
        matrix,
        orientation="columns",
        block_size=block_size,
        chunk_rows=chunk_rows,
    )
    selected = min(
        (rows, columns),
        key=lambda bound: (bound.mst_hamming_lower_bound, bound.orientation),
    )
    return BestOrientationBound(
        rows=rows,
        columns=columns,
        selected_orientation=selected.orientation,
        selected_lower_bound=selected.mst_hamming_lower_bound,
    )


def exact_hamming_distance(left: Any, right: Any) -> int:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise DifferentialTreeError("Hamming operands must be equal-length vectors")
    return int(np.count_nonzero(a != b))


def exact_mst_hamming_weight(matrix: Any, *, orientation: Orientation) -> int:
    """Quadratic Prim reference for registered small synthetic controls only."""

    vectors = oriented_vectors(matrix, orientation)
    vertex_count = vectors.shape[0]
    in_tree = np.zeros(vertex_count, dtype=bool)
    distance = np.full(vertex_count, np.iinfo(np.int32).max, dtype=np.int32)
    distance[0] = 0
    total = 0
    for _ in range(vertex_count):
        candidates = np.where(in_tree, np.iinfo(np.int32).max, distance)
        vertex = int(np.argmin(candidates))
        if in_tree[vertex] or distance[vertex] == np.iinfo(np.int32).max:
            raise DifferentialTreeError("complete Hamming graph became disconnected")
        in_tree[vertex] = True
        total += int(distance[vertex])
        remaining = np.flatnonzero(~in_tree)
        if remaining.size:
            edge_distances = np.count_nonzero(
                vectors[remaining] != vectors[vertex], axis=1
            ).astype(np.int32, copy=False)
            distance[remaining] = np.minimum(distance[remaining], edge_distances)
    return total
