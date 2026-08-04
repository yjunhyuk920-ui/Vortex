"""Minor-first exact modular rank lower-bound witnesses."""
from __future__ import annotations

from typing import Any

import numpy as np

from vortex_runtime.modular_rank import modular_determinant
from vortex_runtime.tensor_train_fast_screen import (
    ThresholdRankWitness,
    threshold_rank_witness as elimination_threshold_rank_witness,
)
from vortex_runtime.tensor_train_rank import TensorTrainRankError


def _index_sets(size: int, rank: int) -> tuple[tuple[int, ...], ...]:
    candidates: list[tuple[int, ...]] = []

    def add(values: Any) -> None:
        item = tuple(int(value) for value in values)
        if len(item) == rank and len(set(item)) == rank and item not in candidates:
            candidates.append(item)

    add(range(rank))
    add(range(size - rank, size))
    if rank == 1:
        add((size // 2,))
    else:
        add(np.rint(np.linspace(0, size - 1, rank)).astype(np.int64))
    if size >= 2 * rank:
        add(range(rank, 2 * rank))
        add(range(size // 2 - rank // 2, size // 2 - rank // 2 + rank))
    return tuple(candidates)


def minor_first_threshold_rank_witness(
    matrix: Any, *, prime: int, required_rank: int
) -> ThresholdRankWitness:
    """Prove rank >= required_rank from small deterministic minors first.

    A nonzero selected minor is a complete exact witness.  If all bounded
    candidate minors are singular, fall back to deterministic truncated
    elimination, preserving completeness for low-structure controls.
    """
    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0 or source.dtype.kind not in "iu":
        raise TensorTrainRankError("a nonempty integer matrix is required")
    minimum = min(source.shape)
    if required_rank <= 0 or required_rank > minimum:
        raise TensorTrainRankError("required rank outside matrix bounds")

    row_sets = _index_sets(int(source.shape[0]), required_rank)
    column_sets = _index_sets(int(source.shape[1]), required_rank)
    pair_candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for index in range(max(len(row_sets), len(column_sets))):
        pair = (
            row_sets[index % len(row_sets)],
            column_sets[index % len(column_sets)],
        )
        if pair not in pair_candidates:
            pair_candidates.append(pair)
    for row_indexes in row_sets[:3]:
        for column_indexes in column_sets[:3]:
            pair = (row_indexes, column_indexes)
            if pair not in pair_candidates:
                pair_candidates.append(pair)

    for row_indexes, column_indexes in pair_candidates:
        minor = source[np.ix_(row_indexes, column_indexes)]
        determinant = modular_determinant(minor, prime=prime)
        if determinant != 0:
            return ThresholdRankWitness(
                prime=int(prime),
                required_rank=required_rank,
                rank_lower_bound=required_rank,
                reached_required_rank=True,
                exact_modular_rank_if_below_required=None,
                minimum_dimension=minimum,
                pivot_rows=row_indexes,
                pivot_columns=column_indexes,
                certified_minor_determinant=determinant,
            )

    return elimination_threshold_rank_witness(
        source, prime=prime, required_rank=required_rank
    )
