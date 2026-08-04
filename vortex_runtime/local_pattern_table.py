"""Exact Q4 short-block pattern table accounting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class LocalPatternError(ValueError):
    """Raised when a local-pattern plan is invalid."""


def ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return int(math.ceil(math.log2(value)))


def bit_reversal_order(length: int) -> tuple[int, ...]:
    if length <= 0 or length & (length - 1):
        raise LocalPatternError("bit reversal requires a positive power-of-two length")
    bits = length.bit_length() - 1
    return tuple(int(f"{index:0{bits}b}"[::-1], 2) for index in range(length))


def _matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(matrix)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise LocalPatternError("matrix must be non-empty and two-dimensional")
    if not np.issubdtype(value.dtype, np.signedinteger):
        raise LocalPatternError("matrix must use a signed integer dtype")
    return value


def lexicographic_signature_order(matrix: np.ndarray) -> tuple[int, ...]:
    value = _matrix(matrix)
    return tuple(sorted(range(value.shape[1]), key=lambda c: tuple(int(x) for x in value[:, c])))


def registered_orders(matrix: np.ndarray, names: Sequence[str]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    value = _matrix(matrix)
    length = int(value.shape[1])
    rows = []
    for name in names:
        if name == "natural":
            order = tuple(range(length))
        elif name == "bit_reversal":
            if length & (length - 1):
                continue
            order = bit_reversal_order(length)
        elif name == "lexicographic_signature":
            order = lexicographic_signature_order(value)
        else:
            raise LocalPatternError(f"unsupported order: {name}")
        rows.append((name, order))
    if not rows:
        raise LocalPatternError("no registered column order applies")
    return tuple(rows)


@dataclass(frozen=True)
class BlockRow:
    block_index: int
    start: int
    stop: int
    width: int
    distinct_pattern_count: int
    nonzero_pattern_count: int
    identifier_bits: int
    dictionary_bits: int
    identifier_stream_bits: int
    offset_bits: int
    partial_dot_terms: int
    gather_terms: int
    accumulation_terms: int

    def as_dict(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class LocalPatternPlan:
    rows: int
    columns: int
    block_width: int
    order_name: str
    block_count: int
    distinct_pattern_total: int
    nonzero_pattern_total: int
    dense_operation_terms: int
    table_operation_terms: int
    dense_q4_bits: int
    row_scale_bits: int
    dictionary_bits: int
    identifier_stream_bits: int
    block_offset_bits: int
    permutation_bits: int
    static_representation_bits: int
    query_bits: int
    reconstruction_mismatches: int
    hash_collision_mismatches: int
    blocks: tuple[BlockRow, ...]

    @property
    def operation_fraction(self) -> float:
        return self.table_operation_terms / self.dense_operation_terms

    @property
    def query_byte_fraction(self) -> float:
        return self.query_bits / self.dense_q4_bits

    @property
    def static_representation_fraction(self) -> float:
        return self.static_representation_bits / self.dense_q4_bits

    @property
    def joint_fraction(self) -> float:
        return max(self.operation_fraction, self.query_byte_fraction, self.static_representation_fraction)

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__ if field != "blocks"}
        result.update({
            "operation_fraction": self.operation_fraction,
            "query_byte_fraction": self.query_byte_fraction,
            "static_representation_fraction": self.static_representation_fraction,
            "joint_fraction": self.joint_fraction,
            "blocks": [row.as_dict() for row in self.blocks],
        })
        return result


def analyze_local_pattern_plan(matrix: np.ndarray, *, block_width: int, order_name: str, order: Sequence[int]) -> LocalPatternPlan:
    value = _matrix(matrix)
    rows, columns = (int(x) for x in value.shape)
    if block_width <= 0:
        raise LocalPatternError("block width must be positive")
    permutation = tuple(int(x) for x in order)
    if len(permutation) != columns or sorted(permutation) != list(range(columns)):
        raise LocalPatternError("column order is not a permutation")
    ordered = np.ascontiguousarray(value[:, permutation])
    reconstructed = np.empty_like(ordered)
    blocks = []
    dictionary_bits = identifiers_bits = offsets_bits = 0
    partial_terms = gather_terms = accumulation_terms = 0
    collision_mismatches = distinct_total = nonzero_total = 0

    for block_index, start in enumerate(range(0, columns, block_width)):
        stop = min(columns, start + block_width)
        block = np.ascontiguousarray(ordered[:, start:stop])
        width = stop - start
        dictionary = []
        by_bytes = {}
        ids = np.empty(rows, dtype=np.int64)
        for row_index in range(rows):
            vector = np.ascontiguousarray(block[row_index])
            key = vector.tobytes()
            pattern = tuple(int(x) for x in vector)
            candidates = by_bytes.setdefault(key, [])
            found = None
            for candidate in candidates:
                if dictionary[candidate] == pattern:
                    found = candidate
                    break
                collision_mismatches += 1
            if found is None:
                found = len(dictionary)
                dictionary.append(pattern)
                candidates.append(found)
            ids[row_index] = found
        distinct = len(dictionary)
        nonzero = sum(any(c != 0 for c in pattern) for pattern in dictionary)
        id_bits = ceil_log2(distinct)
        dictionary_bits_here = distinct * width * 4
        identifier_bits_here = rows * id_bits
        offset_bits_here = 64
        dot_terms_here = sum(sum(c != 0 for c in pattern) for pattern in dictionary)
        gather_here = rows
        accumulation_here = rows if block_index else 0
        dictionary_bits += dictionary_bits_here
        identifiers_bits += identifier_bits_here
        offsets_bits += offset_bits_here
        partial_terms += dot_terms_here
        gather_terms += gather_here
        accumulation_terms += accumulation_here
        distinct_total += distinct
        nonzero_total += nonzero
        reconstructed[:, start:stop] = np.asarray(dictionary, dtype=value.dtype)[ids]
        blocks.append(BlockRow(block_index, start, stop, width, distinct, nonzero, id_bits, dictionary_bits_here, identifier_bits_here, offset_bits_here, dot_terms_here, gather_here, accumulation_here))

    reconstruction_mismatches = int(np.count_nonzero(reconstructed != ordered))
    natural = permutation == tuple(range(columns))
    permutation_bits = 0 if natural else columns * ceil_log2(columns)
    permutation_terms = 0 if natural else columns
    scale_terms = rows
    scale_bits = rows * 32
    dense_terms = rows * columns + scale_terms
    table_terms = partial_terms + gather_terms + accumulation_terms + permutation_terms + scale_terms
    dense_bits = rows * columns * 4 + scale_bits
    representation_bits = dictionary_bits + identifiers_bits + offsets_bits + permutation_bits + scale_bits
    return LocalPatternPlan(rows, columns, int(block_width), str(order_name), len(blocks), distinct_total, nonzero_total, dense_terms, table_terms, dense_bits, scale_bits, dictionary_bits, identifiers_bits, offsets_bits, permutation_bits, representation_bits, representation_bits, reconstruction_mismatches, collision_mismatches, tuple(blocks))


def choose_joint_plan(plans: Iterable[LocalPatternPlan]) -> LocalPatternPlan:
    population = tuple(plans)
    if not population:
        raise LocalPatternError("cannot choose from an empty plan population")
    if any(p.reconstruction_mismatches or p.hash_collision_mismatches for p in population):
        raise LocalPatternError("invalid plan cannot be selected")
    return min(population, key=lambda p: (p.joint_fraction, p.operation_fraction + p.query_byte_fraction + p.static_representation_fraction, p.block_width, p.order_name))
