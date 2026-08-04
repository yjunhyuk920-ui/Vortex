"""Exact Q4 short-block pattern table accounting."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import numpy as np


class LocalPatternError(ValueError):
    pass


def ceil_log2(value: int) -> int:
    return 0 if value <= 1 else int(math.ceil(math.log2(value)))


def _matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(matrix)
    if value.ndim != 2 or min(value.shape) <= 0:
        raise LocalPatternError("matrix must be non-empty and two-dimensional")
    if not np.issubdtype(value.dtype, np.signedinteger):
        raise LocalPatternError("matrix must use a signed integer dtype")
    return value


def bit_reversal_order(length: int) -> tuple[int, ...]:
    if length <= 0 or length & (length - 1):
        raise LocalPatternError("bit reversal requires a positive power-of-two length")
    bits = length.bit_length() - 1
    return tuple(int(f"{index:0{bits}b}"[::-1], 2) for index in range(length))


def lexicographic_signature_order(matrix: np.ndarray) -> tuple[int, ...]:
    value = _matrix(matrix)
    return tuple(sorted(range(value.shape[1]), key=lambda c: tuple(int(x) for x in value[:, c])))


def registered_orders(matrix: np.ndarray, names: Sequence[str]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    value = _matrix(matrix)
    length = int(value.shape[1])
    result = []
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
        result.append((name, order))
    if not result:
        raise LocalPatternError("no registered column order applies")
    return tuple(result)


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
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


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
        result = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "blocks"}
        result.update({
            "operation_fraction": self.operation_fraction,
            "query_byte_fraction": self.query_byte_fraction,
            "static_representation_fraction": self.static_representation_fraction,
            "joint_fraction": self.joint_fraction,
            "blocks": [block.as_dict() for block in self.blocks],
        })
        return result


def analyze_local_pattern_plan(matrix: np.ndarray, *, block_width: int, order_name: str, order: Sequence[int]) -> LocalPatternPlan:
    value = _matrix(matrix)
    rows, columns = map(int, value.shape)
    if block_width <= 0:
        raise LocalPatternError("block width must be positive")
    permutation = tuple(map(int, order))
    if len(permutation) != columns or sorted(permutation) != list(range(columns)):
        raise LocalPatternError("column order is not a permutation")
    ordered = np.ascontiguousarray(value[:, permutation])
    reconstructed = np.empty_like(ordered)
    blocks = []
    dictionary_bits = identifier_bits = offset_bits = partial_terms = gather_terms = 0
    distinct_total = nonzero_total = collisions = 0
    for block_index, start in enumerate(range(0, columns, block_width)):
        stop = min(columns, start + block_width)
        source = np.ascontiguousarray(ordered[:, start:stop])
        width = stop - start
        dictionary: list[tuple[int, ...]] = []
        keyed: dict[bytes, list[int]] = {}
        ids = np.empty(rows, dtype=np.int64)
        for row_index in range(rows):
            vector = np.ascontiguousarray(source[row_index])
            pattern = tuple(map(int, vector))
            candidates = keyed.setdefault(vector.tobytes(), [])
            found = None
            for candidate in candidates:
                if dictionary[candidate] == pattern:
                    found = candidate
                    break
                collisions += 1
            if found is None:
                found = len(dictionary)
                dictionary.append(pattern)
                candidates.append(found)
            ids[row_index] = found
        distinct = len(dictionary)
        nonzero = sum(any(coefficient != 0 for coefficient in pattern) for pattern in dictionary)
        id_width = ceil_log2(distinct)
        block_dictionary_bits = distinct * width * 4
        block_identifier_bits = rows * id_width
        block_partial_terms = sum(sum(coefficient != 0 for coefficient in pattern) for pattern in dictionary)
        dictionary_bits += block_dictionary_bits
        identifier_bits += block_identifier_bits
        offset_bits += 64
        partial_terms += block_partial_terms
        gather_terms += rows  # favorable fused gather-add, one term per row and block
        distinct_total += distinct
        nonzero_total += nonzero
        reconstructed[:, start:stop] = np.asarray(dictionary, dtype=value.dtype)[ids]
        blocks.append(BlockRow(block_index, start, stop, width, distinct, nonzero, id_width, block_dictionary_bits, block_identifier_bits, 64, block_partial_terms, rows, 0))
    reconstruction_mismatches = int(np.count_nonzero(reconstructed != ordered))
    natural = permutation == tuple(range(columns))
    permutation_bits = 0 if natural else columns * ceil_log2(columns)
    permutation_terms = 0 if natural else columns
    scale_bits = rows * 32
    dense_terms = rows * columns + rows
    table_terms = partial_terms + gather_terms + permutation_terms + rows
    dense_bits = rows * columns * 4 + scale_bits
    representation_bits = dictionary_bits + identifier_bits + offset_bits + permutation_bits + scale_bits
    return LocalPatternPlan(rows, columns, block_width, order_name, len(blocks), distinct_total, nonzero_total, dense_terms, table_terms, dense_bits, scale_bits, dictionary_bits, identifier_bits, offset_bits, permutation_bits, representation_bits, representation_bits, reconstruction_mismatches, collisions, tuple(blocks))


def choose_joint_plan(plans: Iterable[LocalPatternPlan]) -> LocalPatternPlan:
    population = tuple(plans)
    if not population:
        raise LocalPatternError("cannot choose from an empty plan population")
    if any(plan.reconstruction_mismatches or plan.hash_collision_mismatches for plan in population):
        raise LocalPatternError("invalid plan cannot be selected")
    return min(population, key=lambda plan: (plan.joint_fraction, plan.operation_fraction + plan.query_byte_fraction + plan.static_representation_fraction, plan.block_width, plan.order_name))
