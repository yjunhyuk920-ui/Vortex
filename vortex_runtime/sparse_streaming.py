"""Exact scalar/run/block sparse streaming plans for integer Q4 matrices.

Only source scalars equal to zero may be skipped.  Every stored value, index,
row pointer, nonzero-block slot, and padded edge slot is charged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


class SparseStreamingError(ValueError):
    """Raised when an exact sparse streaming request is malformed."""


def _matrix(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.size == 0:
        raise SparseStreamingError("matrix must be nonempty and two-dimensional")
    if source.dtype.kind not in "iu":
        raise SparseStreamingError("sparse streaming requires integer weights")
    return source.astype(np.int64, copy=False)


def packed_bytes(scalar_count: int, bits_per_scalar: int) -> int:
    if scalar_count < 0 or bits_per_scalar <= 0:
        raise SparseStreamingError("invalid packed scalar accounting")
    return math.ceil(scalar_count * bits_per_scalar / 8)


def unsigned_width(maximum_value: int) -> int:
    if maximum_value < 0:
        raise SparseStreamingError("unsigned width requires a nonnegative maximum")
    return max(1, math.ceil(max(1, maximum_value).bit_length() / 8))


@dataclass(frozen=True)
class SparseFormatPlan:
    kind: str
    rows: int
    columns: int
    bits_per_scalar: int
    operation_terms: int
    encoded_bytes: int
    stored_value_slots: int
    metadata_bytes: int
    nonzero_scalar_count: int
    zero_scalar_count: int
    block_shape: tuple[int, int] | None = None
    payload: Any | None = None

    @property
    def direct_scalar_terms(self) -> int:
        return self.rows * self.columns

    @property
    def dense_packed_bytes(self) -> int:
        return packed_bytes(self.direct_scalar_terms, self.bits_per_scalar)

    @property
    def operation_fraction(self) -> float:
        return self.operation_terms / self.direct_scalar_terms

    @property
    def query_byte_fraction(self) -> float:
        return self.encoded_bytes / self.dense_packed_bytes

    def accounting(self) -> dict[str, int | float | str | list[int] | None]:
        return {
            "kind": self.kind,
            "rows": self.rows,
            "columns": self.columns,
            "block_shape": list(self.block_shape) if self.block_shape else None,
            "direct_scalar_terms": self.direct_scalar_terms,
            "dense_packed_bytes": self.dense_packed_bytes,
            "operation_terms": self.operation_terms,
            "operation_fraction": self.operation_fraction,
            "encoded_bytes": self.encoded_bytes,
            "query_byte_fraction": self.query_byte_fraction,
            "stored_value_slots": self.stored_value_slots,
            "metadata_bytes": self.metadata_bytes,
            "nonzero_scalar_count": self.nonzero_scalar_count,
            "zero_scalar_count": self.zero_scalar_count,
            "zero_scalar_fraction": self.zero_scalar_count / self.direct_scalar_terms,
        }

    def reconstruct(self) -> np.ndarray:
        if self.payload is None:
            raise SparseStreamingError("plan was compiled without materialized payload")
        result = np.zeros((self.rows, self.columns), dtype=np.int64)
        if self.kind == "dense":
            return np.asarray(self.payload, dtype=np.int64).reshape(self.rows, self.columns)
        if self.kind == "csr":
            for row, column, value in self.payload:
                result[int(row), int(column)] = int(value)
            return result
        if self.kind == "row_runs":
            for row, start, values in self.payload:
                end = int(start) + len(values)
                result[int(row), int(start):end] = np.asarray(values, dtype=np.int64)
            return result
        if self.kind.startswith("bsr_"):
            if self.block_shape is None:
                raise SparseStreamingError("BSR plan is missing a block shape")
            block_rows, block_columns = self.block_shape
            for block_row, block_column, values in self.payload:
                block = np.asarray(values, dtype=np.int64).reshape(
                    block_rows, block_columns
                )
                row_start = int(block_row) * block_rows
                column_start = int(block_column) * block_columns
                row_end = min(self.rows, row_start + block_rows)
                column_end = min(self.columns, column_start + block_columns)
                result[row_start:row_end, column_start:column_end] = block[
                    : row_end - row_start, : column_end - column_start
                ]
            return result
        raise SparseStreamingError(f"unknown sparse plan kind: {self.kind}")


def compile_dense(
    matrix: np.ndarray, *, bits_per_scalar: int = 4, materialize: bool = False
) -> SparseFormatPlan:
    source = _matrix(matrix)
    rows, columns = source.shape
    count = rows * columns
    nonzero = int(np.count_nonzero(source))
    return SparseFormatPlan(
        kind="dense",
        rows=rows,
        columns=columns,
        bits_per_scalar=bits_per_scalar,
        operation_terms=count,
        encoded_bytes=packed_bytes(count, bits_per_scalar),
        stored_value_slots=count,
        metadata_bytes=0,
        nonzero_scalar_count=nonzero,
        zero_scalar_count=count - nonzero,
        payload=tuple(int(value) for value in source.ravel()) if materialize else None,
    )


def compile_csr(
    matrix: np.ndarray, *, bits_per_scalar: int = 4, materialize: bool = False
) -> SparseFormatPlan:
    source = _matrix(matrix)
    rows, columns = source.shape
    coordinates = np.argwhere(source != 0)
    nonzero = int(coordinates.shape[0])
    column_width = unsigned_width(max(0, columns - 1))
    pointer_width = unsigned_width(nonzero)
    metadata = nonzero * column_width + (rows + 1) * pointer_width
    payload = None
    if materialize:
        payload = tuple(
            (int(row), int(column), int(source[row, column]))
            for row, column in coordinates
        )
    return SparseFormatPlan(
        kind="csr",
        rows=rows,
        columns=columns,
        bits_per_scalar=bits_per_scalar,
        operation_terms=nonzero,
        encoded_bytes=packed_bytes(nonzero, bits_per_scalar) + metadata,
        stored_value_slots=nonzero,
        metadata_bytes=metadata,
        nonzero_scalar_count=nonzero,
        zero_scalar_count=rows * columns - nonzero,
        payload=payload,
    )


def compile_row_runs(
    matrix: np.ndarray, *, bits_per_scalar: int = 4, materialize: bool = False
) -> SparseFormatPlan:
    source = _matrix(matrix)
    rows, columns = source.shape
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    nonzero = int(np.count_nonzero(source))
    for row in range(rows):
        column = 0
        while column < columns:
            while column < columns and source[row, column] == 0:
                column += 1
            if column == columns:
                break
            start = column
            values: list[int] = []
            while column < columns and source[row, column] != 0:
                values.append(int(source[row, column]))
                column += 1
            runs.append((row, start, tuple(values)))
    run_count = len(runs)
    index_width = unsigned_width(max(0, columns - 1))
    length_width = unsigned_width(columns)
    pointer_width = unsigned_width(run_count)
    metadata = run_count * (index_width + length_width) + (rows + 1) * pointer_width
    return SparseFormatPlan(
        kind="row_runs",
        rows=rows,
        columns=columns,
        bits_per_scalar=bits_per_scalar,
        operation_terms=nonzero,
        encoded_bytes=packed_bytes(nonzero, bits_per_scalar) + metadata,
        stored_value_slots=nonzero,
        metadata_bytes=metadata,
        nonzero_scalar_count=nonzero,
        zero_scalar_count=rows * columns - nonzero,
        payload=tuple(runs) if materialize else None,
    )


def compile_bsr(
    matrix: np.ndarray,
    *,
    block_shape: tuple[int, int],
    bits_per_scalar: int = 4,
    materialize: bool = False,
) -> SparseFormatPlan:
    source = _matrix(matrix)
    rows, columns = source.shape
    block_rows, block_columns = block_shape
    if block_rows <= 0 or block_columns <= 0:
        raise SparseStreamingError("BSR block dimensions must be positive")
    grid_rows = math.ceil(rows / block_rows)
    grid_columns = math.ceil(columns / block_columns)
    blocks: list[tuple[int, int, tuple[int, ...]]] = []
    for block_row in range(grid_rows):
        row_start = block_row * block_rows
        row_end = min(rows, row_start + block_rows)
        for block_column in range(grid_columns):
            column_start = block_column * block_columns
            column_end = min(columns, column_start + block_columns)
            visible = source[row_start:row_end, column_start:column_end]
            if not np.any(visible):
                continue
            padded = np.zeros((block_rows, block_columns), dtype=np.int64)
            padded[: row_end - row_start, : column_end - column_start] = visible
            blocks.append(
                (block_row, block_column, tuple(int(value) for value in padded.ravel()))
            )
    block_count = len(blocks)
    slots_per_block = block_rows * block_columns
    stored_slots = block_count * slots_per_block
    block_column_width = unsigned_width(max(0, grid_columns - 1))
    pointer_width = unsigned_width(block_count)
    metadata = block_count * block_column_width + (grid_rows + 1) * pointer_width
    nonzero = int(np.count_nonzero(source))
    return SparseFormatPlan(
        kind=f"bsr_{block_rows}x{block_columns}",
        rows=rows,
        columns=columns,
        bits_per_scalar=bits_per_scalar,
        operation_terms=stored_slots,
        encoded_bytes=packed_bytes(stored_slots, bits_per_scalar) + metadata,
        stored_value_slots=stored_slots,
        metadata_bytes=metadata,
        nonzero_scalar_count=nonzero,
        zero_scalar_count=rows * columns - nonzero,
        block_shape=block_shape,
        payload=tuple(blocks) if materialize else None,
    )


DEFAULT_BLOCK_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 4),
    (1, 8),
    (4, 4),
    (8, 8),
    (16, 16),
)


def compile_registered_sparse_formats(
    matrix: np.ndarray,
    *,
    bits_per_scalar: int = 4,
    block_shapes: Sequence[tuple[int, int]] = DEFAULT_BLOCK_SHAPES,
    materialize: bool = False,
) -> tuple[SparseFormatPlan, ...]:
    source = _matrix(matrix)
    plans = [
        compile_dense(source, bits_per_scalar=bits_per_scalar, materialize=materialize),
        compile_csr(source, bits_per_scalar=bits_per_scalar, materialize=materialize),
        compile_row_runs(source, bits_per_scalar=bits_per_scalar, materialize=materialize),
    ]
    plans.extend(
        compile_bsr(
            source,
            block_shape=tuple(shape),
            bits_per_scalar=bits_per_scalar,
            materialize=materialize,
        )
        for shape in block_shapes
    )
    return tuple(plans)


def select_favorable_sparse_format(
    plans: Sequence[SparseFormatPlan],
) -> SparseFormatPlan:
    items = tuple(plans)
    expected = {"dense", "csr", "row_runs"} | {
        f"bsr_{rows}x{columns}" for rows, columns in DEFAULT_BLOCK_SHAPES
    }
    if {item.kind for item in items} != expected:
        raise SparseStreamingError("registered sparse format population mismatch")
    return min(
        items,
        key=lambda plan: (
            plan.operation_fraction,
            plan.query_byte_fraction,
            plan.encoded_bytes,
            plan.kind,
        ),
    )
