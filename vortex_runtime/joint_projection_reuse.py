"""Exact common-arithmetic accounting across jointly evaluated projections."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


class JointProjectionReuseError(ValueError):
    """Raised when an exact-reuse request is malformed."""


@dataclass(frozen=True)
class CanonicalRow:
    primitive: np.ndarray
    multiplier: int
    is_zero: bool


def canonical_integer_row(row: Any) -> CanonicalRow:
    source = np.asarray(row)
    if source.ndim != 1 or source.size == 0 or source.dtype.kind not in "iu":
        raise JointProjectionReuseError(
            "a nonempty one-dimensional integer row is required"
        )
    values = source.astype(np.int64, copy=False)
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return CanonicalRow(
            primitive=np.zeros(values.shape, dtype=np.int16),
            multiplier=0,
            is_zero=True,
        )
    gcd = int(np.gcd.reduce(np.abs(nonzero)))
    primitive = values // gcd
    first = int(primitive[np.flatnonzero(primitive)[0]])
    sign = 1 if first > 0 else -1
    primitive = primitive * sign
    multiplier = gcd * sign
    if not np.array_equal(primitive * multiplier, values):
        raise JointProjectionReuseError("canonical row reconstruction failed")
    return CanonicalRow(
        primitive=np.ascontiguousarray(primitive.astype(np.int16)),
        multiplier=multiplier,
        is_zero=False,
    )


def row_signature(primitive: np.ndarray) -> bytes:
    source = np.asarray(primitive)
    return (
        source.shape[0].to_bytes(8, "little")
        + source.astype("<i2", copy=False).tobytes()
    )


@dataclass(frozen=True)
class JointRowReusePlan:
    matrix_shapes: tuple[tuple[int, int], ...]
    total_rows: int
    columns: int
    zero_rows: int
    exact_unique_rows: int
    primitive_class_count: int
    reusable_rows: int
    multiplier_corrections: int
    baseline_operations: int
    lower_bound_operations: int
    baseline_storage_bytes: int
    lower_bound_storage_bytes: int
    hash_collision_mismatches: int
    maximum_class_size: int

    @property
    def operation_fraction(self) -> float:
        return self.lower_bound_operations / self.baseline_operations

    @property
    def storage_fraction(self) -> float:
        return self.lower_bound_storage_bytes / self.baseline_storage_bytes

    @property
    def reusable_row_fraction(self) -> float:
        return self.reusable_rows / self.total_rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_shapes": [list(shape) for shape in self.matrix_shapes],
            "total_rows": self.total_rows,
            "columns": self.columns,
            "zero_rows": self.zero_rows,
            "exact_unique_rows": self.exact_unique_rows,
            "primitive_class_count": self.primitive_class_count,
            "reusable_rows": self.reusable_rows,
            "reusable_row_fraction": self.reusable_row_fraction,
            "multiplier_corrections": self.multiplier_corrections,
            "baseline_operations": self.baseline_operations,
            "lower_bound_operations": self.lower_bound_operations,
            "operation_fraction": self.operation_fraction,
            "baseline_storage_bytes": self.baseline_storage_bytes,
            "lower_bound_storage_bytes": self.lower_bound_storage_bytes,
            "storage_fraction": self.storage_fraction,
            "hash_collision_mismatches": self.hash_collision_mismatches,
            "maximum_class_size": self.maximum_class_size,
        }


def analyze_joint_rows(
    matrices: Iterable[Any],
    *,
    bits_per_weight: int = 4,
    scale_bytes_per_row: int = 4,
    mapping_bytes_per_row: int = 4,
    multiplier_bytes_per_row: int = 1,
) -> JointRowReusePlan:
    population = tuple(np.asarray(matrix) for matrix in matrices)
    if len(population) < 2:
        raise JointProjectionReuseError("at least two matrices are required")
    if any(
        matrix.ndim != 2 or matrix.size == 0 for matrix in population
    ):
        raise JointProjectionReuseError(
            "every matrix must be nonempty and two-dimensional"
        )
    if any(matrix.dtype.kind not in "iu" for matrix in population):
        raise JointProjectionReuseError("integer matrices are required")
    columns = int(population[0].shape[1])
    if any(int(matrix.shape[1]) != columns for matrix in population):
        raise JointProjectionReuseError(
            "joint matrices must share the input width"
        )
    if min(
        bits_per_weight,
        scale_bytes_per_row,
        mapping_bytes_per_row,
        multiplier_bytes_per_row,
    ) <= 0:
        raise JointProjectionReuseError("accounting widths must be positive")

    stacked = np.ascontiguousarray(np.concatenate(population, axis=0))
    exact_rows = {row.tobytes() for row in stacked}
    classes: dict[bytes, list[CanonicalRow]] = {}
    zero_rows = 0
    collisions = 0
    for row in stacked:
        canonical = canonical_integer_row(row)
        if canonical.is_zero:
            zero_rows += 1
            continue
        signature = row_signature(canonical.primitive)
        members = classes.setdefault(signature, [])
        if members and not np.array_equal(
            members[0].primitive, canonical.primitive
        ):
            collisions += 1
        members.append(canonical)
    if collisions:
        raise JointProjectionReuseError("canonical signature collision")

    primitive_classes = len(classes)
    total_rows = int(stacked.shape[0])
    nonzero_rows = total_rows - zero_rows
    reusable_rows = sum(
        max(0, len(members) - 1) for members in classes.values()
    )
    maximum_class = max(
        (len(members) for members in classes.values()), default=0
    )

    # Favorable exact integer-domain accounting: one primitive dot product per
    # class and one scalar correction for each nonzero output row. Dense
    # baseline includes one row-scale correction per output row as well.
    baseline_operations = total_rows * columns + total_rows
    lower_bound_operations = primitive_classes * columns + nonzero_rows

    baseline_storage_bytes = (
        math.ceil(total_rows * columns * bits_per_weight / 8)
        + total_rows * scale_bytes_per_row
    )
    lower_bound_storage_bytes = (
        math.ceil(primitive_classes * columns * bits_per_weight / 8)
        + total_rows * scale_bytes_per_row
        + total_rows * mapping_bytes_per_row
        + nonzero_rows * multiplier_bytes_per_row
        + 16
    )
    return JointRowReusePlan(
        matrix_shapes=tuple(
            tuple(int(value) for value in matrix.shape)
            for matrix in population
        ),
        total_rows=total_rows,
        columns=columns,
        zero_rows=zero_rows,
        exact_unique_rows=len(exact_rows),
        primitive_class_count=primitive_classes,
        reusable_rows=reusable_rows,
        multiplier_corrections=nonzero_rows,
        baseline_operations=baseline_operations,
        lower_bound_operations=lower_bound_operations,
        baseline_storage_bytes=baseline_storage_bytes,
        lower_bound_storage_bytes=lower_bound_storage_bytes,
        hash_collision_mismatches=collisions,
        maximum_class_size=maximum_class,
    )


def exact_repeated_block_stats(
    matrices: Sequence[Any], *, block_rows: int
) -> dict[str, int | float]:
    if block_rows <= 0:
        raise JointProjectionReuseError("block_rows must be positive")
    population = tuple(np.asarray(matrix) for matrix in matrices)
    if not population:
        raise JointProjectionReuseError("matrix population is empty")
    columns = int(population[0].shape[1])
    if any(
        matrix.ndim != 2 or int(matrix.shape[1]) != columns
        for matrix in population
    ):
        raise JointProjectionReuseError(
            "block matrices must share the input width"
        )
    blocks: list[bytes] = []
    for matrix in population:
        for start in range(
            0, int(matrix.shape[0]) - block_rows + 1, block_rows
        ):
            block = np.ascontiguousarray(
                matrix[start : start + block_rows]
            )
            blocks.append(block.tobytes())
    unique = len(set(blocks))
    reusable = len(blocks) - unique
    return {
        "block_rows": block_rows,
        "block_count": len(blocks),
        "unique_block_count": unique,
        "reusable_block_count": reusable,
        "reusable_block_fraction": (
            reusable / len(blocks) if blocks else 0.0
        ),
    }
