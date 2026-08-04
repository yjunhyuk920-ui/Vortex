"""Exact output-row grouping and prototype-plus-sparse-delta accounting."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np


class OutputRowStructureError(ValueError):
    """Raised when an exact output-row plan is malformed."""


def _packed_bytes(scalar_count: int, bits: int) -> int:
    if scalar_count < 0 or bits <= 0:
        raise OutputRowStructureError("invalid packed byte request")
    return math.ceil(scalar_count * bits / 8)


def _unsigned_width(maximum_value: int) -> int:
    if maximum_value < 0:
        raise OutputRowStructureError("negative unsigned maximum")
    return max(1, math.ceil(max(1, maximum_value).bit_length() / 8))


def _matrix(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise OutputRowStructureError("weights must be a nonempty rank-two matrix")
    if array.dtype.kind not in "iu":
        raise OutputRowStructureError("weights must use an integer execution dtype")
    return np.ascontiguousarray(array)


def _scales(value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != rows:
        raise OutputRowStructureError("scale count mismatch")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise OutputRowStructureError("scales must be finite and positive")
    return np.ascontiguousarray(array)


def _row_key(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row).tobytes()


def _canonical_row(row: np.ndarray) -> tuple[bytes, int]:
    direct = _row_key(row)
    negated = _row_key(-row)
    return (direct, 1) if direct <= negated else (negated, -1)


def _candidate_rows(source: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0:
        raise OutputRowStructureError("candidate cap must be positive")
    unique, counts = np.unique(source, axis=0, return_counts=True)
    order = sorted(
        range(unique.shape[0]),
        key=lambda index: (
            -int(counts[index]),
            hashlib.sha256(np.ascontiguousarray(unique[index]).tobytes()).digest(),
        ),
    )
    zero = np.zeros((1, source.shape[1]), dtype=source.dtype)
    candidates = [zero[0]]
    for index in order:
        row = unique[index]
        if not np.array_equal(row, zero[0]):
            candidates.append(row)
        if len(candidates) >= cap:
            break
    return np.stack(candidates)


def _distance_table(source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    distances = np.empty((candidates.shape[0], source.shape[0]), dtype=np.int32)
    for index, candidate in enumerate(candidates):
        distances[index] = np.count_nonzero(source != candidate, axis=1)
    return distances


def _frequency_selection(candidates: np.ndarray, count: int) -> tuple[int, ...]:
    return tuple(range(min(count, candidates.shape[0])))


def _greedy_selection(
    distances: np.ndarray, candidates: np.ndarray, count: int
) -> tuple[int, ...]:
    selected: list[int] = []
    best = np.full(distances.shape[1], np.iinfo(np.int32).max, dtype=np.int32)
    remaining = set(range(candidates.shape[0]))
    while remaining and len(selected) < count:
        winner = min(
            remaining,
            key=lambda index: (
                int(np.minimum(best, distances[index]).sum(dtype=np.int64)),
                hashlib.sha256(
                    np.ascontiguousarray(candidates[index]).tobytes()
                ).digest(),
            ),
        )
        selected.append(winner)
        best = np.minimum(best, distances[winner])
        remaining.remove(winner)
    return tuple(selected)


@dataclass(frozen=True)
class OutputRowPlan:
    mechanism: str
    row_count: int
    column_count: int
    bits_per_weight: int
    activation_bytes: int
    has_bias: bool
    prototype_count: int
    duplicate_row_count: int
    negative_row_count: int
    residual_nnz: int
    compile_scalar_comparisons: int
    baseline_operations: int
    candidate_operations: int
    baseline_query_bytes: int
    candidate_query_bytes: int
    static_storage_bytes: int
    reconstruction_mismatches: int

    @property
    def operation_fraction(self) -> float:
        return self.candidate_operations / self.baseline_operations

    @property
    def query_byte_fraction(self) -> float:
        return self.candidate_query_bytes / self.baseline_query_bytes

    @property
    def residual_scalar_fraction(self) -> float:
        return self.residual_nnz / (self.row_count * self.column_count)

    def accounting(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "bits_per_weight": self.bits_per_weight,
            "activation_bytes": self.activation_bytes,
            "has_bias": self.has_bias,
            "prototype_count": self.prototype_count,
            "duplicate_row_count": self.duplicate_row_count,
            "negative_row_count": self.negative_row_count,
            "residual_nnz": self.residual_nnz,
            "residual_scalar_fraction": self.residual_scalar_fraction,
            "compile_scalar_comparisons": self.compile_scalar_comparisons,
            "baseline_operations": self.baseline_operations,
            "candidate_operations": self.candidate_operations,
            "operation_fraction": self.operation_fraction,
            "baseline_query_bytes": self.baseline_query_bytes,
            "candidate_query_bytes": self.candidate_query_bytes,
            "query_byte_fraction": self.query_byte_fraction,
            "static_storage_bytes": self.static_storage_bytes,
            "reconstruction_mismatches": self.reconstruction_mismatches,
        }


def _baseline(
    rows: int,
    columns: int,
    *,
    bits: int,
    activation_bytes: int,
    has_bias: bool,
) -> tuple[int, int]:
    dot_terms = rows * columns
    output_terms = rows + (rows if has_bias else 0)
    operations = dot_terms + output_terms
    query_bytes = (
        _packed_bytes(rows * columns, bits)
        + rows * columns * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    return operations, query_bytes


def _dense_plan(
    source: np.ndarray, *, bits: int, activation_bytes: int, has_bias: bool
) -> OutputRowPlan:
    rows, columns = source.shape
    operations, query_bytes = _baseline(
        rows,
        columns,
        bits=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    storage = (
        _packed_bytes(rows * columns, bits)
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    return OutputRowPlan(
        mechanism="dense",
        row_count=rows,
        column_count=columns,
        bits_per_weight=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
        prototype_count=rows,
        duplicate_row_count=0,
        negative_row_count=0,
        residual_nnz=0,
        compile_scalar_comparisons=0,
        baseline_operations=operations,
        candidate_operations=operations,
        baseline_query_bytes=query_bytes,
        candidate_query_bytes=query_bytes,
        static_storage_bytes=storage,
        reconstruction_mismatches=0,
    )


def _group_plan(
    source: np.ndarray,
    *,
    bits: int,
    activation_bytes: int,
    has_bias: bool,
    sign_canonical: bool,
) -> OutputRowPlan:
    rows, columns = source.shape
    groups: dict[bytes, list[tuple[int, int]]] = {}
    for index, row in enumerate(source):
        if sign_canonical:
            key, polarity = _canonical_row(row)
        else:
            key, polarity = _row_key(row), 1
        groups.setdefault(key, []).append((index, polarity))
    prototypes = len(groups)
    duplicates = rows - prototypes
    negatives = sum(
        polarity < 0 for members in groups.values() for _, polarity in members
    )
    baseline_ops, baseline_bytes = _baseline(
        rows,
        columns,
        bits=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    candidate_ops = (
        prototypes * columns
        + duplicates
        + negatives
        + rows
        + (rows if has_bias else 0)
    )
    group_width = _unsigned_width(max(0, prototypes - 1))
    mapping_bytes = rows * group_width
    polarity_bytes = math.ceil(rows / 8) if sign_canonical else 0
    prototype_weight_bytes = _packed_bytes(prototypes * columns, bits)
    prototype_activation_bytes = prototypes * columns * activation_bytes
    accumulator_reads = rows * 4
    scale_bias_bytes = rows * 4 + (rows * 4 if has_bias else 0)
    candidate_bytes = (
        prototype_weight_bytes
        + prototype_activation_bytes
        + mapping_bytes
        + polarity_bytes
        + accumulator_reads
        + scale_bias_bytes
    )
    static_storage = (
        prototype_weight_bytes
        + mapping_bytes
        + polarity_bytes
        + scale_bias_bytes
    )
    return OutputRowPlan(
        mechanism="sign_canonical_rows" if sign_canonical else "identical_rows",
        row_count=rows,
        column_count=columns,
        bits_per_weight=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
        prototype_count=prototypes,
        duplicate_row_count=duplicates,
        negative_row_count=negatives,
        residual_nnz=0,
        compile_scalar_comparisons=(
            2 * rows * columns if sign_canonical else rows * columns
        ),
        baseline_operations=baseline_ops,
        candidate_operations=candidate_ops,
        baseline_query_bytes=baseline_bytes,
        candidate_query_bytes=candidate_bytes,
        static_storage_bytes=static_storage,
        reconstruction_mismatches=0,
    )


def _prototype_plan(
    source: np.ndarray,
    *,
    bits: int,
    activation_bytes: int,
    has_bias: bool,
    candidates: np.ndarray,
    distances: np.ndarray,
    selected_indices: Sequence[int],
    strategy: str,
) -> OutputRowPlan:
    rows, columns = source.shape
    selected = np.asarray(selected_indices, dtype=np.int64)
    selected_distances = distances[selected]
    assignment = np.argmin(selected_distances, axis=0)
    prototypes = candidates[selected]
    chosen = prototypes[assignment]
    residual = source.astype(np.int32) - chosen.astype(np.int32)
    reconstructed = chosen.astype(np.int32) + residual
    mismatches = int(np.count_nonzero(reconstructed != source.astype(np.int32)))
    residual_nnz = int(np.count_nonzero(residual))
    prototype_count = int(prototypes.shape[0])
    baseline_ops, baseline_bytes = _baseline(
        rows,
        columns,
        bits=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
    )
    candidate_ops = (
        prototype_count * columns
        + residual_nnz
        + rows
        + rows
        + (rows if has_bias else 0)
    )
    group_width = _unsigned_width(max(0, prototype_count - 1))
    column_width = _unsigned_width(max(0, columns - 1))
    residual_value_bytes = max(1, math.ceil((bits + 1) / 8))
    row_pointer_width = _unsigned_width(max(0, residual_nnz))
    mapping_bytes = rows * group_width
    row_pointer_bytes = (rows + 1) * row_pointer_width
    prototype_weight_bytes = _packed_bytes(prototype_count * columns, bits)
    prototype_activation_bytes = prototype_count * columns * activation_bytes
    residual_bytes = residual_nnz * (column_width + residual_value_bytes)
    residual_activation_bytes = residual_nnz * activation_bytes
    accumulator_reads = 2 * rows * 4
    scale_bias_bytes = rows * 4 + (rows * 4 if has_bias else 0)
    candidate_bytes = (
        prototype_weight_bytes
        + prototype_activation_bytes
        + residual_bytes
        + residual_activation_bytes
        + mapping_bytes
        + row_pointer_bytes
        + accumulator_reads
        + scale_bias_bytes
    )
    static_storage = (
        prototype_weight_bytes
        + residual_bytes
        + mapping_bytes
        + row_pointer_bytes
        + scale_bias_bytes
    )
    compile_comparisons = int(candidates.shape[0] * rows * columns)
    return OutputRowPlan(
        mechanism=f"prototype_sparse_delta:{strategy}:k{prototype_count}",
        row_count=rows,
        column_count=columns,
        bits_per_weight=bits,
        activation_bytes=activation_bytes,
        has_bias=has_bias,
        prototype_count=prototype_count,
        duplicate_row_count=0,
        negative_row_count=0,
        residual_nnz=residual_nnz,
        compile_scalar_comparisons=compile_comparisons,
        baseline_operations=baseline_ops,
        candidate_operations=candidate_ops,
        baseline_query_bytes=baseline_bytes,
        candidate_query_bytes=candidate_bytes,
        static_storage_bytes=static_storage,
        reconstruction_mismatches=mismatches,
    )


def compile_output_row_plans(
    weights: Any,
    *,
    scales: Any,
    bits_per_weight: int = 4,
    activation_bytes: int = 4,
    has_bias: bool = True,
    prototype_counts: Iterable[int] = (1, 2, 4, 8),
    candidate_cap: int = 32,
) -> tuple[OutputRowPlan, ...]:
    source = _matrix(weights)
    _scales(scales, source.shape[0])
    if bits_per_weight <= 0 or activation_bytes <= 0:
        raise OutputRowStructureError("bit and activation widths must be positive")
    counts = tuple(sorted({int(value) for value in prototype_counts if int(value) > 0}))
    if not counts:
        raise OutputRowStructureError("prototype counts are empty")
    candidates = _candidate_rows(source, candidate_cap)
    distances = _distance_table(source, candidates)
    plans: list[OutputRowPlan] = [
        _dense_plan(
            source,
            bits=bits_per_weight,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
        ),
        _group_plan(
            source,
            bits=bits_per_weight,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
            sign_canonical=False,
        ),
        _group_plan(
            source,
            bits=bits_per_weight,
            activation_bytes=activation_bytes,
            has_bias=has_bias,
            sign_canonical=True,
        ),
    ]
    for count in counts:
        frequency = _frequency_selection(candidates, count)
        greedy = _greedy_selection(distances, candidates, count)
        plans.append(
            _prototype_plan(
                source,
                bits=bits_per_weight,
                activation_bytes=activation_bytes,
                has_bias=has_bias,
                candidates=candidates,
                distances=distances,
                selected_indices=frequency,
                strategy="frequency",
            )
        )
        plans.append(
            _prototype_plan(
                source,
                bits=bits_per_weight,
                activation_bytes=activation_bytes,
                has_bias=has_bias,
                candidates=candidates,
                distances=distances,
                selected_indices=greedy,
                strategy="greedy",
            )
        )
    return tuple(plans)


def select_output_row_plan(plans: Sequence[OutputRowPlan]) -> OutputRowPlan:
    if not plans:
        raise OutputRowStructureError("plan population is empty")
    if any(plan.reconstruction_mismatches for plan in plans):
        raise OutputRowStructureError("cannot select a non-exact plan")
    return min(
        plans,
        key=lambda plan: (
            plan.operation_fraction,
            plan.query_byte_fraction,
            plan.static_storage_bytes,
            plan.mechanism,
        ),
    )
