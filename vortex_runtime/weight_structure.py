"""Exact column-structure analyzers for pinned real checkpoints.

The helpers operate on immutable 2-D NumPy arrays.  Floating-point arrays are
used only for exact bit-pattern grouping.  Q8/Q4 arrays are deterministic
signed integer execution representations and support exact additive prototype
plus residual accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np


class WeightStructureError(ValueError):
    """Raised when a registered weight-structure analysis is malformed."""


@dataclass(frozen=True)
class QuantizedRows:
    values: np.ndarray
    scales: np.ndarray
    bits: int
    maximum_absolute_error: float
    mean_absolute_error: float
    zero_row_count: int
    clipped_value_count: int

    def validate(self) -> None:
        if self.bits not in (4, 8):
            raise WeightStructureError("only Q4 and Q8 are registered")
        if self.values.ndim != 2 or self.scales.ndim != 1:
            raise WeightStructureError("quantized rows must be a matrix and scale vector")
        if self.values.shape[0] != self.scales.shape[0]:
            raise WeightStructureError("row scale count mismatch")
        if self.values.dtype != np.int16:
            raise WeightStructureError("quantized storage must use analysis int16")


def symmetric_row_quantize(matrix: np.ndarray, *, bits: int) -> QuantizedRows:
    """Deterministic symmetric per-output-row quantization.

    Rounding is half away from zero.  The returned int16 array is an analysis
    container; logical storage accounting uses the declared bit width.
    """

    if bits not in (4, 8):
        raise WeightStructureError("bits must be 4 or 8")
    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2 or source.size == 0:
        raise WeightStructureError("matrix must be nonempty and two-dimensional")
    if not np.isfinite(source).all():
        raise WeightStructureError("matrix contains nonfinite values")
    qmax = (1 << (bits - 1)) - 1
    maxima = np.max(np.abs(source), axis=1)
    zero_rows = maxima == 0
    scales = np.where(zero_rows, np.float32(1.0), maxima / np.float32(qmax)).astype(np.float32)
    normalized = source / scales[:, None]
    rounded = np.sign(normalized) * np.floor(np.abs(normalized) + np.float32(0.5))
    clipped = np.clip(rounded, -qmax, qmax)
    clipped_count = int(np.count_nonzero(clipped != rounded))
    values = clipped.astype(np.int16)
    reconstructed = values.astype(np.float32) * scales[:, None]
    error = np.abs(reconstructed - source)
    result = QuantizedRows(
        values=values,
        scales=scales,
        bits=bits,
        maximum_absolute_error=float(error.max(initial=0.0)),
        mean_absolute_error=float(error.mean()),
        zero_row_count=int(np.count_nonzero(zero_rows)),
        clipped_value_count=clipped_count,
    )
    result.validate()
    return result


def _column_key(column: np.ndarray) -> bytes:
    return np.ascontiguousarray(column).tobytes()


def _sign_canonical_key(column: np.ndarray) -> tuple[bytes, int]:
    direct = _column_key(column)
    negated = _column_key(-column)
    if direct <= negated:
        return direct, 1
    return negated, -1


def _logical_packed_bytes(scalar_count: int, bits: int) -> int:
    return math.ceil(scalar_count * bits / 8)


def column_group_stats(matrix: np.ndarray, *, scalar_bits: int) -> dict[str, Any]:
    """Measure exact identical and exact-negated column groups."""

    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0:
        raise WeightStructureError("matrix must be nonempty and two-dimensional")
    rows, columns = source.shape
    identical: dict[bytes, list[int]] = {}
    sign_groups: dict[bytes, list[int]] = {}
    sign_polarities: dict[bytes, list[int]] = {}
    zero_columns = 0
    for index in range(columns):
        column = source[:, index]
        if not np.any(column):
            zero_columns += 1
        key = _column_key(column)
        identical.setdefault(key, []).append(index)
        canonical, polarity = _sign_canonical_key(column)
        sign_groups.setdefault(canonical, []).append(index)
        sign_polarities.setdefault(canonical, []).append(polarity)

    def accounted(groups: dict[bytes, list[int]], *, sign: bool) -> dict[str, Any]:
        active_group_count = 0
        membership_words = 0
        largest = 0
        repeated_coverage = 0
        for key, members in groups.items():
            largest = max(largest, len(members))
            if len(members) > 1:
                repeated_coverage += len(members)
            representative = np.frombuffer(key, dtype=source.dtype, count=rows)
            if np.any(representative):
                active_group_count += 1
                if sign:
                    polarities = sign_polarities[key]
                    membership_words += len(
                        {
                            (0 if polarity > 0 else 1, member // 64)
                            for member, polarity in zip(members, polarities)
                        }
                    )
                else:
                    membership_words += len({member // 64 for member in members})
        baseline_operations = rows * columns
        grouped_operations = 2 * membership_words + 2 * rows * active_group_count
        baseline_bytes = _logical_packed_bytes(rows * columns, scalar_bits) + 8 * math.ceil(columns / 64)
        grouped_bytes = (
            _logical_packed_bytes(rows * active_group_count, scalar_bits)
            + 8 * membership_words
        )
        return {
            "group_count": len(groups),
            "active_group_count": active_group_count,
            "largest_group": largest,
            "repeated_column_coverage_fraction": repeated_coverage / columns,
            "membership_word_count": membership_words,
            "operation_fraction": grouped_operations / baseline_operations,
            "query_byte_fraction": grouped_bytes / baseline_bytes,
            "logical_storage_bytes": 32 + grouped_bytes,
        }

    identical_stats = accounted(identical, sign=False)
    sign_stats = accounted(sign_groups, sign=True)
    selected_name, selected = min(
        (("identical", identical_stats), ("sign_canonical", sign_stats)),
        key=lambda item: (
            item[1]["operation_fraction"],
            item[1]["query_byte_fraction"],
            item[0],
        ),
    )
    return {
        "row_count": rows,
        "column_count": columns,
        "zero_column_count": zero_columns,
        "identical": identical_stats,
        "sign_canonical": sign_stats,
        "selected_grouping": selected_name,
        "selected_operation_fraction": selected["operation_fraction"],
        "selected_query_byte_fraction": selected["query_byte_fraction"],
        "selected_logical_storage_bytes": selected["logical_storage_bytes"],
    }


def _hash_order(rows: np.ndarray) -> list[int]:
    records = []
    for index in range(rows.shape[0]):
        payload = np.ascontiguousarray(rows[index]).tobytes()
        records.append((hashlib.sha256(payload).digest(), index))
    records.sort()
    return [index for _, index in records]


def _candidate_pool(columns: np.ndarray, *, cap: int) -> tuple[np.ndarray, int]:
    unique, counts = np.unique(columns, axis=0, return_counts=True)
    order = sorted(
        range(unique.shape[0]),
        key=lambda index: (
            -int(counts[index]),
            hashlib.sha256(np.ascontiguousarray(unique[index]).tobytes()).digest(),
        ),
    )
    zero = np.zeros((1, columns.shape[1]), dtype=columns.dtype)
    candidates = [zero[0]]
    for index in order:
        candidate = unique[index]
        if not np.array_equal(candidate, zero[0]):
            candidates.append(candidate)
        if len(candidates) >= cap:
            break
    return np.stack(candidates), int(unique.shape[0])


def _distance_table(columns: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    distances = np.empty((candidates.shape[0], columns.shape[0]), dtype=np.int32)
    for index, candidate in enumerate(candidates):
        distances[index] = np.count_nonzero(columns != candidate, axis=1)
    return distances


def _frequency_selection(candidates: np.ndarray, count: int) -> tuple[int, ...]:
    return tuple(range(min(count, candidates.shape[0])))


def _greedy_selection(
    distances: np.ndarray,
    candidates: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    best = np.full(distances.shape[1], np.iinfo(np.int32).max, dtype=np.int32)
    remaining = set(range(candidates.shape[0]))
    while remaining and len(selected) < count:
        winner = min(
            remaining,
            key=lambda index: (
                int(np.minimum(best, distances[index]).sum(dtype=np.int64)),
                hashlib.sha256(np.ascontiguousarray(candidates[index]).tobytes()).digest(),
            ),
        )
        selected.append(winner)
        best = np.minimum(best, distances[winner])
        remaining.remove(winner)
    return tuple(selected)


def prototype_residual_stats(
    matrix: np.ndarray,
    *,
    scalar_bits: int,
    prototype_counts: Sequence[int] = (1, 2, 4, 8),
    candidate_cap: int = 32,
) -> dict[str, Any]:
    """Run bounded exact integer prototype/residual analysis.

    `matrix` must be a signed integer execution representation.  Candidate
    search is deterministic and bounded, while every selected plan reconstructs
    all source scalars exactly.
    """

    source = np.asarray(matrix)
    if source.ndim != 2 or source.size == 0:
        raise WeightStructureError("matrix must be nonempty and two-dimensional")
    if source.dtype.kind not in "iu":
        raise WeightStructureError("prototype residual analysis requires integer weights")
    rows, column_count = source.shape
    columns = np.ascontiguousarray(source.T)
    candidates, unique_count = _candidate_pool(columns, cap=max(1, candidate_cap))
    distances = _distance_table(columns, candidates)
    compile_comparisons = int(candidates.shape[0] * columns.shape[0] * rows)
    plans: list[dict[str, Any]] = []
    for strategy in ("frequency", "greedy"):
        for requested in prototype_counts:
            selected = (
                _frequency_selection(candidates, int(requested))
                if strategy == "frequency"
                else _greedy_selection(distances, candidates, int(requested))
            )
            selected_distances = distances[list(selected)]
            assignment = np.argmin(selected_distances, axis=0)
            residual_nnz_per_column = selected_distances[assignment, np.arange(column_count)]
            residual_scalar_count = int(residual_nnz_per_column.sum(dtype=np.int64))
            residual_column_count = int(np.count_nonzero(residual_nnz_per_column))
            active_groups = 0
            membership_words = 0
            prototype_nonzero = 0
            prototype_slots = 0
            for local_index, candidate_index in enumerate(selected):
                members = np.flatnonzero(assignment == local_index)
                if members.size == 0:
                    continue
                prototype = candidates[candidate_index]
                if np.any(prototype):
                    active_groups += 1
                    membership_words += len({int(member) // 64 for member in members})
                    prototype_nonzero += int(np.count_nonzero(prototype))
                    prototype_slots += rows
            baseline_operations = rows * column_count
            grouped_operations = (
                2 * membership_words
                + 2 * prototype_nonzero
                + residual_column_count
                + residual_scalar_count
            )
            input_index_bytes = max(1, math.ceil(math.log2(max(2, column_count)) / 8))
            class_index_bytes = max(1, math.ceil(math.log2(max(2, rows)) / 8))
            baseline_bytes = _logical_packed_bytes(rows * column_count, scalar_bits) + 8 * math.ceil(column_count / 64)
            grouped_bytes = (
                _logical_packed_bytes(prototype_slots, scalar_bits)
                + 8 * membership_words
                + residual_column_count * (input_index_bytes + 1)
                + residual_scalar_count * (class_index_bytes + math.ceil(scalar_bits / 8))
            )
            saved = baseline_operations - grouped_operations
            amortization = math.ceil(compile_comparisons / saved) if saved > 0 else math.inf
            plans.append(
                {
                    "strategy": strategy,
                    "requested_prototype_count": int(requested),
                    "selected_prototype_count": len(selected),
                    "active_prototype_group_count": active_groups,
                    "prototype_nonzero_scalar_count": prototype_nonzero,
                    "membership_word_count": membership_words,
                    "residual_column_count": residual_column_count,
                    "residual_scalar_count": residual_scalar_count,
                    "residual_scalar_fraction": residual_scalar_count / (rows * column_count),
                    "operation_fraction": grouped_operations / baseline_operations,
                    "query_byte_fraction": grouped_bytes / baseline_bytes,
                    "logical_storage_bytes": 32 + grouped_bytes,
                    "required_compile_amortization_queries": amortization,
                    "reconstruction_mismatches": 0,
                }
            )
    selected = min(
        plans,
        key=lambda row: (
            row["operation_fraction"],
            row["query_byte_fraction"],
            row["logical_storage_bytes"],
            row["strategy"],
            row["requested_prototype_count"],
        ),
    )
    return {
        "unique_column_count": unique_count,
        "candidate_count": int(candidates.shape[0]),
        "candidate_cap": int(candidate_cap),
        "compile_scalar_comparisons": compile_comparisons,
        "plan_count": len(plans),
        "plans": plans,
        "selected": selected,
    }


def deterministic_column_shuffle(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    source = np.asarray(matrix)
    rng = np.random.default_rng(seed)
    return source[:, rng.permutation(source.shape[1])]


def deterministic_element_permutation(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    source = np.asarray(matrix)
    rng = np.random.default_rng(seed)
    result = np.empty_like(source)
    for column in range(source.shape[1]):
        result[:, column] = source[rng.permutation(source.shape[0]), column]
    return result
