"""Exact bit-pattern grouping and accounting for cached attention K/V vectors.

Groups are defined only by identical dtype, shape, and scalar bit patterns.
Approximate similarity, tolerance, quantization, and reordered output reductions
are intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np


class KVEquivalenceError(ValueError):
    """Raised when exact cached-KV grouping or accounting is malformed."""


def unsigned_width(maximum_value: int) -> int:
    if maximum_value < 0:
        raise KVEquivalenceError("unsigned width requires nonnegative input")
    return max(1, math.ceil(max(1, maximum_value).bit_length() / 8))


def _vector_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise KVEquivalenceError("vectors must be a nonempty rank-two array")
    if array.dtype.kind not in "fiu":
        raise KVEquivalenceError("unsupported vector dtype")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise KVEquivalenceError("NaN or infinite cached vectors are rejected")
    return np.ascontiguousarray(array)


def vector_bit_pattern(vector: np.ndarray) -> bytes:
    row = np.ascontiguousarray(vector)
    return (
        row.dtype.str.encode("ascii")
        + b"|"
        + ",".join(str(value) for value in row.shape).encode("ascii")
        + b"|"
        + row.tobytes()
    )


@dataclass(frozen=True)
class ExactGroups:
    group_ids: tuple[int, ...]
    representative_positions: tuple[int, ...]
    group_sizes: tuple[int, ...]
    fingerprint_sha256: tuple[str, ...]

    @property
    def position_count(self) -> int:
        return len(self.group_ids)

    @property
    def unique_count(self) -> int:
        return len(self.representative_positions)

    @property
    def duplicate_count(self) -> int:
        return self.position_count - self.unique_count

    @property
    def duplicate_fraction(self) -> float:
        return self.duplicate_count / self.position_count

    def validate(self) -> None:
        if self.position_count <= 0 or self.unique_count <= 0:
            raise KVEquivalenceError("group population must be nonempty")
        if len(self.group_sizes) != self.unique_count:
            raise KVEquivalenceError("group-size population mismatch")
        if len(self.fingerprint_sha256) != self.unique_count:
            raise KVEquivalenceError("fingerprint population mismatch")
        if any(group < 0 or group >= self.unique_count for group in self.group_ids):
            raise KVEquivalenceError("group identifier outside population")
        if sum(self.group_sizes) != self.position_count:
            raise KVEquivalenceError("group sizes do not cover positions")
        if tuple(sorted(set(self.representative_positions))) != tuple(
            self.representative_positions
        ):
            raise KVEquivalenceError("representatives must be unique and ordered")


def group_exact_vectors(vectors: Any) -> ExactGroups:
    """Group rows by exact dtype/shape/bit pattern, preserving first occurrence."""

    source = _vector_array(vectors)
    lookup: dict[bytes, int] = {}
    group_ids: list[int] = []
    representatives: list[int] = []
    sizes: list[int] = []
    fingerprints: list[str] = []
    for position, vector in enumerate(source):
        pattern = vector_bit_pattern(vector)
        group = lookup.get(pattern)
        if group is None:
            group = len(representatives)
            lookup[pattern] = group
            representatives.append(position)
            sizes.append(0)
            fingerprints.append(hashlib.sha256(pattern).hexdigest())
        group_ids.append(group)
        sizes[group] += 1
    result = ExactGroups(
        group_ids=tuple(group_ids),
        representative_positions=tuple(representatives),
        group_sizes=tuple(sizes),
        fingerprint_sha256=tuple(fingerprints),
    )
    result.validate()
    return result


def group_exact_kv_pairs(keys: Any, values: Any) -> ExactGroups:
    """Group aligned Key/Value rows by exact joint bit pattern."""

    key_array = _vector_array(keys)
    value_array = _vector_array(values)
    if key_array.shape != value_array.shape:
        raise KVEquivalenceError("Key and Value vector shapes differ")
    if key_array.dtype != value_array.dtype:
        raise KVEquivalenceError("Key and Value dtypes differ")
    joined = np.concatenate(
        [
            key_array.view(np.uint8).reshape(key_array.shape[0], -1),
            value_array.view(np.uint8).reshape(value_array.shape[0], -1),
        ],
        axis=1,
    )
    return group_exact_vectors(joined)


@dataclass(frozen=True)
class KVReuseAccounting:
    model_id: str
    prompt_family: str
    phase: str
    decode_step: int
    layer_index: int
    head_index: int
    attention_kind: str
    eligible_length: int
    head_dimension: int
    unique_key_count: int
    unique_kv_count: int
    duplicate_key_count: int
    duplicate_kv_count: int
    dense_qk_multiplications: int
    candidate_qk_multiplications: int
    score_copy_terms: int
    softmax_terms: int
    dense_value_multiplications: int
    candidate_value_multiplications: int
    value_addition_terms: int
    incremental_hash_scan_terms: int
    dense_attention_operation_terms: int
    candidate_attention_operation_terms: int
    dense_cache_bytes: int
    candidate_cache_bytes: int
    metadata_bytes: int

    @property
    def key_duplicate_fraction(self) -> float:
        return self.duplicate_key_count / self.eligible_length

    @property
    def kv_duplicate_fraction(self) -> float:
        return self.duplicate_kv_count / self.eligible_length

    @property
    def attention_operation_fraction(self) -> float:
        return self.candidate_attention_operation_terms / self.dense_attention_operation_terms

    @property
    def attention_query_byte_fraction(self) -> float:
        return (self.candidate_cache_bytes + self.metadata_bytes) / self.dense_cache_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_family": self.prompt_family,
            "phase": self.phase,
            "decode_step": self.decode_step,
            "layer_index": self.layer_index,
            "head_index": self.head_index,
            "attention_kind": self.attention_kind,
            "eligible_length": self.eligible_length,
            "head_dimension": self.head_dimension,
            "unique_key_count": self.unique_key_count,
            "unique_kv_count": self.unique_kv_count,
            "duplicate_key_count": self.duplicate_key_count,
            "duplicate_kv_count": self.duplicate_kv_count,
            "key_duplicate_fraction": self.key_duplicate_fraction,
            "kv_duplicate_fraction": self.kv_duplicate_fraction,
            "dense_qk_multiplications": self.dense_qk_multiplications,
            "candidate_qk_multiplications": self.candidate_qk_multiplications,
            "score_copy_terms": self.score_copy_terms,
            "softmax_terms": self.softmax_terms,
            "dense_value_multiplications": self.dense_value_multiplications,
            "candidate_value_multiplications": self.candidate_value_multiplications,
            "value_addition_terms": self.value_addition_terms,
            "incremental_hash_scan_terms": self.incremental_hash_scan_terms,
            "dense_attention_operation_terms": self.dense_attention_operation_terms,
            "candidate_attention_operation_terms": self.candidate_attention_operation_terms,
            "attention_operation_fraction": self.attention_operation_fraction,
            "dense_cache_bytes": self.dense_cache_bytes,
            "candidate_cache_bytes": self.candidate_cache_bytes,
            "metadata_bytes": self.metadata_bytes,
            "attention_query_byte_fraction": self.attention_query_byte_fraction,
        }


def account_kv_reuse(
    *,
    model_id: str,
    prompt_family: str,
    phase: str,
    decode_step: int,
    layer_index: int,
    head_index: int,
    attention_kind: str,
    keys: Any,
    values: Any,
) -> tuple[KVReuseAccounting, ExactGroups, ExactGroups]:
    """Group eligible K/V vectors and compute favorable exact-reuse accounting."""

    key_array = _vector_array(keys)
    value_array = _vector_array(values)
    if key_array.shape != value_array.shape or key_array.dtype != value_array.dtype:
        raise KVEquivalenceError("eligible Key/Value arrays are not aligned")
    length, dimension = (int(value) for value in key_array.shape)
    key_groups = group_exact_vectors(key_array)
    kv_groups = group_exact_kv_pairs(key_array, value_array)
    if any(
        kv_groups.group_ids[left] == kv_groups.group_ids[right]
        and key_groups.group_ids[left] != key_groups.group_ids[right]
        for left in range(length)
        for right in range(left + 1, length)
    ):
        raise KVEquivalenceError("joint KV group is not a subset of Key groups")

    dense_qk = dimension * length
    candidate_qk = dimension * key_groups.unique_count
    score_copies = key_groups.duplicate_count
    softmax = length
    dense_value_multiplies = dimension * length
    candidate_value_multiplies = dimension * kv_groups.unique_count
    value_additions = dimension * length
    # Incremental maintenance scans only the newly appended K and V vectors.
    hash_scan = 2 * dimension
    dense_operations = dense_qk + softmax + dense_value_multiplies + value_additions
    candidate_operations = (
        candidate_qk
        + score_copies
        + softmax
        + candidate_value_multiplies
        + value_additions
        + hash_scan
    )
    scalar_bytes = int(key_array.dtype.itemsize)
    dense_cache_bytes = 2 * dimension * length * scalar_bytes
    # Favorable candidate reads each unique K and unique KV representative once,
    # plus the newly appended K/V vectors used for incremental exact grouping.
    candidate_cache_bytes = (
        dimension
        * (key_groups.unique_count + kv_groups.unique_count + 2)
        * scalar_bytes
    )
    key_group_width = unsigned_width(max(0, key_groups.unique_count - 1))
    kv_group_width = unsigned_width(max(0, kv_groups.unique_count - 1))
    position_width = unsigned_width(max(0, length - 1))
    metadata = (
        length * (key_group_width + kv_group_width)
        + (key_groups.unique_count + kv_groups.unique_count) * (32 + position_width)
    )
    accounting = KVReuseAccounting(
        model_id=model_id,
        prompt_family=prompt_family,
        phase=phase,
        decode_step=decode_step,
        layer_index=layer_index,
        head_index=head_index,
        attention_kind=attention_kind,
        eligible_length=length,
        head_dimension=dimension,
        unique_key_count=key_groups.unique_count,
        unique_kv_count=kv_groups.unique_count,
        duplicate_key_count=key_groups.duplicate_count,
        duplicate_kv_count=kv_groups.duplicate_count,
        dense_qk_multiplications=dense_qk,
        candidate_qk_multiplications=candidate_qk,
        score_copy_terms=score_copies,
        softmax_terms=softmax,
        dense_value_multiplications=dense_value_multiplies,
        candidate_value_multiplications=candidate_value_multiplies,
        value_addition_terms=value_additions,
        incremental_hash_scan_terms=hash_scan,
        dense_attention_operation_terms=dense_operations,
        candidate_attention_operation_terms=candidate_operations,
        dense_cache_bytes=dense_cache_bytes,
        candidate_cache_bytes=candidate_cache_bytes,
        metadata_bytes=metadata,
    )
    return accounting, key_groups, kv_groups


def combine_whole_model_accounting(
    *,
    linear_dense_operations: int,
    linear_dense_q4_bytes: int,
    rows: Sequence[KVReuseAccounting],
) -> dict[str, int | float]:
    items = tuple(rows)
    if not items:
        raise KVEquivalenceError("KV accounting is empty")
    if linear_dense_operations < 0 or linear_dense_q4_bytes < 0:
        raise KVEquivalenceError("Linear accounting cannot be negative")
    dense_attention_ops = sum(row.dense_attention_operation_terms for row in items)
    candidate_attention_ops = sum(
        row.candidate_attention_operation_terms for row in items
    )
    dense_attention_bytes = sum(row.dense_cache_bytes for row in items)
    candidate_attention_bytes = sum(
        row.candidate_cache_bytes + row.metadata_bytes for row in items
    )
    dense_ops = linear_dense_operations + dense_attention_ops
    candidate_ops = linear_dense_operations + candidate_attention_ops
    dense_bytes = linear_dense_q4_bytes + dense_attention_bytes
    candidate_bytes = linear_dense_q4_bytes + candidate_attention_bytes
    return {
        "linear_dense_operations": linear_dense_operations,
        "linear_dense_q4_bytes": linear_dense_q4_bytes,
        "dense_attention_operations": dense_attention_ops,
        "candidate_attention_operations": candidate_attention_ops,
        "dense_attention_bytes": dense_attention_bytes,
        "candidate_attention_bytes": candidate_attention_bytes,
        "dense_whole_model_operations": dense_ops,
        "candidate_whole_model_operations": candidate_ops,
        "whole_model_operation_fraction": candidate_ops / dense_ops,
        "dense_whole_model_bytes": dense_bytes,
        "candidate_whole_model_bytes": candidate_bytes,
        "whole_model_query_byte_fraction": candidate_bytes / dense_bytes,
    }
