from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.kv_equivalence import (
    KVEquivalenceError,
    account_kv_reuse,
    combine_whole_model_accounting,
    group_exact_kv_pairs,
    group_exact_vectors,
)


def test_exact_key_duplicates_are_grouped() -> None:
    keys = np.asarray([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], dtype=np.float32)
    groups = group_exact_vectors(keys)
    assert groups.unique_count == 2
    assert groups.duplicate_count == 1
    assert groups.group_ids == (0, 1, 0)


def test_exact_kv_duplicates_are_grouped() -> None:
    keys = np.asarray([[1, 2], [1, 2], [1, 2]], dtype=np.int16)
    values = np.asarray([[3, 4], [5, 6], [3, 4]], dtype=np.int16)
    groups = group_exact_kv_pairs(keys, values)
    assert groups.unique_count == 2
    assert groups.group_ids == (0, 1, 0)


def test_one_bit_key_difference_prevents_key_group() -> None:
    first = np.asarray([1.0], dtype=np.float32)
    second = first.view(np.uint32).copy()
    second[0] ^= np.uint32(1)
    keys = np.vstack([first, second.view(np.float32)])
    assert group_exact_vectors(keys).unique_count == 2


def test_one_bit_value_difference_prevents_kv_group_only() -> None:
    keys = np.asarray([[1.0], [1.0]], dtype=np.float32)
    values = np.asarray([[2.0], [2.0]], dtype=np.float32)
    changed = values[1].view(np.uint32)
    changed[0] ^= np.uint32(1)
    assert group_exact_vectors(keys).unique_count == 1
    assert group_exact_kv_pairs(keys, values).unique_count == 2


def test_positive_and_negative_zero_are_distinct_bit_patterns() -> None:
    vectors = np.asarray([[0.0], [-0.0]], dtype=np.float32)
    assert group_exact_vectors(vectors).unique_count == 2


def test_nan_or_infinity_is_rejected() -> None:
    with pytest.raises(KVEquivalenceError):
        group_exact_vectors(np.asarray([[float("nan")]], dtype=np.float32))
    with pytest.raises(KVEquivalenceError):
        group_exact_vectors(np.asarray([[float("inf")]], dtype=np.float32))


def test_group_construction_preserves_first_position_order() -> None:
    vectors = np.asarray([[2], [1], [2], [3], [1]], dtype=np.int8)
    groups = group_exact_vectors(vectors)
    assert groups.representative_positions == (0, 1, 3)
    assert groups.group_sizes == (2, 2, 1)


def test_accounting_reuses_key_and_kv_work() -> None:
    keys = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values = np.asarray([[5.0, 6.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    row, key_groups, kv_groups = account_kv_reuse(
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=3,
        layer_index=0,
        head_index=0,
        attention_kind="global",
        keys=keys,
        values=values,
    )
    assert key_groups.unique_count == 2
    assert kv_groups.unique_count == 2
    assert row.candidate_qk_multiplications == 4
    assert row.candidate_value_multiplications == 4
    assert row.score_copy_terms == 1
    assert row.value_addition_terms == 6


def test_no_duplicates_can_exceed_dense_after_incremental_hashing() -> None:
    keys = np.arange(16, dtype=np.float32).reshape(4, 4)
    values = keys + 100.0
    row, _, _ = account_kv_reuse(
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=4,
        layer_index=0,
        head_index=0,
        attention_kind="global",
        keys=keys,
        values=values,
    )
    assert row.key_duplicate_fraction == 0.0
    assert row.kv_duplicate_fraction == 0.0
    assert row.attention_operation_fraction > 1.0
    assert row.attention_query_byte_fraction > 1.0


def test_whole_model_accounting_retains_linear_work() -> None:
    keys = np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    values = np.asarray([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)
    row, _, _ = account_kv_reuse(
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_index=0,
        attention_kind="global",
        keys=keys,
        values=values,
    )
    combined = combine_whole_model_accounting(
        linear_dense_operations=10000,
        linear_dense_q4_bytes=2000,
        rows=[row],
    )
    assert combined["dense_whole_model_operations"] > 10000
    assert combined["whole_model_operation_fraction"] > 0.9
