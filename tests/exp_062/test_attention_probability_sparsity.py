from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.attention_probability_sparsity import (
    AttentionProbabilitySparsityError,
    account_attention_probabilities,
    causal_eligible_mask,
    combine_whole_model_accounting,
    structural_eligible_mask,
    zero_skipped_value_accumulation,
)


def causal_probabilities(length: int) -> torch.Tensor:
    rows = []
    for query in range(length):
        row = torch.zeros(length, dtype=torch.float32)
        row[: query + 1] = 1.0 / (query + 1)
        rows.append(row)
    return torch.stack(rows).reshape(1, 1, length, length)


def local_probabilities(length: int, window: int) -> torch.Tensor:
    rows = []
    for query in range(length):
        row = torch.zeros(length, dtype=torch.float32)
        start = max(0, query - window + 1)
        count = query - start + 1
        row[start : query + 1] = 1.0 / count
        rows.append(row)
    return torch.stack(rows).reshape(1, 1, length, length)


def test_causal_mask_shape_and_population() -> None:
    mask = causal_eligible_mask(
        torch=torch,
        batch_size=1,
        head_count=2,
        query_length=3,
        key_length=3,
        past_length=0,
        device=torch.device("cpu"),
    )
    assert tuple(mask.shape) == (1, 2, 3, 3)
    assert int(mask.sum().item()) == 12


def test_local_window_excludes_old_structural_entries() -> None:
    mask = structural_eligible_mask(
        torch=torch,
        batch_size=1,
        head_count=1,
        query_length=4,
        key_length=4,
        past_length=0,
        device=torch.device("cpu"),
        local_window_size=2,
    )
    assert int(mask.sum().item()) == 7
    probabilities = local_probabilities(4, 2)
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="local",
        phase="prefill",
        decode_step=0,
        layer_index=0,
        head_dimension=8,
        past_length=0,
        attention_kind="local",
        local_window_size=2,
    )
    assert row.eligible_probability_count == 7
    assert row.structural_masked_probability_count == 9
    assert row.exact_nonmask_zero_count == 0


def test_rejects_malformed_key_and_past_lengths() -> None:
    with pytest.raises(AttentionProbabilitySparsityError):
        causal_eligible_mask(
            torch=torch,
            batch_size=1,
            head_count=1,
            query_length=2,
            key_length=4,
            past_length=1,
            device=torch.device("cpu"),
        )


def test_mask_zeros_are_excluded_from_savings() -> None:
    probabilities = causal_probabilities(4)
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="p",
        phase="prefill",
        decode_step=0,
        layer_index=0,
        head_dimension=8,
        past_length=0,
    )
    assert row.structural_masked_probability_count == 6
    assert row.eligible_probability_count == 10
    assert row.exact_nonmask_zero_count == 0


def test_exact_unmasked_underflow_zero_is_detected() -> None:
    logits = torch.tensor([[[[0.0, -200.0, -300.0]]]], dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    assert int((probabilities == 0).sum().item()) >= 1
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="underflow",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_dimension=16,
        past_length=2,
    )
    assert row.exact_nonmask_zero_count >= 1
    assert row.sparse_value_operation_terms < row.dense_value_operation_terms


def test_moderate_logits_have_no_false_zero() -> None:
    probabilities = torch.softmax(
        torch.tensor([[[[0.0, -2.0, 1.0]]]], dtype=torch.float32), dim=-1
    )
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="moderate",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_dimension=4,
        past_length=2,
    )
    assert row.exact_nonmask_zero_count == 0


def test_negative_or_nonfinite_probability_is_rejected() -> None:
    negative = torch.tensor([[[[1.1, -0.1]]]], dtype=torch.float32)
    with pytest.raises(AttentionProbabilitySparsityError):
        account_attention_probabilities(
            negative,
            model_id="m",
            prompt_family="bad",
            phase="warm_decode",
            decode_step=2,
            layer_index=0,
            head_dimension=4,
            past_length=1,
        )
    nonfinite = torch.tensor([[[[float("nan")]]]], dtype=torch.float32)
    with pytest.raises(AttentionProbabilitySparsityError):
        account_attention_probabilities(
            nonfinite,
            model_id="m",
            prompt_family="bad",
            phase="warm_decode",
            decode_step=2,
            layer_index=0,
            head_dimension=4,
            past_length=0,
        )


def test_nonzero_structurally_masked_entry_is_rejected() -> None:
    probabilities = causal_probabilities(2)
    probabilities[0, 0, 0, 1] = 0.1
    with pytest.raises(AttentionProbabilitySparsityError):
        account_attention_probabilities(
            probabilities,
            model_id="m",
            prompt_family="bad_mask",
            phase="prefill",
            decode_step=0,
            layer_index=0,
            head_dimension=4,
            past_length=0,
        )


def test_zero_skipped_value_accumulation_matches_dense() -> None:
    dense, sparse = zero_skipped_value_accumulation(
        probabilities=[0.5, 0.0, -0.0, 0.5],
        values=[[1.0, 2.0], [8.0, 9.0], [5.0, 6.0], [3.0, 4.0]],
    )
    assert dense == sparse == (2.0, 3.0)


def test_attention_accounting_charges_qk_softmax_and_scan() -> None:
    probabilities = torch.tensor([[[[0.5, 0.5, 0.0, 0.0]]]], dtype=torch.float32)
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=3,
        layer_index=0,
        head_dimension=8,
        past_length=3,
    )
    assert row.eligible_probability_count == 4
    assert row.exact_nonmask_zero_count == 2
    assert row.qk_operation_terms == 32
    assert row.softmax_operation_terms == 4
    assert row.dense_value_operation_terms == 32
    assert row.sparse_value_operation_terms == 16
    assert row.probability_scan_terms == 4
    assert row.dense_attention_operation_terms == 68
    assert row.sparse_attention_operation_terms == 56


def test_whole_model_accounting_retains_unchanged_linear_work() -> None:
    probabilities = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    row = account_attention_probabilities(
        probabilities,
        model_id="m",
        prompt_family="p",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_dimension=4,
        past_length=1,
    )
    combined = combine_whole_model_accounting(
        linear_dense_operations=10_000,
        linear_dense_q4_bytes=2_000,
        attention_rows=[row],
    )
    assert combined["dense_whole_model_operations"] > 10_000
    assert combined["whole_model_operation_fraction"] > 0.99
    assert math.isfinite(float(combined["whole_model_query_byte_fraction"]))
