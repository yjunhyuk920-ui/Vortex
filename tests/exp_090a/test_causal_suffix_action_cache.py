from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.causal_suffix_action_cache import (
    SuffixActionInvariantError,
    bf16_roundtrip,
    longest_common_prefix,
    predict_suffix_action,
    project_target_hot_state,
    quantize_rowwise_symmetric_int4,
    select_predictor_mode,
    stable_argmax,
    stable_token_rank,
    synthetic_carry_control,
    synthetic_nearest_control,
    tensor_bytes,
    verification_weight_fraction,
)


def frozen_target() -> dict[str, int | float | str]:
    return {
        "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "revision": "f9801cba95a53242b3cc928a4a418d12571d1c5f",
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 126,
        "vocab_size": 128256,
        "max_context": 131072,
        "hot_limit_bytes": 8589934592,
        "whole_model_fraction_limit": 0.011851851851851851,
        "history_entries": 64,
        "checkpoint_compression_ratio_favorable": 1.261972,
    }


def test_q4_quantization_is_deterministic_and_accounted() -> None:
    source = torch.tensor(
        [[-3.0, -1.0, 0.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    first = quantize_rowwise_symmetric_int4(source)
    second = quantize_rowwise_symmetric_int4(source.clone())
    assert first.manifest.to_dict() == second.manifest.to_dict()
    assert tensor_bytes(first.quantized) == tensor_bytes(second.quantized)
    assert first.manifest.packed_weight_bytes == 5
    assert first.manifest.scale_bytes == 4
    assert first.manifest.artifact_bytes == 9
    assert first.quantized.min().item() >= -8
    assert first.quantized.max().item() <= 7
    logits = first.logits(torch.ones(1, 5, dtype=torch.bfloat16))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_q4_rejects_wrong_shape_or_range() -> None:
    with pytest.raises(SuffixActionInvariantError):
        quantize_rowwise_symmetric_int4(torch.tensor([1.0]))
    with pytest.raises(SuffixActionInvariantError):
        quantize_rowwise_symmetric_int4(torch.ones(2, 2), minimum=-7)


def test_bf16_roundtrip_is_finite_and_idempotent() -> None:
    source = torch.tensor([0.1, -5.25, 1024.5], dtype=torch.float32)
    first = bf16_roundtrip(source)
    second = bf16_roundtrip(first)
    assert tensor_bytes(first) == tensor_bytes(second)
    assert torch.isfinite(first).all()


def test_stable_rank_and_argmax_use_token_id_tie_rule() -> None:
    scores = torch.tensor([2.0, 5.0, 5.0, 1.0], dtype=torch.float32)
    assert stable_argmax(scores) == 1
    assert stable_token_rank(scores, 1) == 1
    assert stable_token_rank(scores, 2) == 2
    assert stable_token_rank(scores, 0) == 3


def test_longest_common_prefix_stops_at_first_mismatch() -> None:
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([], [1]) == 0


def test_frozen_predictor_formulas() -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    residual = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=torch.bfloat16
    )
    query = torch.tensor([0.0, 1.0], dtype=torch.float32)

    raw, raw_index = predict_suffix_action(
        "raw_q4",
        step_offset=0,
        query_hidden=query,
        history_hidden=hidden,
        history_residual=residual,
    )
    assert torch.count_nonzero(raw) == 0
    assert raw_index is None

    carry, carry_index = predict_suffix_action(
        "carry_bf16",
        step_offset=7,
        query_hidden=query,
        history_hidden=hidden,
        history_residual=residual,
    )
    assert tensor_bytes(carry) == tensor_bytes(residual[-1].float())
    assert carry_index == 1

    secant, secant_index = predict_suffix_action(
        "secant_bf16",
        step_offset=1,
        query_hidden=query,
        history_hidden=hidden,
        history_residual=residual,
    )
    expected = bf16_roundtrip(residual[-1].float() + 2 * (residual[-1].float() - residual[-2].float()))
    assert tensor_bytes(secant) == tensor_bytes(expected)
    assert secant_index == 1

    nearest, nearest_index = predict_suffix_action(
        "nearest_history_bf16",
        step_offset=0,
        query_hidden=query,
        history_hidden=hidden,
        history_residual=residual,
    )
    assert nearest_index == 1
    assert tensor_bytes(nearest) == tensor_bytes(residual[1].float())


def test_synthetic_positive_controls_pass() -> None:
    assert synthetic_carry_control()["passed"]
    assert synthetic_nearest_control()["passed"]


def test_build_selection_is_lexicographic_and_deterministic() -> None:
    rows = [
        {"mode": "raw_q4", "accepted_length": 2, "true_path_ranks": [1, 50]},
        {"mode": "raw_q4", "accepted_length": 3, "true_path_ranks": [2, 3]},
        {"mode": "carry_bf16", "accepted_length": 5, "true_path_ranks": [1, 8]},
        {"mode": "carry_bf16", "accepted_length": 5, "true_path_ranks": [1, 7]},
        {"mode": "nearest_history_bf16", "accepted_length": 5, "true_path_ranks": [1, 4]},
        {"mode": "nearest_history_bf16", "accepted_length": 5, "true_path_ranks": [1, 5]},
    ]
    order = ["raw_q4", "carry_bf16", "nearest_history_bf16"]
    selected = select_predictor_mode(rows, order)
    assert selected.mode == "nearest_history_bf16"
    assert selected.minimum_accepted_length == 5


def test_target_hot_projection_matches_frozen_equation() -> None:
    projection = project_target_hot_state(frozen_target())
    assert projection.layer_parameter_count == 3_187_671_040
    assert projection.layer_bf16_bytes == 6_375_342_080
    assert projection.total_hot_bytes == 7_981_689_344
    assert projection.total_hot_gib == pytest.approx(7.43352746963501)
    assert projection.margin_bytes == 608_245_248
    assert projection.hot_limit_passed


def test_verification_fraction_requires_long_acceptance() -> None:
    assert verification_weight_fraction(128) == pytest.approx(0.0078125)
    assert verification_weight_fraction(128, compression_ratio=1.261972) == pytest.approx(
        0.006190707876244481
    )
    assert verification_weight_fraction(64) > 0.011851851851851851
    assert math.isinf(verification_weight_fraction(0))
