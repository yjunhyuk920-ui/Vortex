from __future__ import annotations

import pytest
import torch

from vortex_runtime.one_sweep_true_token_rank import (
    TrueTokenRankInvariantError,
    candidate_id_bytes,
    longest_common_prefix,
    model_free_controls,
    rank_histogram,
    rank_metrics,
    stable_argmax_token,
    stable_true_token_rank,
    static_topk_gate,
)


def test_rank_matches_descending_logit_and_ascending_token_ties() -> None:
    logits = torch.tensor([7.0, 7.0, 6.0, 7.0], dtype=torch.float32)
    assert [stable_true_token_rank(logits, token) for token in range(4)] == [
        1,
        2,
        4,
        3,
    ]
    assert stable_argmax_token(logits) == 0


def test_rank_rejects_nonfinite_and_bad_tokens() -> None:
    with pytest.raises(TrueTokenRankInvariantError):
        stable_true_token_rank(torch.tensor([1.0, float("nan")]), 0)
    with pytest.raises(TrueTokenRankInvariantError):
        stable_true_token_rank(torch.tensor([1.0, 2.0]), 2)
    with pytest.raises(TrueTokenRankInvariantError):
        stable_true_token_rank(torch.ones(2, 2), 0)


def test_metrics_and_histogram_are_exact() -> None:
    ranks = [1, 1, 4, 16]
    metrics = rank_metrics(ranks)
    assert metrics.count == 4
    assert metrics.minimum == 1
    assert metrics.maximum == 16
    assert metrics.top1_fraction == 0.5
    assert metrics.top4_fraction == 0.75
    assert metrics.top16_fraction == 1.0
    assert metrics.static_path_information_bits == pytest.approx(6.0)
    assert rank_histogram(ranks) == [
        {"rank": 1, "count": 2},
        {"rank": 4, "count": 1},
        {"rank": 16, "count": 1},
    ]


def test_static_gate_requires_both_population_and_worst_case() -> None:
    passed = static_topk_gate(
        [1] * 96 + [4] * 26 + [16] * 6,
        p95_limit=4,
        maximum_limit=16,
    )
    assert passed["passed"]
    failed = static_topk_gate(
        [1] * 120 + [17] * 8,
        p95_limit=4,
        maximum_limit=16,
    )
    assert not failed["passed"]
    assert failed["p95_passed"]
    assert not failed["maximum_passed"]


def test_prefix_and_candidate_bytes() -> None:
    assert longest_common_prefix([1, 2, 3], [1, 2, 9]) == 2
    assert candidate_id_bytes(
        block_length=128, maximum_candidates=16, token_bytes=2
    ) == 4096


def test_model_free_controls_pass() -> None:
    controls = model_free_controls()
    assert controls["passed"]
    assert controls["candidate_id_bytes"] == 4096
