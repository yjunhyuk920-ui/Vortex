from __future__ import annotations

import pytest
import torch

from vortex_runtime.online_residual_affine_hull import (
    ResidualAffineHullInvariantError,
    accepted_prefix,
    best_single_residual_rank,
    causal_affine_transport,
    model_free_controls,
    project_target_hot_bytes,
    residual_span_projection,
    stable_argmax64,
    stable_nearest_indices,
    stable_true_token_rank,
    summarize_ranks,
)


def test_model_free_controls_pass() -> None:
    controls = model_free_controls()
    assert controls["passed"] is True


def test_nearest_neighbor_tie_is_lowest_index() -> None:
    bank = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([1.0, 0.0])
    assert stable_nearest_indices(bank, query, 2) == (0, 1)


def test_affine_transport_is_exact_on_identical_hidden_row() -> None:
    hidden = torch.eye(4, dtype=torch.float64)
    residual = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    coarse = torch.zeros(5, dtype=torch.float64)
    result = causal_affine_transport(
        coarse_logits=coarse,
        residual_bank=residual,
        hidden_bank=hidden,
        query_hidden=hidden[2],
        neighbor_count=1,
        ridge_lambda=2.0**-10,
        coefficient_clip=4.0,
    )
    assert result.neighbor_indices == (2,)
    assert result.coefficients == pytest.approx((1.0,))
    assert torch.equal(result.scores, residual[2])


def test_span_projection_reconstructs_row_span_and_rejects_orthogonal() -> None:
    bank = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    inside = torch.tensor([[3.0, -2.0, 3.0, -2.0]], dtype=torch.float64)
    positive = residual_span_projection(
        bank, inside, relative_eigenvalue_cutoff=2.0**-40
    )
    assert positive.effective_rank == 2
    assert torch.equal(positive.projected_residual, inside)
    assert positive.projection_residual_l2 == 0.0

    outside = torch.tensor([[1.0, 0.0, -1.0, 0.0]], dtype=torch.float64)
    negative = residual_span_projection(
        bank, outside, relative_eigenvalue_cutoff=2.0**-40
    )
    assert negative.projection_residual_l2 > 0.0
    assert not torch.equal(negative.projected_residual, outside)


def test_single_residual_oracle_and_stable_rank() -> None:
    coarse = torch.tensor([0.0, 0.0, 0.0])
    bank = torch.tensor([[0.0, 4.0, 0.0], [3.0, 0.0, 0.0]])
    rank, index = best_single_residual_rank(coarse, bank, 1)
    assert (rank, index) == (1, 0)
    tie = torch.tensor([5.0, 5.0, 4.0])
    assert stable_argmax64(tie) == 0
    assert stable_true_token_rank(tie, 1) == 2


def test_rank_summary_and_prefix() -> None:
    summary = summarize_ranks([1, 2, 4, 16, 32])
    assert summary.maximum == 32
    assert summary.top1_fraction == pytest.approx(0.2)
    assert summary.top4_fraction == pytest.approx(0.6)
    assert summary.top16_fraction == pytest.approx(0.8)
    assert accepted_prefix([1, 2, 9, 4], [1, 2, 3, 4]) == 2


def test_hot_projection_charges_hidden_and_gram_banks() -> None:
    report = project_target_hot_bytes(
        inherited_hot_bytes=1000,
        block_length=128,
        vocabulary_size=256,
        hidden_size=64,
    )
    assert report["float64_residual_bank_bytes"] == 128 * 256 * 8
    assert report["float64_hidden_bank_bytes"] == 128 * 64 * 8
    assert report["float64_gram_bytes"] == 128 * 128 * 8
    assert report["total_hot_bytes"] == 1000 + 128 * 64 * 8 + 128 * 128 * 8


def test_nonfinite_input_fails_closed() -> None:
    with pytest.raises(ResidualAffineHullInvariantError):
        residual_span_projection(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            relative_eigenvalue_cutoff=2.0**-40,
        )
