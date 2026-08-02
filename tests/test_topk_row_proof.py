from __future__ import annotations

import torch

from vortex_runtime.residual_proof import rowwise_residual_block_norms
from vortex_runtime.topk_row_proof import (
    certify_with_exact_topk_rows,
    topk_row_proof_budget,
)


def test_topk_row_certificate_never_accepts_wrong_argmax() -> None:
    generator = torch.Generator().manual_seed(15001)
    for top_k in (1, 3, 8):
        for _ in range(30):
            exact_weight = torch.randn(41, 23, generator=generator)
            residual = 0.06 * torch.randn(41, 23, generator=generator)
            hot_weight = exact_weight - residual
            activation = torch.randn(23, generator=generator)
            hot_logits = hot_weight @ activation
            exact_logits = exact_weight @ activation
            norms = rowwise_residual_block_norms(residual, column_block=4)
            certificate = certify_with_exact_topk_rows(
                hot_logits=hot_logits,
                activation=activation,
                residual=residual,
                residual_norms=norms,
                column_block=4,
                top_k=top_k,
            )
            if certificate.certified:
                assert certificate.candidate == int(torch.argmax(exact_logits).item())


def test_missing_exact_winner_cannot_be_unsafely_certified() -> None:
    hot_logits = torch.tensor([10.0, 9.0, 0.0])
    activation = torch.tensor([1.0, 0.0])
    residual = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ]
    )
    norms = rowwise_residual_block_norms(residual, column_block=1)
    certificate = certify_with_exact_topk_rows(
        hot_logits=hot_logits,
        activation=activation,
        residual=residual,
        residual_norms=norms,
        column_block=1,
        top_k=1,
    )
    assert not certificate.certified


def test_exact_top2_rows_certify_close_competitors() -> None:
    hot_logits = torch.tensor([10.0, 9.9, 0.0])
    activation = torch.tensor([1.0, 0.5])
    residual = torch.tensor(
        [
            [-0.2, 0.0],
            [0.3, 0.0],
            [0.001, 0.001],
        ]
    )
    norms = rowwise_residual_block_norms(residual, column_block=1)
    certificate = certify_with_exact_topk_rows(
        hot_logits=hot_logits,
        activation=activation,
        residual=residual,
        residual_norms=norms,
        column_block=1,
        top_k=2,
    )
    exact_logits = hot_logits + residual @ activation
    assert certificate.certified
    assert certificate.candidate == int(torch.argmax(exact_logits).item()) == 1


def test_405b_top32_lm_head_refinement_is_sub_megabyte() -> None:
    budget = topk_row_proof_budget(
        rows=128_256,
        columns=16_384,
        top_k=32,
        source_bits=16,
        hot_bits=4,
    )
    assert budget.exact_residual_bytes_per_token == 786_432
    assert budget.exact_residual_gib_per_token < 0.001
