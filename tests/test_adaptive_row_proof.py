from __future__ import annotations

import torch

from vortex_runtime.adaptive_row_proof import (
    adaptive_row_proof_budget,
    certify_with_adaptive_exact_rows,
)
from vortex_runtime.residual_proof import rowwise_residual_block_norms


def test_adaptive_proof_never_certifies_wrong_argmax() -> None:
    generator = torch.Generator().manual_seed(17001)
    for _ in range(30):
        exact_weight = torch.randn(53, 29, generator=generator)
        residual = 0.08 * torch.randn(53, 29, generator=generator)
        hot_weight = exact_weight - residual
        activation = torch.randn(29, generator=generator)
        hot_logits = hot_weight @ activation
        exact_logits = exact_weight @ activation
        norms = rowwise_residual_block_norms(residual, column_block=5)
        certificate = certify_with_adaptive_exact_rows(
            hot_logits=hot_logits,
            activation=activation,
            residual=residual,
            residual_norms=norms,
            column_block=5,
            initial_top_k=2,
            refinement_batch=4,
            max_refined_rows=20,
        )
        if certificate.certified:
            assert certificate.candidate == int(torch.argmax(exact_logits).item())


def test_all_rows_eventually_certify_exact_argmax() -> None:
    generator = torch.Generator().manual_seed(17011)
    exact_weight = torch.randn(31, 17, generator=generator)
    residual = 0.2 * torch.randn(31, 17, generator=generator)
    hot_weight = exact_weight - residual
    activation = torch.randn(17, generator=generator)
    hot_logits = hot_weight @ activation
    exact_logits = exact_weight @ activation
    norms = rowwise_residual_block_norms(residual, column_block=4)
    certificate = certify_with_adaptive_exact_rows(
        hot_logits=hot_logits,
        activation=activation,
        residual=residual,
        residual_norms=norms,
        column_block=4,
        initial_top_k=1,
        refinement_batch=3,
        max_refined_rows=31,
    )
    assert certificate.certified
    assert certificate.candidate == int(torch.argmax(exact_logits).item())
    assert certificate.refined_row_count <= 31


def test_large_margin_case_refines_only_initial_row() -> None:
    hot_logits = torch.tensor([20.0, 1.0, 0.0, -1.0])
    residual = torch.full((4, 3), 0.001)
    activation = torch.tensor([1.0, 0.5, -0.25])
    norms = rowwise_residual_block_norms(residual, column_block=1)
    certificate = certify_with_adaptive_exact_rows(
        hot_logits=hot_logits,
        activation=activation,
        residual=residual,
        residual_norms=norms,
        column_block=1,
        initial_top_k=1,
        refinement_batch=1,
        max_refined_rows=4,
    )
    assert certificate.certified
    assert certificate.refined_row_count == 1


def test_refinement_limit_can_return_sound_abstention() -> None:
    hot_logits = torch.tensor([10.0, 9.9, 9.8, 0.0])
    residual = torch.tensor(
        [
            [-0.4, 0.0],
            [0.5, 0.0],
            [0.6, 0.0],
            [0.0, 0.0],
        ]
    )
    activation = torch.tensor([1.0, 0.0])
    norms = rowwise_residual_block_norms(residual, column_block=1)
    certificate = certify_with_adaptive_exact_rows(
        hot_logits=hot_logits,
        activation=activation,
        residual=residual,
        residual_norms=norms,
        column_block=1,
        initial_top_k=1,
        refinement_batch=1,
        max_refined_rows=1,
    )
    assert not certificate.certified
    assert certificate.ambiguous_rows_remaining > 0


def test_405b_4096_rows_cost_below_point_one_gib() -> None:
    budget = adaptive_row_proof_budget(
        columns=16_384,
        refined_rows=4_096,
        source_bits=16,
        hot_bits=4,
    )
    assert budget.residual_gib == 0.09375
