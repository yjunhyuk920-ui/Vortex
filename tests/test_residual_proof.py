from __future__ import annotations

import torch

from vortex_runtime.residual_proof import (
    certify_linear_argmax,
    residual_metadata_budget,
    residual_tile_norms,
    rowwise_residual_block_norms,
    rowwise_residual_effect_bounds,
    tiled_bilinear_residual_bound,
)


def test_tiled_bilinear_bound_is_sound_and_tightens_with_finer_tiles() -> None:
    generator = torch.Generator().manual_seed(9001)
    residual = torch.randn(12, 16, generator=generator)
    left = torch.randn(12, generator=generator)
    right = torch.randn(16, generator=generator)
    exact = abs(float((left @ residual @ right).item()))

    coarse_norms = residual_tile_norms(
        residual,
        row_block=12,
        column_block=16,
    )
    fine_norms = residual_tile_norms(
        residual,
        row_block=3,
        column_block=4,
    )
    coarse = tiled_bilinear_residual_bound(
        left=left,
        right=right,
        tile_norms=coarse_norms,
        row_block=12,
        column_block=16,
    )
    fine = tiled_bilinear_residual_bound(
        left=left,
        right=right,
        tile_norms=fine_norms,
        row_block=3,
        column_block=4,
    )

    assert exact <= fine + 1e-5
    assert fine <= coarse + 1e-5


def test_rowwise_effect_bounds_dominate_exact_residual_output() -> None:
    generator = torch.Generator().manual_seed(9007)
    residual = torch.randn(17, 23, generator=generator)
    activation = torch.randn(23, generator=generator)
    norms = rowwise_residual_block_norms(residual, column_block=5)
    bounds = rowwise_residual_effect_bounds(
        residual_norms=norms,
        activation=activation,
        column_block=5,
    )
    exact = torch.abs(residual @ activation)
    assert torch.all(exact <= bounds + 1e-5)


def test_argmax_certificate_never_accepts_wrong_candidate() -> None:
    generator = torch.Generator().manual_seed(9011)
    for _ in range(40):
        exact_weight = torch.randn(31, 19, generator=generator)
        residual = 0.02 * torch.randn(31, 19, generator=generator)
        hot_weight = exact_weight - residual
        activation = torch.randn(19, generator=generator)
        hot_logits = hot_weight @ activation
        exact_logits = exact_weight @ activation
        norms = rowwise_residual_block_norms(residual, column_block=4)
        certificate = certify_linear_argmax(
            approximate_logits=hot_logits,
            activation=activation,
            residual_norms=norms,
            column_block=4,
        )
        if certificate.certified:
            assert certificate.candidate == int(torch.argmax(exact_logits).item())


def test_certificate_accepts_large_margin_case() -> None:
    hot_weight = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    residual = torch.full_like(hot_weight, 0.001)
    activation = torch.tensor([1.0, 0.5, -0.25, 0.125])
    norms = rowwise_residual_block_norms(residual, column_block=2)
    certificate = certify_linear_argmax(
        approximate_logits=hot_weight @ activation,
        activation=activation,
        residual_norms=norms,
        column_block=2,
    )
    assert certificate.certified
    assert certificate.candidate == 0
    assert certificate.certified_margin > 0


def test_405b_lm_head_metadata_is_small() -> None:
    budget = residual_metadata_budget(
        rows=128_256,
        columns=16_384,
        column_block=256,
        metadata_bits=16,
    )
    assert budget.blocks_per_row == 64
    assert budget.metadata_gib < 0.02
