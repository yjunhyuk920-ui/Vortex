from __future__ import annotations

import torch

from vortex_runtime.activation_proof_atlas import (
    activation_proof_atlas_budget,
    apply_activation_proof_atlas,
    certify_activation_atlas_argmax,
    compile_activation_proof_atlas,
)


def test_atlas_correction_is_exact_on_build_span() -> None:
    generator = torch.Generator().manual_seed(19001)
    residual = torch.randn(37, 13, generator=generator)
    build = torch.randn(7, 13, generator=generator)
    atlas = compile_activation_proof_atlas(
        residual=residual,
        build_activations=build,
        rank=7,
    )
    coordinates = torch.randn(7, generator=generator)
    activation = atlas.basis @ coordinates
    hot_logits = torch.randn(37, generator=generator)
    refined, effects, _, perpendicular_ratio = apply_activation_proof_atlas(
        hot_logits=hot_logits,
        activation=activation,
        atlas=atlas,
    )
    exact = hot_logits + residual @ activation
    assert torch.allclose(refined, exact, atol=2e-5, rtol=2e-5)
    assert float(effects.max().item()) < 2e-4
    assert perpendicular_ratio < 2e-6


def test_atlas_remainder_bounds_are_sound() -> None:
    generator = torch.Generator().manual_seed(19007)
    residual = torch.randn(43, 17, generator=generator)
    build = torch.randn(9, 17, generator=generator)
    activation = torch.randn(17, generator=generator)
    hot_logits = torch.randn(43, generator=generator)
    atlas = compile_activation_proof_atlas(
        residual=residual,
        build_activations=build,
        rank=6,
    )
    refined, effects, _, _ = apply_activation_proof_atlas(
        hot_logits=hot_logits,
        activation=activation,
        atlas=atlas,
    )
    exact = hot_logits + residual @ activation
    assert torch.all(torch.abs(exact - refined) <= effects + 2e-5)


def test_atlas_certificate_never_accepts_wrong_argmax() -> None:
    generator = torch.Generator().manual_seed(19013)
    for seed in range(20):
        exact_weight = torch.randn(29, 15, generator=generator)
        residual = 0.07 * torch.randn(29, 15, generator=generator)
        hot_weight = exact_weight - residual
        build = torch.randn(8, 15, generator=generator)
        activation = torch.randn(15, generator=generator)
        atlas = compile_activation_proof_atlas(
            residual=residual,
            build_activations=build,
            rank=5,
        )
        hot_logits = hot_weight @ activation
        certificate = certify_activation_atlas_argmax(
            hot_logits=hot_logits,
            activation=activation,
            atlas=atlas,
        )
        if certificate.certified:
            exact_argmax = int(torch.argmax(exact_weight @ activation).item())
            assert certificate.candidate == exact_argmax


def test_more_prompt_rank_reduces_build_perpendicular_error() -> None:
    generator = torch.Generator().manual_seed(19021)
    residual = torch.randn(31, 19, generator=generator)
    build = torch.randn(12, 19, generator=generator)
    low = compile_activation_proof_atlas(
        residual=residual,
        build_activations=build,
        rank=4,
    )
    high = compile_activation_proof_atlas(
        residual=residual,
        build_activations=build,
        rank=10,
    )
    assert high.build_mean_perpendicular_ratio <= low.build_mean_perpendicular_ratio
    assert high.build_max_perpendicular_ratio <= low.build_max_perpendicular_ratio


def test_405b_rank16_atlas_is_small() -> None:
    budget = activation_proof_atlas_budget(
        rows=128_256,
        columns=16_384,
        rank=16,
        metadata_bits=32,
    )
    assert budget.metadata_gib < 0.01
