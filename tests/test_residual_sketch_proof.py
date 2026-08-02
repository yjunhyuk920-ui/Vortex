from __future__ import annotations

import torch

from vortex_runtime.residual_sketch_proof import (
    apply_residual_sketch,
    certify_sketch_argmax,
    compile_orthogonal_residual_sketch,
    residual_sketch_budget,
)


def test_sketch_remainder_bound_is_sound() -> None:
    generator = torch.Generator().manual_seed(12001)
    residual = torch.randn(37, 23, generator=generator)
    activation = torch.randn(23, generator=generator)
    hot_logits = torch.randn(37, generator=generator)
    sketch = compile_orthogonal_residual_sketch(
        residual,
        rank=7,
        oversample=3,
        power_iterations=1,
        seed=12007,
    )
    refined, effects, _ = apply_residual_sketch(
        hot_logits=hot_logits,
        activation=activation,
        sketch=sketch,
    )
    exact = hot_logits + residual @ activation
    assert torch.all(torch.abs(exact - refined) <= effects + 2e-5)


def test_sketch_certificate_never_accepts_wrong_argmax() -> None:
    generator = torch.Generator().manual_seed(12011)
    for seed in range(20):
        exact_weight = torch.randn(29, 17, generator=generator)
        residual = 0.08 * torch.randn(29, 17, generator=generator)
        hot_weight = exact_weight - residual
        activation = torch.randn(17, generator=generator)
        sketch = compile_orthogonal_residual_sketch(
            residual,
            rank=5,
            oversample=2,
            power_iterations=1,
            seed=seed,
        )
        hot_logits = hot_weight @ activation
        certificate = certify_sketch_argmax(
            hot_logits=hot_logits,
            activation=activation,
            sketch=sketch,
        )
        if certificate.certified:
            exact_argmax = int(torch.argmax(exact_weight @ activation).item())
            assert certificate.candidate == exact_argmax


def test_full_input_rank_reconstructs_residual() -> None:
    generator = torch.Generator().manual_seed(12017)
    residual = torch.randn(11, 7, generator=generator)
    sketch = compile_orthogonal_residual_sketch(
        residual,
        rank=7,
        oversample=0,
        power_iterations=0,
        seed=12019,
    )
    reconstructed = sketch.coefficients @ sketch.basis.T
    assert torch.allclose(reconstructed, residual, atol=2e-5, rtol=2e-5)
    assert sketch.maximum_row_remainder_norm < 2e-5


def test_more_rank_reduces_remainder() -> None:
    generator = torch.Generator().manual_seed(12023)
    residual = torch.randn(41, 19, generator=generator)
    low = compile_orthogonal_residual_sketch(
        residual,
        rank=3,
        oversample=4,
        power_iterations=1,
        seed=12029,
    )
    high = compile_orthogonal_residual_sketch(
        residual,
        rank=9,
        oversample=4,
        power_iterations=1,
        seed=12029,
    )
    assert high.relative_remainder_l2 <= low.relative_remainder_l2 + 1e-6


def test_405b_lm_head_rank32_metadata_is_small() -> None:
    budget = residual_sketch_budget(
        rows=128_256,
        columns=16_384,
        rank=32,
        metadata_bits=32,
    )
    assert budget.metadata_gib < 0.02
