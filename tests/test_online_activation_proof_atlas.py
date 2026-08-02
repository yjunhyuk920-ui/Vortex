from __future__ import annotations

import torch

from vortex_runtime.online_activation_proof_atlas import (
    OnlineActivationProofAtlas,
    online_atlas_traffic_budget,
)


def test_expansion_makes_current_activation_exact() -> None:
    generator = torch.Generator().manual_seed(21001)
    residual = torch.randn(31, 17, generator=generator)
    build = torch.randn(5, 17, generator=generator)
    activation = torch.randn(17, generator=generator)
    hot_logits = torch.randn(31, generator=generator)
    atlas = OnlineActivationProofAtlas.from_prompt(
        residual=residual,
        build_activations=build,
        rank=5,
    )
    expanded = atlas.expand(
        activation=activation,
        residual=residual,
    )
    assert expanded
    refined, effects, _, ratio = atlas.apply(
        hot_logits=hot_logits,
        activation=activation,
    )
    exact = hot_logits + residual @ activation
    assert torch.allclose(refined, exact, atol=3e-5, rtol=3e-5)
    assert float(effects.max().item()) < 5e-4
    assert ratio < 3e-6


def test_repeated_direction_reuses_existing_image() -> None:
    generator = torch.Generator().manual_seed(21007)
    residual = torch.randn(23, 13, generator=generator)
    build = torch.randn(4, 13, generator=generator)
    activation = torch.randn(13, generator=generator)
    atlas = OnlineActivationProofAtlas.from_prompt(
        residual=residual,
        build_activations=build,
        rank=4,
    )
    assert atlas.expand(activation=activation, residual=residual)
    rank = atlas.rank
    assert not atlas.expand(activation=activation, residual=residual)
    assert atlas.rank == rank
    assert atlas.expansions == 1


def test_online_certificate_is_sound_after_expansion() -> None:
    generator = torch.Generator().manual_seed(21011)
    for _ in range(20):
        exact_weight = torch.randn(29, 15, generator=generator)
        residual = 0.08 * torch.randn(29, 15, generator=generator)
        hot_weight = exact_weight - residual
        build = torch.randn(5, 15, generator=generator)
        activation = torch.randn(15, generator=generator)
        atlas = OnlineActivationProofAtlas.from_prompt(
            residual=residual,
            build_activations=build,
            rank=5,
        )
        atlas.expand(activation=activation, residual=residual)
        hot_logits = hot_weight @ activation
        certificate = atlas.certify(
            hot_logits=hot_logits,
            activation=activation,
        )
        assert certificate.certified
        assert certificate.candidate == int(torch.argmax(exact_weight @ activation).item())


def test_405b_lm_head_one_expansion_already_exceeds_one_gib() -> None:
    budget = online_atlas_traffic_budget(
        rows=128_256,
        columns=16_384,
        expansions=1,
        tokens=1,
        source_bits=16,
        hot_bits=4,
    )
    assert budget.residual_stream_gib_per_expansion > 2.9
    assert budget.residual_stream_gib_per_expansion < 3.0
