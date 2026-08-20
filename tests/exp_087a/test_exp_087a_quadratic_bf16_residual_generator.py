from __future__ import annotations

import pytest
import torch

from vortex_runtime.quadratic_bf16_residual_generator import (
    QuadraticResidualError,
    bf16_words,
    gf2_rank,
    quadratic_features,
    solve_gf2,
    synthetic_quadratic_control,
    target_resources,
    words_to_bf16,
)


def test_bf16_word_roundtrip() -> None:
    tensor = torch.tensor(
        [[0.0, -0.0, 1.0, -2.5, 0.10009765625, 31.75]],
        dtype=torch.bfloat16,
    )
    restored = words_to_bf16(bf16_words(tensor))
    assert torch.equal(restored.view(torch.int16), tensor.view(torch.int16))


def test_quadratic_feature_count_and_rank() -> None:
    bits = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=torch.uint8,
    )
    features = quadratic_features(bits)
    assert features.shape == (5, 7)
    assert gf2_rank(features) == 5


def test_gf2_solver_replays_exact_targets() -> None:
    generator = torch.Generator().manual_seed(20260820)
    bits = torch.randint(0, 2, (16, 6), generator=generator, dtype=torch.uint8)
    features = quadratic_features(bits)
    source = torch.randint(
        0, 2, (features.shape[1], 80), generator=generator, dtype=torch.uint8
    )
    targets = torch.remainder(
        features.to(torch.int16) @ source.to(torch.int16), 2
    ).to(torch.uint8)
    fitted, rank, consistent = solve_gf2(features, targets)
    replay = torch.remainder(
        features.to(torch.int16) @ fitted.to(torch.int16), 2
    ).to(torch.uint8)
    assert consistent
    assert rank > 0
    assert torch.equal(replay, targets)


def test_synthetic_quadratic_control() -> None:
    control = synthetic_quadratic_control()
    assert control["consistent"]
    assert control["mismatches"] == 0
    assert control["rank"] > 0


def test_target_resource_shape_passes_early_gate() -> None:
    resources = target_resources(program_count=4, context_bits=12)
    assert resources["feature_count"] == 79
    assert resources["sidecar_gib"] < 4.0
    assert resources["compiled_mlp_operation_fraction"] < 0.011851851851851851
    assert resources["whole_model_operation_fraction_if_only_mlp_replaced"] > 0.1


def test_non_bf16_words_fail_closed() -> None:
    with pytest.raises(QuadraticResidualError):
        bf16_words(torch.ones((2, 3), dtype=torch.float32))
    with pytest.raises(QuadraticResidualError):
        words_to_bf16(torch.ones((2, 3), dtype=torch.float32))
