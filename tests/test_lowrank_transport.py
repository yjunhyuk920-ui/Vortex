from __future__ import annotations

import torch

from vortex_runtime.lowrank_transport import (
    fit_randomized_low_rank_residual,
    low_rank_corrected_linear,
    materialize_low_rank_correction,
)


def test_randomized_residual_recovers_exact_rank_two_signal() -> None:
    generator = torch.Generator().manual_seed(19)
    base = torch.randn(24, 18, generator=generator)
    left_true = torch.randn(24, 2, generator=generator)
    right_true = torch.randn(18, 2, generator=generator)
    target = base + left_true @ right_true.T

    left, right, stats = fit_randomized_low_rank_residual(
        target_weight=target,
        base_weight=base,
        rank=2,
        oversample=2,
        power_iterations=1,
        seed=23,
        factor_bits=16,
    )
    corrected = materialize_low_rank_correction(
        base_weight=base,
        left=left,
        right=right,
    )
    relative_error = torch.linalg.vector_norm(target - corrected) / torch.linalg.vector_norm(target)
    assert float(relative_error.item()) < 2e-4
    assert stats.corrected_relative_l2_error < stats.baseline_relative_l2_error
    assert stats.relative_error_reduction > 0.999
    assert stats.factor_elements == (24 + 18) * 2


def test_fp8_factors_reduce_general_residual_error() -> None:
    generator = torch.Generator().manual_seed(29)
    base = torch.randn(30, 20, generator=generator)
    target = base + 0.2 * torch.randn(30, 20, generator=generator)
    _, _, stats = fit_randomized_low_rank_residual(
        target_weight=target,
        base_weight=base,
        rank=4,
        oversample=4,
        power_iterations=1,
        seed=31,
        factor_bits=8,
    )
    assert stats.corrected_relative_l2_error < stats.baseline_relative_l2_error
    assert stats.factor_bytes == (30 + 20) * 4


def test_factorized_execution_matches_materialized_weight() -> None:
    generator = torch.Generator().manual_seed(37)
    base = torch.randn(9, 7, generator=generator)
    left = torch.randn(9, 3, generator=generator)
    right = torch.randn(7, 3, generator=generator)
    bias = torch.randn(9, generator=generator)
    x = torch.randn(2, 5, 7, generator=generator)
    materialized = materialize_low_rank_correction(
        base_weight=base,
        left=left,
        right=right,
    )
    expected = torch.nn.functional.linear(x, materialized, bias)
    actual = low_rank_corrected_linear(
        x,
        base_weight=base,
        left=left,
        right=right,
        bias=bias,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
