from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from vortex_runtime.gated_projected_linear import (
    GatedProjectedLinear,
    activation_basis,
)


def test_activation_basis_spans_low_rank_samples() -> None:
    torch.manual_seed(4)
    directions = torch.randn(12, 3)
    coefficients = torch.randn(20, 3)
    samples = coefficients @ directions.T
    basis = activation_basis(samples, rank=3)
    residual = samples - (samples @ basis) @ basis.T
    assert basis.shape == (12, 3)
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5)
    assert torch.linalg.vector_norm(residual) < 1e-4


def test_full_span_uses_fast_path_and_releases_source_storage() -> None:
    torch.manual_seed(5)
    linear = nn.Linear(8, 6, bias=True)
    x = torch.randn(4, 8)
    expected = linear(x).detach()
    basis = activation_basis(torch.eye(8), rank=8)
    wrapped = GatedProjectedLinear.from_linear(
        linear, basis, epsilon=1e-5, offload_exact=True
    )
    actual = wrapped(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert wrapped.stats.fast_rows == 4
    assert wrapped.stats.cold_weight_reads == 0
    assert linear.weight.numel() == 0
    assert linear.bias is not None and linear.bias.numel() == 0


def test_zero_epsilon_falls_back_for_unseen_direction() -> None:
    torch.manual_seed(6)
    weight = torch.randn(5, 7)
    linear = nn.Linear(7, 5, bias=False)
    linear.weight.data.copy_(weight)
    basis = activation_basis(torch.eye(7)[:2], rank=2)
    wrapped = GatedProjectedLinear.from_linear(
        linear, basis, epsilon=0.0, offload_exact=True
    )
    x = torch.randn(3, 7)
    actual = wrapped(x)
    expected = F.linear(x, weight)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert wrapped.stats.slow_rows == 3
    assert wrapped.stats.cold_weight_reads == 1
    assert wrapped.stats.cold_weight_bytes == weight.numel() * weight.element_size()
    assert linear.weight.numel() == 0
