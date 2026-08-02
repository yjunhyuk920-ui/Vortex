from __future__ import annotations

import torch

from vortex_runtime.diagonal_transport import (
    diagonal_transport_linear,
    diagonal_transport_metadata_budget,
    fit_diagonal_transport,
    materialize_diagonal_transport,
)
from vortex_runtime.feasibility import default_specs


def test_diagonal_transport_recovers_exact_scaled_matrix() -> None:
    generator = torch.Generator().manual_seed(7)
    representative = torch.randn(12, 9, generator=generator)
    input_scale = torch.linspace(0.6, 1.4, 9)
    output_scale = torch.linspace(0.7, 1.3, 12)
    target = representative * output_scale[:, None] * input_scale[None, :]

    fitted_input, fitted_output, stats = fit_diagonal_transport(
        target_weight=target,
        representative_weight=representative,
        iterations=12,
    )
    reconstructed = materialize_diagonal_transport(
        representative_weight=representative,
        input_scale=fitted_input,
        output_scale=fitted_output,
    )
    relative_error = torch.linalg.vector_norm(target - reconstructed) / torch.linalg.vector_norm(target)
    assert float(relative_error.item()) < 5e-5
    assert stats.adapted_relative_l2_error < stats.baseline_relative_l2_error
    assert stats.relative_error_reduction > 0.999
    assert stats.metadata_bytes == (12 + 9) * 2


def test_factorized_linear_matches_materialized_weight() -> None:
    generator = torch.Generator().manual_seed(11)
    representative = torch.randn(7, 5, generator=generator)
    input_scale = torch.randn(5, generator=generator)
    output_scale = torch.randn(7, generator=generator)
    bias = torch.randn(7, generator=generator)
    x = torch.randn(3, 4, 5, generator=generator)

    materialized = materialize_diagonal_transport(
        representative_weight=representative,
        input_scale=input_scale,
        output_scale=output_scale,
    )
    expected = torch.nn.functional.linear(x, materialized, bias)
    actual = diagonal_transport_linear(
        x,
        representative_weight=representative,
        input_scale=input_scale,
        output_scale=output_scale,
        bias=bias,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_405b_transport_metadata_is_small_relative_to_dictionary() -> None:
    target, _ = default_specs()
    budget = diagonal_transport_metadata_budget(model=target, metadata_bits=16)
    assert budget.metadata_gib < 0.1
    assert budget.total_elements > 0
    assert budget.scale_elements_per_layer > budget.exact_vector_elements_per_layer


def test_transport_rejects_shape_mismatch() -> None:
    try:
        fit_diagonal_transport(
            target_weight=torch.zeros(4, 3),
            representative_weight=torch.zeros(4, 2),
        )
    except ValueError as error:
        assert "shapes must match" in str(error)
    else:
        raise AssertionError("shape mismatch should fail")
