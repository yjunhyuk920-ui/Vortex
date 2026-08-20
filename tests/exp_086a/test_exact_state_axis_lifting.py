from __future__ import annotations

import numpy as np
import pytest
import torch

from vortex_runtime.exact_state_axis_lifting import (
    ExactStateAxisLiftingError,
    analyze_lifting_block,
    analyze_mlp_block,
    encode_bf16_block_exact,
)


def bf(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bfloat16)


def test_exact_dyadic_encoding_preserves_equal_words() -> None:
    x = bf([[0.0, 1.0, -2.0, 0.5], [0.0, 1.0, -2.0, 0.5]])
    encoded = encode_bf16_block_exact(x)
    assert encoded.states == 2
    assert encoded.width == 4
    assert np.array_equal(encoded.scaled[0], encoded.scaled[1])


def test_identical_states_collapse_to_one_base_pass() -> None:
    x = bf([[1.0, -2.0, 0.5, 3.0]] * 8)
    analysis = analyze_lifting_block(x, mode="coordinate")
    assert analysis.residual_nonzero_count == 0
    assert analysis.residual_atom_union_count == 0
    assert analysis.atom_union_operation_fraction == pytest.approx(1 / 8)


def test_linear_recurrence_is_detected_exactly() -> None:
    rows = [[float(t), float(2 * t), float(-t), 1.0] for t in range(8)]
    x = bf(rows)
    analysis = analyze_lifting_block(x, mode="vector")
    assert analysis.predictor_histogram.get("linear2", 0) >= 4 * 6
    assert analysis.reconstruction_mismatches == 0


def test_coordinate_oracle_is_never_worse_than_vector_oracle() -> None:
    torch.manual_seed(7)
    x = torch.randn(16, 32, dtype=torch.float32).to(torch.bfloat16)
    vector = analyze_lifting_block(x, mode="vector")
    coordinate = analyze_lifting_block(x, mode="coordinate")
    assert coordinate.residual_nonzero_count <= vector.residual_nonzero_count
    assert coordinate.residual_popcount_sum <= vector.residual_popcount_sum


def test_mlp_accounting_uses_two_input_and_one_down_streams() -> None:
    x = bf([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    z = bf(
        [
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]
    )
    result = analyze_mlp_block(x, z, mode="coordinate")
    assert result.projected_mlp_operation_fraction == pytest.approx(0.25)
    assert result.projected_whole_model_weight_fraction > 0


def test_nonfinite_fails_closed() -> None:
    x = torch.tensor([[float("inf")], [1.0]], dtype=torch.bfloat16)
    with pytest.raises(ExactStateAxisLiftingError):
        analyze_lifting_block(x, mode="coordinate")
