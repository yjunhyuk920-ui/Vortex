from __future__ import annotations

import torch

from vortex_runtime.global_margin_refinement import (
    compare_equal_layer_and_global_refinement,
    concatenate_signed_dual_terms,
    dual_price_global_refinement,
)
from vortex_runtime.signed_dual_mlp import SignedDualTerms


def _terms(low: list[float], up: list[float]) -> SignedDualTerms:
    approximate = torch.zeros(len(low), dtype=torch.float32)
    exact = torch.tensor(
        [(right - left) * 0.25 for left, right in zip(low, up)],
        dtype=torch.float32,
    )
    return SignedDualTerms(
        exact_contributions=exact,
        approximate_contributions=approximate,
        lower_contributions=-torch.tensor(low, dtype=torch.float32),
        upper_contributions=torch.tensor(up, dtype=torch.float32),
        activation_error_bounds=torch.zeros(len(low)),
        directional_error_bounds=torch.zeros(len(low)),
    )


def test_concatenation_preserves_all_neurons() -> None:
    combined = concatenate_signed_dual_terms(
        [_terms([1.0, 2.0], [3.0, 4.0]), _terms([5.0], [6.0])]
    )
    assert combined.exact_contributions.numel() == 3
    assert combined.lower_contributions.tolist() == [-1.0, -2.0, -5.0]
    assert combined.upper_contributions.tolist() == [3.0, 4.0, 6.0]


def test_dual_price_selection_is_sound_and_meets_both_sides() -> None:
    terms = _terms(
        [9.0, 8.0, 1.0, 1.0],
        [1.0, 1.0, 9.0, 8.0],
    )
    selection = dual_price_global_refinement(
        terms,
        target_absolute_error=2.0,
        price_steps=21,
    )
    certificate = selection.certificate
    assert certificate.target_error_met
    assert certificate.interval_contains_exact
    assert not certificate.unsafe_certificate
    assert selection.refined_neurons <= selection.certificate.total_neurons


def test_dual_price_sweep_is_never_worse_than_width_global() -> None:
    layers = [
        _terms([5.0, 4.0, 1.0], [1.0, 2.0, 6.0]),
        _terms([1.0, 7.0, 2.0], [8.0, 1.0, 2.0]),
    ]
    comparison = compare_equal_layer_and_global_refinement(
        layers,
        total_absolute_error=3.0,
        price_steps=21,
    )
    assert comparison.dual_price_refined_neurons <= comparison.width_global_refined_neurons
    assert comparison.dual_price_refined_neurons <= comparison.equal_layer_refined_neurons
    assert comparison.dual_price_certificate.target_error_met
    assert not comparison.dual_price_certificate.unsafe_certificate


def test_global_budget_can_strictly_reuse_slack_across_layers() -> None:
    # Equal per-layer targets are 0.75. The difficult first layer needs both
    # exact refinements, while the easy second layer needs one because its
    # initial uncertainty is 1.2. Under one global 1.5 budget, refining only the
    # difficult layer leaves the easy layer's full 1.2 uncertainty admissible.
    layers = [
        _terms([3.0, 3.0], [3.0, 3.0]),
        _terms([0.6, 0.6], [0.6, 0.6]),
    ]
    comparison = compare_equal_layer_and_global_refinement(
        layers,
        total_absolute_error=1.5,
        price_steps=21,
    )
    assert comparison.equal_layer_refined_neurons == 3
    assert comparison.dual_price_refined_neurons == 2
    assert comparison.dual_price_refined_neurons < comparison.equal_layer_refined_neurons
    assert comparison.dual_price_certificate.target_error_met
    assert not comparison.dual_price_certificate.unsafe_certificate
