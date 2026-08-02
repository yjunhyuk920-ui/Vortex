from __future__ import annotations

import torch

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.signed_dual_mlp import (
    build_signed_dual_terms,
    refine_signed_dual_certificate,
    signed_dual_refinement_budget,
)


def _random_problem(seed: int = 31001) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = 7
    intermediate = 13
    return {
        "gate_weight": torch.randn(intermediate, hidden, generator=generator),
        "up_weight": torch.randn(intermediate, hidden, generator=generator),
        "down_weight": torch.randn(hidden, intermediate, generator=generator),
        "activation": torch.randn(hidden, generator=generator),
        "output_dual": torch.randn(hidden, generator=generator),
    }


def test_per_neuron_product_intervals_contain_exact_contributions() -> None:
    terms = build_signed_dual_terms(**_random_problem(), bits=4)
    tolerance = 1e-5
    assert torch.all(terms.exact_contributions >= terms.lower_contributions - tolerance)
    assert torch.all(terms.exact_contributions <= terms.upper_contributions + tolerance)


def test_full_precision_path_certifies_without_refinement() -> None:
    terms = build_signed_dual_terms(**_random_problem(31003), bits=16)
    certificate = refine_signed_dual_certificate(terms, require_sign=True)
    assert certificate.interval_contains_exact
    assert not certificate.unsafe_certificate
    assert certificate.refined_neurons == 0
    assert certificate.certified_sign
    assert abs(certificate.approximate_scalar - certificate.exact_scalar) < 1e-5


def test_zero_error_target_eventually_refines_to_exact_scalar() -> None:
    terms = build_signed_dual_terms(**_random_problem(31007), bits=3)
    certificate = refine_signed_dual_certificate(
        terms,
        target_absolute_error=0.0,
        require_sign=False,
    )
    assert certificate.interval_contains_exact
    assert not certificate.unsafe_certificate
    assert certificate.refined_neurons == certificate.total_neurons
    assert certificate.target_error_met
    assert abs(certificate.approximate_scalar - certificate.exact_scalar) < 1e-5
    assert abs(certificate.lower_bound - certificate.exact_scalar) < 1e-5
    assert abs(certificate.upper_bound - certificate.exact_scalar) < 1e-5


def test_signed_certificates_are_never_wrong_across_random_problems() -> None:
    for seed in range(31020, 31040):
        terms = build_signed_dual_terms(**_random_problem(seed), bits=4)
        certificate = refine_signed_dual_certificate(terms, require_sign=True)
        assert certificate.interval_contains_exact
        assert not certificate.unsafe_certificate
        if certificate.certified_sign:
            assert certificate.certified_sign_value == certificate.exact_sign


def test_405b_quarter_percent_refinement_budget_is_explicit() -> None:
    target = ModelSpec(
        parameters=405_849_243_648,
        layers=126,
        hidden_size=16_384,
        intermediate_size=53_248,
        attention_heads=128,
        kv_heads=8,
        vocab_size=128_256,
        context_tokens=8_192,
        weight_bits=16,
        kv_bits=4,
    )
    budget = signed_dual_refinement_budget(
        target=target,
        selected_fraction=0.0025,
        source_bits=16,
        partial_limit_gib=1.6,
    )
    assert budget.selected_neurons_per_layer == 134
    assert budget.exact_refinement_gib_per_token < 1.6
    assert budget.partial_traffic_pass
