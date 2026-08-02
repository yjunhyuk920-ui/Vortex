from __future__ import annotations

import torch

from vortex_runtime.block_signed_residual_code import (
    build_block_signed_residual_terms,
    compile_block_signed_residual_kernel,
    signed_residual_code_budget,
)
from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.signed_dual_mlp import refine_signed_dual_certificate


def _problem(seed: int = 33001) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = 19
    intermediate = 31
    return {
        "gate_weight": torch.randn(intermediate, hidden, generator=generator),
        "up_weight": torch.randn(intermediate, hidden, generator=generator),
        "down_weight": torch.randn(hidden, intermediate, generator=generator),
        "activation": torch.randn(hidden, generator=generator),
        "output_dual": torch.randn(hidden, generator=generator),
    }


def test_signed_code_intervals_contain_exact_contributions() -> None:
    problem = _problem()
    build_activation = [torch.randn_like(problem["activation"]) for _ in range(2)]
    build_dual = [torch.randn_like(problem["output_dual"]) for _ in range(2)]
    kernel = compile_block_signed_residual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        activation_build_vectors=build_activation,
        dual_build_vectors=build_dual,
        bits=4,
        block_size=7,
        rank=2,
    )
    terms, _ = build_block_signed_residual_terms(
        kernel,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    tolerance = 1e-5
    assert torch.all(terms.exact_contributions >= terms.lower_contributions - tolerance)
    assert torch.all(terms.exact_contributions <= terms.upper_contributions + tolerance)
    certificate = refine_signed_dual_certificate(terms, require_sign=True)
    assert certificate.interval_contains_exact
    assert not certificate.unsafe_certificate


def test_basis_aligned_trace_removes_residual_dot_uncertainty() -> None:
    problem = _problem(33003)
    kernel = compile_block_signed_residual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        activation_build_vectors=[problem["activation"]],
        dual_build_vectors=[problem["output_dual"]],
        bits=4,
        block_size=8,
        rank=1,
    )
    terms, diagnostics = build_block_signed_residual_terms(
        kernel,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    certificate = refine_signed_dual_certificate(
        terms,
        target_absolute_error=1e-4,
        require_sign=False,
    )
    assert diagnostics.activation_perpendicular_ratio < 1e-5
    assert diagnostics.dual_perpendicular_ratio < 1e-5
    assert diagnostics.gate_radius_to_global < 1e-4
    assert diagnostics.up_radius_to_global < 1e-4
    assert diagnostics.directional_radius_to_global < 1e-4
    assert certificate.refined_neurons == 0
    assert certificate.target_error_met
    assert not certificate.unsafe_certificate


def test_full_precision_code_is_exact() -> None:
    problem = _problem(33007)
    kernel = compile_block_signed_residual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        activation_build_vectors=[problem["activation"]],
        dual_build_vectors=[problem["output_dual"]],
        bits=16,
        block_size=6,
        rank=1,
    )
    terms, _ = build_block_signed_residual_terms(
        kernel,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    certificate = refine_signed_dual_certificate(terms, require_sign=True)
    assert certificate.refined_neurons == 0
    assert certificate.certified_sign
    assert not certificate.unsafe_certificate


def test_405b_rank1_block1024_float32_metadata_is_explicit() -> None:
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
    budget = signed_residual_code_budget(
        target=target,
        block_size=1024,
        rank=1,
        coefficient_bits=32,
        remainder_bits=32,
        basis_bits=32,
        metadata_limit_gib=6.0,
    )
    assert budget.blocks_per_hidden_vector == 16
    assert abs(budget.total_metadata_gib - 2.414794921875) < 1e-12
    assert budget.metadata_pass
