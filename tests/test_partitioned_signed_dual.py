from __future__ import annotations

import torch

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.partitioned_signed_dual import (
    build_partitioned_signed_dual_terms,
    compile_partitioned_signed_dual_kernel,
    partitioned_cone_metadata_budget,
)
from vortex_runtime.signed_dual_mlp import refine_signed_dual_certificate


def _problem(seed: int = 32001) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = 17
    intermediate = 29
    return {
        "gate_weight": torch.randn(intermediate, hidden, generator=generator),
        "up_weight": torch.randn(intermediate, hidden, generator=generator),
        "down_weight": torch.randn(hidden, intermediate, generator=generator),
        "activation": torch.randn(hidden, generator=generator),
        "output_dual": torch.randn(hidden, generator=generator),
    }


def test_partitioned_intervals_contain_exact_contributions() -> None:
    problem = _problem()
    kernel = compile_partitioned_signed_dual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        bits=4,
        block_size=5,
    )
    terms, diagnostics = build_partitioned_signed_dual_terms(
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
    assert diagnostics.partitioned_gate_radius_sum <= diagnostics.global_gate_radius_sum + tolerance
    assert diagnostics.partitioned_up_radius_sum <= diagnostics.global_up_radius_sum + tolerance
    assert (
        diagnostics.partitioned_directional_radius_sum
        <= diagnostics.global_directional_radius_sum + tolerance
    )


def test_finer_partitions_never_worsen_dot_radius_on_nested_blocks() -> None:
    problem = _problem(32003)
    coarse = compile_partitioned_signed_dual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        bits=4,
        block_size=17,
    )
    fine = compile_partitioned_signed_dual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        bits=4,
        block_size=1,
    )
    _, coarse_diagnostics = build_partitioned_signed_dual_terms(
        coarse,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    _, fine_diagnostics = build_partitioned_signed_dual_terms(
        fine,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    tolerance = 1e-5
    assert (
        fine_diagnostics.partitioned_gate_radius_sum
        <= coarse_diagnostics.partitioned_gate_radius_sum + tolerance
    )
    assert (
        fine_diagnostics.partitioned_up_radius_sum
        <= coarse_diagnostics.partitioned_up_radius_sum + tolerance
    )
    assert (
        fine_diagnostics.partitioned_directional_radius_sum
        <= coarse_diagnostics.partitioned_directional_radius_sum + tolerance
    )


def test_full_precision_partitioned_kernel_needs_no_refinement() -> None:
    problem = _problem(32007)
    kernel = compile_partitioned_signed_dual_kernel(
        gate_weight=problem["gate_weight"],
        up_weight=problem["up_weight"],
        down_weight=problem["down_weight"],
        bits=16,
        block_size=4,
    )
    terms, _ = build_partitioned_signed_dual_terms(
        kernel,
        activation=problem["activation"],
        output_dual=problem["output_dual"],
    )
    certificate = refine_signed_dual_certificate(terms, require_sign=True)
    assert certificate.refined_neurons == 0
    assert certificate.certified_sign
    assert not certificate.unsafe_certificate


def test_405b_block256_metadata_budget_is_explicit() -> None:
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
    budget = partitioned_cone_metadata_budget(
        target=target,
        block_size=256,
        metadata_bits=8,
        metadata_limit_gib=1.5,
    )
    assert budget.blocks_per_hidden_vector == 64
    assert abs(budget.metadata_gib - 1.19970703125) < 1e-12
    assert budget.metadata_pass
