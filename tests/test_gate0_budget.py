from __future__ import annotations

import math

import pytest

from vortex_runtime.gate0_budget import (
    BaselineMeasurement,
    DenseModelGeometry,
    ProjectedCapsuleCandidate,
    calculate_gate0_certificate,
    conservative_4b_proxy,
    llama_31_405b_geometry,
)


def test_default_candidate_is_conditional_not_claimed_complete() -> None:
    certificate = calculate_gate0_certificate(
        llama_31_405b_geometry(),
        ProjectedCapsuleCandidate(),
        conservative_4b_proxy(),
    )
    assert certificate.status == "conditional_pass"
    assert certificate.evidence_level == "E0-budget/E1-calculator"
    assert certificate.memory["passes"] is True
    assert certificate.traffic["passes_at_target_A"] is True
    assert certificate.compute["passes_at_target_A"] is True
    assert certificate.falsification_thresholds[
        "minimum_amortized_tokens_per_full_stream"
    ] > 1
    assert "cold-repair" in certificate.decisive_unknowns[0]


def test_candidate_rejects_when_hot_traffic_alone_exceeds_gate() -> None:
    candidate = ProjectedCapsuleCandidate(
        pre_attention_rank=1024,
        attention_output_rank=1024,
        pre_mlp_rank=1024,
        down_projection_rank=1024,
        lm_head_rank=1024,
    )
    certificate = calculate_gate0_certificate(
        llama_31_405b_geometry(), candidate, conservative_4b_proxy()
    )
    assert certificate.status == "rejected_by_budget"
    assert not certificate.traffic["passes_at_target_A"]
    assert math.isinf(certificate.traffic["required_A_from_bandwidth"])


def test_memory_gate_is_independent_of_amortization() -> None:
    candidate = ProjectedCapsuleCandidate(
        workspace_gib=6.0,
        allocator_reserve_gib=2.0,
        target_amortized_tokens_per_full_stream=1_000_000,
    )
    certificate = calculate_gate0_certificate(
        llama_31_405b_geometry(), candidate, conservative_4b_proxy()
    )
    assert certificate.memory["passes"] is False
    assert certificate.status == "rejected_by_budget"


def test_geometry_rejects_inconsistent_head_shape() -> None:
    with pytest.raises(ValueError, match="attention heads"):
        DenseModelGeometry(
            name="bad",
            parameter_count=1,
            layers=1,
            hidden_size=10,
            intermediate_size=20,
            vocab_size=100,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
        )


def test_baseline_requires_positive_values() -> None:
    with pytest.raises(ValueError):
        BaselineMeasurement(
            name="bad", traffic_gib_per_token=0, compute_gflops_per_token=1
        )
