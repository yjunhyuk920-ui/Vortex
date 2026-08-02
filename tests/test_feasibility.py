from vortex_runtime.feasibility import (
    ModelSpec,
    ObservedMechanism,
    WaveCandidate,
    architecture_gate0_report,
    default_gate0_report,
)


def test_default_candidate_closes_analytic_envelope_but_not_observed_mechanism() -> None:
    report = default_gate0_report()
    assert report["memory"]["pass"] is True
    assert report["traffic"]["analytic_pass"] is True
    assert report["compute"]["analytic_pass"] is True
    assert report["mechanism"]["pass"] is False
    assert report["status"] == "blocked-mechanism-unproven"
    assert report["mechanism"]["shortfall_factor"] > 100


def test_memory_gate_rejects_large_capsule_rank() -> None:
    target = ModelSpec(
        parameters=405_849_243_648,
        layers=126,
        hidden_size=16_384,
        intermediate_size=53_248,
        attention_heads=128,
        kv_heads=8,
        vocab_size=128_256,
        context_tokens=4096,
        weight_bits=16,
        kv_bits=16,
    )
    baseline = ModelSpec(
        parameters=4_000_000_000,
        layers=32,
        hidden_size=3072,
        intermediate_size=8192,
        attention_heads=24,
        kv_heads=8,
        vocab_size=128_256,
        context_tokens=4096,
        weight_bits=4,
        kv_bits=16,
    )
    report = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(linear_rank=256),
        observed=ObservedMechanism(1024, 0.01, "test"),
    )
    assert report["memory"]["pass"] is False
    assert report["status"] == "rejected-memory"


def test_sufficient_measured_repair_efficiency_advances_candidate() -> None:
    baseline_report = default_gate0_report()
    required = baseline_report["mechanism"][
        "required_tokens_per_full_repair_equivalent"
    ]

    target = ModelSpec(**baseline_report["target"])
    baseline = ModelSpec(**baseline_report["baseline"])
    report = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(),
        observed=ObservedMechanism(required * 0.25 * 1.05, 0.25, "test"),
    )
    assert report["mechanism"]["pass"] is True
    assert report["status"] == "gate0-candidate-ready-for-e2-falsification"
