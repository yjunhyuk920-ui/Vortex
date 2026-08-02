from vortex_runtime.feasibility import (
    ObservedMechanism,
    WaveCandidate,
    architecture_gate0_report,
    default_gate0_report,
    default_specs,
)


def test_default_candidate_is_rejected_by_corrected_compute_accounting() -> None:
    report = default_gate0_report()
    assert report["memory"]["pass"] is True
    assert report["traffic"]["analytic_pass"] is True
    assert report["compute"]["analytic_pass"] is False
    assert report["status"] == "rejected-analytic-compute"
    assert (
        report["compute"]["projected_gflop_per_token"]
        > report["compute"]["limit_gflop_per_token"]
    )
    assert report["compute"]["maximum_repair_fraction"] < 0.011


def test_memory_gate_rejects_large_capsule_rank() -> None:
    target, baseline = default_specs()
    report = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(linear_rank=256),
        observed=ObservedMechanism(1024, 0.01, "test"),
    )
    assert report["memory"]["pass"] is False
    assert report["status"] == "rejected-memory"


def test_longer_block_reduces_traffic_but_not_exact_arithmetic() -> None:
    target, baseline = default_specs()
    short = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(
            repair_fraction=0.01,
            committed_tokens_per_repair=8,
        ),
        observed=ObservedMechanism(8, 0.01, "test"),
    )
    long = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(
            repair_fraction=0.01,
            committed_tokens_per_repair=128,
        ),
        observed=ObservedMechanism(128, 0.01, "test"),
    )
    assert (
        long["traffic"]["projected_gib_per_token"]
        < short["traffic"]["projected_gib_per_token"]
    )
    assert (
        long["compute"]["projected_gflop_per_token"]
        == short["compute"]["projected_gflop_per_token"]
    )


def test_compute_valid_low_fraction_candidate_can_advance() -> None:
    target, baseline = default_specs()
    report = architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(
            repair_fraction=0.01,
            committed_tokens_per_repair=8,
        ),
        observed=ObservedMechanism(8, 0.01, "test"),
    )
    assert report["traffic"]["analytic_pass"] is True
    assert report["compute"]["analytic_pass"] is True
    assert report["mechanism"]["traffic_pass"] is True
    assert report["mechanism"]["compute_pass"] is True
    assert report["status"] == "gate0-candidate-ready-for-e2-falsification"
