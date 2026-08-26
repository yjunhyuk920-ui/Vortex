from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "e0_three_invention_validation"
    / "run_validation.py"
)
SPEC = importlib.util.spec_from_file_location("three_invention_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_cspe_binary_counting_gate_is_decisive() -> None:
    summary = MOD.build_summary()
    gate = summary["cspe"]["universal_binary_counting_gate"]
    assert gate["gate_pass"] is False
    assert gate["minimum_log2_atom_count_lower_bound"] > 89
    assert gate["coverage_information_fraction_upper"] < 0.24


def test_cspe_restricted_real_evidence_misses_target() -> None:
    summary = MOD.build_summary()
    evidence = summary["cspe"]["restricted_real_checkpoint_evidence"]
    assert evidence["reconstruction_mismatches"] == 0
    assert evidence["operation_miss_factor"] > 17
    assert evidence["temporal_exact_replay_hits"] == 0


def test_stc_pointer_chasing_indistinguishability() -> None:
    for k in (2, 4, 8, 16, 85):
        row = MOD.stc_pointer_chasing_control(k)
        assert row["same_transcript_before_final_query"]
        assert row["different_kth_state"]
        assert row["universal_queries_required"] == k


def test_stc_constructive_arithmetic_misses_target() -> None:
    summary = MOD.build_summary()
    row = summary["stc"]["perfect_future_constructive_arithmetic"]
    assert row["joint_pass_count"] == 0
    assert row["miss_factor"] > 30


def test_nrm_macro_cover_is_subset_of_scalar_hits() -> None:
    control = MOD.nrm_subset_control()
    assert control["macro_cover_subset_of_scalar_hits"]


def test_nrm_real_oracle_is_far_from_required_elimination() -> None:
    summary = MOD.build_summary()
    row = summary["nrm"]["real_checkpoint_free_oracle_upper"]
    assert row["same_position_macro_miss_factor"] > 80
    assert row["best_combined_local_miss_factor"] > 77


def test_no_candidate_survives() -> None:
    summary = MOD.build_summary()
    assert summary["overall"]["surviving_core_candidate_count"] == 0
