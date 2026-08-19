from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from vortex_runtime.rounding_session_atlas_composition import (
    DECISION,
    P50_TARGET_FRACTION,
    derive_composition_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixtures() -> tuple[dict, dict]:
    native = {
        "registered_budget": {
            "full_405b_gflop_per_token": 811.698487296,
            "allowed_gflop_per_token": 9.6,
            "allowed_work_fraction": 0.011827051731955725,
        },
        "omega_roundlock_oracle": {
            "atlas_plus_one_page_down": {
                "coordinates": 1024,
                "locked_coordinates": 1,
                "oracle_lock_fraction": 1 / 1024,
                "oracle_unlocked_fraction": 1023 / 1024,
                "whole_vector_locked": False,
                "reference_words_are_free_oracle_information": True,
            }
        },
        "candidate_decisions": {
            "omega_roundlock_atlas_one_page": (
                "REJECT_ORACLE_UNLOCKED_ROWS_EXCEED_COMPLETE_ALLOWANCE"
            )
        },
    }
    hyperblock = {
        "DERIVED": {
            "decision": (
                "REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY"
            ),
            "best_constructive_row": {
                "block_length": 16384,
                "constructive_ratio": 0.3611157967135951,
            },
        }
    }
    return native, hyperblock


def test_composition_separates_traffic_from_arithmetic() -> None:
    native, hyperblock = _fixtures()
    result = derive_composition_audit(native, hyperblock)

    assert result["decision"] == DECISION
    derived = result["DERIVED"]
    assert derived["minimum_perfect_accepted_length_for_p50_traffic"] == 85
    assert derived["minimum_perfect_accepted_length_for_p95_traffic"] == 68
    assert math.isclose(
        derived["repair_arithmetic_miss_factor"],
        84.46935552,
        rel_tol=0.0,
        abs_tol=1e-10,
    )

    rows = {row["accepted_length"]: row for row in derived["accepted_length_scenarios"]}
    assert rows[96]["traffic_p50_pass"] is True
    assert rows[96]["repair_arithmetic_compute_pass"] is False
    assert math.isclose(
        rows[96]["optimistic_repair_weight_traffic_fraction_per_token"],
        (1023 / 1024) / 96,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        rows[96]["repair_arithmetic_fraction_per_token_lower_bound"],
        1023 / 1024,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_source_drift_fails_closed() -> None:
    native, hyperblock = _fixtures()
    changed = copy.deepcopy(native)
    changed["omega_roundlock_oracle"]["atlas_plus_one_page_down"][
        "locked_coordinates"
    ] = 2
    with pytest.raises(ValueError, match="lock fraction"):
        derive_composition_audit(changed, hyperblock)


def test_actual_committed_sources_match_the_audit() -> None:
    native_path = ROOT / "results/e0_native_exact_shortcut_frontier/summary.json"
    hyperblock_path = ROOT / "results/exp_080a/summary.json"
    if not (native_path.is_file() and hyperblock_path.is_file()):
        pytest.skip("repository evidence is not present in this isolated test tree")

    native = json.loads(native_path.read_text(encoding="utf-8"))
    hyperblock = json.loads(hyperblock_path.read_text(encoding="utf-8"))
    result = derive_composition_audit(native, hyperblock)
    assert result["decision"] == DECISION
    assert result["DERIVED"]["gates"] == {
        "source_native_decision_matches": True,
        "source_hyperblock_decision_matches": True,
        "raw_row_repair_compute_within_allowance": False,
        "standard_constructive_block_arithmetic_within_p50": False,
    }
    assert result["DERIVED"][
        "standard_strassen_best_constructive_arithmetic_fraction"
    ] > P50_TARGET_FRACTION
