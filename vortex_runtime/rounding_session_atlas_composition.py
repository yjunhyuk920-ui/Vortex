"""Composition audit for prompt-basis rounding repair plus block speculation.

The audit deliberately grants optimistic reuse.  It distinguishes checkpoint
weight traffic, which can be shared across a verified block, from row-dot
arithmetic, which must still be performed for every candidate position unless a
separate exact cross-position arithmetic algorithm is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


NAME = "rounding_session_atlas_composition_audit"
DECISION = "REJECT_ROUNDING_SESSION_ATLAS_RAW_REPAIR_COMPOSITION"
NATIVE_SOURCE_DECISION = (
    "REJECT_ORACLE_UNLOCKED_ROWS_EXCEED_COMPLETE_ALLOWANCE"
)
HYPERBLOCK_SOURCE_DECISION = (
    "REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY"
)
P50_TARGET_FRACTION = 1.2 * 4.0 / 405.0
P95_TARGET_FRACTION = 1.5 * 4.0 / 405.0
DEFAULT_ACCEPTED_LENGTHS = (1, 4, 8, 16, 32, 64, 68, 85, 96, 128)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_fraction(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    lower = 0.0 if allow_zero else 0.0
    if not math.isfinite(result) or result < lower or result > 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    if not allow_zero and result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def derive_composition_audit(
    native_summary: Mapping[str, Any],
    hyperblock_summary: Mapping[str, Any],
    *,
    accepted_lengths: Sequence[int] = DEFAULT_ACCEPTED_LENGTHS,
) -> dict[str, Any]:
    """Derive the fail-closed composition verdict from frozen evidence.

    The native summary supplies a measured oracle lock rate on one pinned
    real-checkpoint projection.  The Hyperblock summary supplies an independently
    derived constructive arithmetic ratio under perfect future activations.
    Neither source is reclassified as a new measurement here.
    """

    native = _mapping(native_summary, "native_summary")
    hyper = _mapping(hyperblock_summary, "hyperblock_summary")

    registered_budget = _mapping(native.get("registered_budget"), "registered_budget")
    allowed_compute_fraction = _finite_fraction(
        registered_budget.get("allowed_work_fraction"),
        "allowed_work_fraction",
    )
    full_gflop = float(registered_budget.get("full_405b_gflop_per_token"))
    allowed_gflop = float(registered_budget.get("allowed_gflop_per_token"))
    if not (math.isfinite(full_gflop) and full_gflop > 0):
        raise ValueError("full_405b_gflop_per_token must be positive")
    if not (math.isfinite(allowed_gflop) and allowed_gflop > 0):
        raise ValueError("allowed_gflop_per_token must be positive")
    if not math.isclose(
        allowed_gflop / full_gflop,
        allowed_compute_fraction,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("native compute budget is internally inconsistent")

    roundlock = _mapping(native.get("omega_roundlock_oracle"), "omega_roundlock_oracle")
    down = _mapping(roundlock.get("atlas_plus_one_page_down"), "atlas_plus_one_page_down")
    coordinates = _positive_int(down.get("coordinates"), "coordinates")
    locked_coordinates = int(down.get("locked_coordinates"))
    if not 0 <= locked_coordinates <= coordinates:
        raise ValueError("locked_coordinates is outside the coordinate population")
    measured_lock_fraction = _finite_fraction(
        down.get("oracle_lock_fraction"),
        "oracle_lock_fraction",
        allow_zero=True,
    )
    measured_unlocked_fraction = _finite_fraction(
        down.get("oracle_unlocked_fraction"),
        "oracle_unlocked_fraction",
        allow_zero=True,
    )
    if not math.isclose(
        measured_lock_fraction,
        locked_coordinates / coordinates,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("lock fraction does not match lock count")
    if not math.isclose(
        measured_lock_fraction + measured_unlocked_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("lock and unlocked fractions do not sum to one")
    if bool(down.get("whole_vector_locked")):
        raise ValueError("source unexpectedly reports a completely locked vector")

    candidate_decisions = _mapping(native.get("candidate_decisions"), "candidate_decisions")
    if candidate_decisions.get("omega_roundlock_atlas_one_page") != NATIVE_SOURCE_DECISION:
        raise ValueError("native source decision drifted")

    hyper_derived = _mapping(hyper.get("DERIVED"), "hyperblock DERIVED")
    if hyper_derived.get("decision") != HYPERBLOCK_SOURCE_DECISION:
        raise ValueError("Hyperblock source decision drifted")
    best_constructive = _mapping(
        hyper_derived.get("best_constructive_row"),
        "best_constructive_row",
    )
    best_constructive_ratio = _finite_fraction(
        best_constructive.get("constructive_ratio"),
        "constructive_ratio",
    )
    best_constructive_block = _positive_int(
        best_constructive.get("block_length"),
        "best_constructive block_length",
    )

    normalized_lengths = tuple(sorted({_positive_int(v, "accepted_length") for v in accepted_lengths}))
    if not normalized_lengths:
        raise ValueError("accepted_lengths must be nonempty")

    # Favorable traffic grant: all raw repair weights are loaded once and reused
    # across every accepted position.  Arithmetic does not receive the same 1/A
    # factor because A row-dot evaluations are still required for A positions.
    scenario_rows = []
    for accepted in normalized_lengths:
        traffic_fraction = measured_unlocked_fraction / accepted
        scenario_rows.append(
            {
                "accepted_length": accepted,
                "optimistic_repair_weight_traffic_fraction_per_token": traffic_fraction,
                "traffic_p50_pass": traffic_fraction <= P50_TARGET_FRACTION,
                "traffic_p95_pass": traffic_fraction <= P95_TARGET_FRACTION,
                "repair_arithmetic_fraction_per_token_lower_bound": measured_unlocked_fraction,
                "repair_arithmetic_compute_pass": measured_unlocked_fraction
                <= allowed_compute_fraction,
            }
        )

    minimum_p50_accepted_length = math.ceil(
        measured_unlocked_fraction / P50_TARGET_FRACTION
    )
    minimum_p95_accepted_length = math.ceil(
        measured_unlocked_fraction / P95_TARGET_FRACTION
    )
    compute_miss_factor = measured_unlocked_fraction / allowed_compute_fraction
    strassen_miss_factor = best_constructive_ratio / P50_TARGET_FRACTION

    gates = {
        "source_native_decision_matches": True,
        "source_hyperblock_decision_matches": True,
        "raw_row_repair_compute_within_allowance": measured_unlocked_fraction
        <= allowed_compute_fraction,
        "standard_constructive_block_arithmetic_within_p50": best_constructive_ratio
        <= P50_TARGET_FRACTION,
    }
    decision = DECISION if not all(gates.values()) else "INCONCLUSIVE_PROMOTE_TO_DISCRETE_REPAIR_GATE"

    return {
        "name": NAME,
        "decision": decision,
        "MEASURED": {
            "source_scope": "one pinned unchanged Qwen3.5-0.8B layer-23 down projection and frozen EXP-083B causal row",
            "coordinates": coordinates,
            "locked_coordinates": locked_coordinates,
            "oracle_lock_fraction": measured_lock_fraction,
            "oracle_unlocked_fraction": measured_unlocked_fraction,
            "reference_words_and_selector_granted_free": bool(
                down.get("reference_words_are_free_oracle_information")
            ),
            "whole_vector_locked": bool(down.get("whole_vector_locked")),
        },
        "DERIVED": {
            "p50_target_fraction": P50_TARGET_FRACTION,
            "p95_target_fraction": P95_TARGET_FRACTION,
            "allowed_compute_fraction": allowed_compute_fraction,
            "repair_arithmetic_fraction_per_committed_token_lower_bound": measured_unlocked_fraction,
            "repair_arithmetic_miss_factor": compute_miss_factor,
            "minimum_perfect_accepted_length_for_p50_traffic": minimum_p50_accepted_length,
            "minimum_perfect_accepted_length_for_p95_traffic": minimum_p95_accepted_length,
            "accepted_length_scenarios": scenario_rows,
            "standard_strassen_best_block_length": best_constructive_block,
            "standard_strassen_best_constructive_arithmetic_fraction": best_constructive_ratio,
            "standard_strassen_p50_miss_factor": strassen_miss_factor,
            "composition_equations": {
                "optimistic_weight_traffic_per_committed_token": "rho/A",
                "raw_repair_arithmetic_for_A_positions": "A*rho*C_dense",
                "raw_repair_arithmetic_per_perfectly_accepted_token": "(A*rho*C_dense)/A = rho*C_dense",
            },
            "gates": gates,
        },
        "source_claims": {
            "native_roundlock_decision": NATIVE_SOURCE_DECISION,
            "hyperblock_decision": HYPERBLOCK_SOURCE_DECISION,
        },
        "claim_boundary": {
            "new_model_forward_executed": False,
            "new_hardware_measurement": False,
            "actual_405b_execution": False,
            "dflash_rejected": False,
            "every_rounding_firewall_rejected": False,
            "scope_rejected": "prompt/Atlas proposal plus BF16 coordinate lock plus ordinary raw row/page repair, even with perfect block acceptance and optimistic weight reuse",
            "remaining_escape_requires": "a materially different exact non-rowwise or cross-position coded repair whose arithmetic, not only weight traffic, is below the final fraction",
        },
    }
