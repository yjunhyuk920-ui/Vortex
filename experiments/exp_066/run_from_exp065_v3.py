#!/usr/bin/env python3
"""Run EXP-066 with verified EXP-065 and EXP-058 frozen-rank evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.exp_066 import run_from_exp065_v2 as v2
from vortex_runtime.tt_rank_propagation import strengthen_with_full_matrix_rank

base = v2.base
ROOT = Path(__file__).resolve().parents[2]
EXP058 = ROOT / "results/exp_058"
EXP065 = ROOT / "results/exp_065"
CONFIG = json.loads(
    (ROOT / "experiments/exp_066/config.json").read_text(encoding="utf-8")
)


def _load_exp065_q4_index() -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base.read_jsonl(EXP065 / "raw/matrix_rows.jsonl"):
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-065 matrix row: {key}")
        indexed[key] = row
    return indexed


def _load_exp058_rank_index(
    exp065_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], int, dict[str, str]]:
    expected = base.parse_checksums(EXP058 / "checksums.sha256")
    required = ("summary.json", "raw/matrix_rank_rows.jsonl")
    mismatches = 0
    hashes: dict[str, str] = {}
    for relative in required:
        digest = base.sha256_file(EXP058 / relative)
        hashes[f"exp058/{relative}"] = digest
        mismatches += int(expected.get(relative) != digest)

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base.read_jsonl(EXP058 / "raw/matrix_rank_rows.jsonl"):
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-058 matrix-rank row: {key}")
        source = exp065_index.get(key)
        if source is None:
            mismatches += 1
            continue
        mismatches += int(
            str(source["q4_integer_sha256"])
            != str(row["q4_integer_sha256"])
        )
        certificate = row["certificate"]
        shape = tuple(int(value) for value in row["shape"])
        full_rank = int(certificate["rank_lower_bound"]) == min(shape)
        mismatches += int(
            not bool(certificate["full_integer_rank_proven"]) or not full_rank
        )
        indexed[key] = row

    mismatches += len(set(exp065_index) - set(indexed))
    mismatches += len(set(indexed) - set(exp065_index))
    return indexed, mismatches, hashes


def _unresolved_survivor_metrics(output_root: Path) -> dict[str, int]:
    plan_path = output_root / "raw/plan_rows.jsonl"
    if not plan_path.exists():
        return {
            "surviving_plan_count_after_exp058_propagation": 0,
            "unresolved_unit_boundary_cut_occurrences_in_survivors": 0,
            "unique_unresolved_unit_boundary_cut_count": 0,
            "selected_unresolved_unit_boundary_cut_count": 0,
        }

    surviving = [
        row
        for row in base.read_jsonl(plan_path)
        if row.get("matrix_role") == "dense_projection"
        and bool(row.get("meets_joint_p90_screen"))
    ]
    unresolved_occurrences = 0
    selected_unresolved = 0
    unique: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for row in surviving:
        for record in row.get("cut_records", []):
            source = str(record.get("source", ""))
            if "unmeasured unit-boundary cut" not in source:
                continue
            unresolved_occurrences += 1
            selected_unresolved += int(bool(row.get("selected")))
            factors = tuple(int(value) for value in record["factors"])
            unique.add(
                (
                    str(row["model_id"]),
                    str(row["tensor_name"]),
                    factors,
                )
            )
    return {
        "surviving_plan_count_after_exp058_propagation": len(surviving),
        "unresolved_unit_boundary_cut_occurrences_in_survivors": (
            unresolved_occurrences
        ),
        "unique_unresolved_unit_boundary_cut_count": len(unique),
        "selected_unresolved_unit_boundary_cut_count": selected_unresolved,
    }


_EXP065_Q4_INDEX = _load_exp065_q4_index()
_EXP058_INDEX, _EXP058_CHECKSUM_MISMATCHES, _EXP058_HASHES = (
    _load_exp058_rank_index(_EXP065_Q4_INDEX)
)
_original_verify = base.verify_frozen_inputs
_original_derive = base.derive_reused_tt_plan
_original_write_json = base.write_json


def verify_frozen_inputs():
    summary, mismatches, hashes = _original_verify()
    hashes.update(_EXP058_HASHES)
    return summary, mismatches + _EXP058_CHECKSUM_MISMATCHES, hashes


def derive_reused_tt_plan(**kwargs):
    plan = _original_derive(**kwargs)
    key = (str(kwargs["model_id"]), str(kwargs["tensor_name"]))
    source = _EXP058_INDEX.get(key)
    if source is None:
        raise ValueError(f"missing verified EXP-058 matrix-rank row: {key}")
    expected_shape = (int(kwargs["rows"]), int(kwargs["columns"]))
    observed_shape = tuple(int(value) for value in source["shape"])
    if observed_shape != expected_shape:
        raise ValueError(
            f"EXP-058 shape mismatch for {key}: {observed_shape} != {expected_shape}"
        )
    return strengthen_with_full_matrix_rank(
        plan,
        matrix_rank_lower_bound=int(source["certificate"]["rank_lower_bound"]),
        bits_per_core=int(kwargs.get("bits_per_core", 4)),
        activation_bytes=int(kwargs.get("activation_bytes", 4)),
        has_bias=bool(kwargs["has_bias"]),
    )


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and value.get("experiment") == "EXP-066":
        measured = value.setdefault("MEASURED", {})
        derived = value.setdefault("DERIVED", {})
        provenance = value.setdefault("provenance", {})
        measured.update(_unresolved_survivor_metrics(path.parent))
        provenance["exp058_input_hashes"] = _EXP058_HASHES
        provenance["exp058_checksum_mismatches"] = _EXP058_CHECKSUM_MISMATCHES
        provenance["exp058_full_rank_rows_reused"] = len(_EXP058_INDEX)

        gate = CONFIG["gate"]
        corrected_adversary_gate = float(
            measured.get("dense_random_control_p50", 0.0)
        ) >= float(gate["minimum_dense_random_control_p50"])
        derived["adversary_gate_pass"] = corrected_adversary_gate
        derived["adversary_gate_definition"] = (
            "dense-random p50 operation fraction must remain at or above the "
            "25% target boundary; random matrices must not appear cheaply compressible"
        )
        derived["exp058_full_matrix_rank_reused"] = True
        derived["adjacent_tt_rank_inequalities_applied"] = True
        derived["derivation"] = (
            "Nontrivial TT cuts reuse checksum-verified EXP-065 Kronecker ranks. "
            "Cuts equal to W or W-transpose reuse checksum-matched EXP-058 full-rank "
            "certificates. Exact adjacent TT-rank inequalities propagate those lower "
            "bounds. Remaining unit-boundary cuts are explicitly unresolved."
        )

        scientific_gates_pass = all(
            bool(derived.get(name))
            for name in (
                "correctness_gate_pass",
                "population_gate_pass",
                "operation_gate_pass",
                "storage_gate_pass",
                "adversary_gate_pass",
                "model_trend_gate_pass",
            )
        )
        survivor_count = int(
            measured.get("dense_with_any_joint_p90_survivor", 0)
        )
        unresolved = int(
            measured.get("unique_unresolved_unit_boundary_cut_count", 0)
        )
        lower_bound_screen_survives = (
            scientific_gates_pass and survivor_count > 0
        )
        derived["tt_mpo_lower_bound_survives_gate"] = (
            lower_bound_screen_survives
        )
        derived["real_population_lower_bound_screen_survives"] = (
            lower_bound_screen_survives
        )
        derived["no_joint_p90_survivor"] = survivor_count == 0
        derived["family_closure_justified"] = (
            scientific_gates_pass and survivor_count == 0
        )
        derived["exact_integer_mpo_reconstruction_gate_pass"] = False

        if lower_bound_screen_survives and unresolved > 0:
            decision = "REVISE_REAL_Q4_TT_MPO_UNIT_BOUNDARY_RANKS_REQUIRED"
            derived["next_gate"] = (
                "certify only the unique unresolved unit-boundary cuts that occur "
                "in post-propagation surviving plans"
            )
        elif lower_bound_screen_survives:
            decision = (
                "RETAIN_REAL_Q4_TT_MPO_CANDIDATES_FOR_EXACT_RECONSTRUCTION_GATE"
            )
            derived["next_gate"] = "exact integer MPO core reconstruction"
        else:
            decision = str(CONFIG["failure_decision"])
            derived["next_gate"] = (
                "change execution class and return to bounded E0 triage"
            )
        value["authoritative_decision"] = decision
        derived["decision"] = decision

    _original_write_json(path, value)


base.verify_frozen_inputs = verify_frozen_inputs
base.derive_reused_tt_plan = derive_reused_tt_plan
base.write_json = write_json


if __name__ == "__main__":
    base.main()
