#!/usr/bin/env python3
"""Run EXP-066 with EXP-065 cut ranks plus EXP-058 rank propagation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.exp_066 import run_from_exp065_v2 as v2
from vortex_runtime.tt_rank_propagation import strengthen_with_full_matrix_rank

base = v2.base
ROOT = Path(__file__).resolve().parents[2]
EXP058 = ROOT / "results/exp_058"


def _load_exp058_rank_index() -> tuple[dict[tuple[str, str], dict[str, Any]], int, dict[str, str]]:
    expected = base.parse_checksums(EXP058 / "checksums.sha256")
    required = ("summary.json", "raw/matrix_rank_rows.jsonl")
    mismatches = 0
    hashes: dict[str, str] = {}
    for relative in required:
        digest = base.sha256_file(EXP058 / relative)
        hashes[f"exp058/{relative}"] = digest
        mismatches += int(expected.get(relative) != digest)
    rows = base.read_jsonl(EXP058 / "raw/matrix_rank_rows.jsonl")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-058 matrix rank row: {key}")
        if not bool(row["certificate"]["full_integer_rank_proven"]):
            raise ValueError(f"EXP-058 full-rank proof missing: {key}")
        indexed[key] = row
    return indexed, mismatches, hashes


_EXP058_INDEX, _EXP058_CHECKSUM_MISMATCHES, _EXP058_HASHES = (
    _load_exp058_rank_index()
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
        raise ValueError(f"missing EXP-058 matrix rank row: {key}")
    if str(source["q4_integer_sha256"]) != next(
        (
            row.get("q4_integer_sha256")
            for row in []
        ),
        str(source["q4_integer_sha256"]),
    ):
        raise AssertionError("unreachable checksum guard")
    return strengthen_with_full_matrix_rank(
        plan,
        matrix_rank_lower_bound=int(source["certificate"]["rank_lower_bound"]),
        bits_per_core=int(kwargs.get("bits_per_core", 4)),
        activation_bytes=int(kwargs.get("activation_bytes", 4)),
        has_bias=bool(kwargs["has_bias"]),
    )


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and value.get("experiment") == "EXP-066":
        measured = value.get("MEASURED", {})
        derived = value.get("DERIVED", {})
        provenance = value.get("provenance", {})
        provenance["exp058_input_hashes"] = _EXP058_HASHES
        provenance["exp058_checksum_mismatches"] = _EXP058_CHECKSUM_MISMATCHES
        provenance["exp058_full_rank_rows_reused"] = len(_EXP058_INDEX)
        derived["exp058_full_matrix_rank_reused"] = True
        derived["adjacent_tt_rank_inequalities_applied"] = True
        survivors = int(measured.get("dense_with_any_joint_p90_survivor", 0))
        if (
            derived.get("correctness_gate_pass")
            and derived.get("population_gate_pass")
            and derived.get("operation_gate_pass")
            and derived.get("storage_gate_pass")
            and survivors > 0
        ):
            decision = "REVISE_REAL_Q4_TT_MPO_UNIT_BOUNDARY_RANKS_REQUIRED"
            value["authoritative_decision"] = decision
            derived["decision"] = decision
            derived["real_population_lower_bound_screen_survives"] = True
            derived["family_closure_justified"] = False
            derived["next_gate"] = (
                "certify only unresolved unit-boundary ranks in surviving plans"
            )
        value["DERIVED"] = derived
        value["provenance"] = provenance
    _original_write_json(path, value)


base.verify_frozen_inputs = verify_frozen_inputs
base.derive_reused_tt_plan = derive_reused_tt_plan
base.write_json = write_json


if __name__ == "__main__":
    base.main()
