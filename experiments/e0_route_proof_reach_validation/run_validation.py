#!/usr/bin/env python3
"""Deterministic E0/E1 validation for ROUTECELL, PROOFWAVE, and REACHQUOTIENT."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from proofwave import TARGET_FRACTION, validate_proofwave  # noqa: E402
from reachquotient import validate_reachquotient  # noqa: E402
from routecell import validate_routecell  # noqa: E402


def build_summary() -> dict[str, Any]:
    routecell = validate_routecell()
    proofwave = validate_proofwave()
    reachquotient = validate_reachquotient()

    core: dict[str, Any] = {
        "schema": "e0-route-proof-reach-validation-v1",
        "date": "2026-08-26 Asia/Seoul",
        "base_ref": "f5a432a5152ee6d9e1399583a4829a39aacdf5e0",
        "target_fraction": {
            "exact": "8/675",
            "decimal": TARGET_FRACTION,
            "percent": TARGET_FRACTION * 100,
        },
        "routecell": routecell,
        "proofwave": proofwave,
        "reachquotient": reachquotient,
        "overall": {
            "decision": "NO_ROUTE_PROOF_REACH_CORE_PROMOTED",
            "routecell": "registered uniform grammar rejected; unrestricted adaptive nonlinear cold source remains open without constructor",
            "proofwave": "rejected as standalone source; proof checker remains auxiliary",
            "reachquotient": "valid state contract; no compressed checkpoint-specific quotient or transition compiler",
            "numbered_experiment_authorized": False,
            "checkpoint_run_performed": False,
            "target_hardware_performed": False,
            "next_gate": "NON_ENTRYWISE_GLOBAL_ROUTE_GRAMMAR_WITH_TWO_SCALE_EXACT_DECODER_AND_NATIVE_LIFT_PREREQUISITE",
        },
        "claim_boundary": {
            "405b_execution": "NOT_TESTED",
            "physical_8gib_allocation": "NOT_TESTED",
            "native_q4_bf16_fp32_lift": "NOT_TESTED",
            "same_machine_4b_latency": "NOT_TESTED",
            "general_adaptive_nonlinear_cell_probe_impossibility": "NOT_ESTABLISHED",
            "actual_transformer_operation_replacement": "NOT_TESTED",
        },
    }
    deterministic_bytes = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary = dict(core)
    summary["deterministic_core_sha256"] = hashlib.sha256(deterministic_bytes).hexdigest()
    return summary


def main() -> None:
    print(json.dumps(build_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
