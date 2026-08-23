from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vortex_runtime.resident_slice_budget import (
    BF16,
    GIB,
    Q4_G128,
    Q4_IDEAL,
    Q8_G128,
    LlamaInventory,
    default_reserve,
    draft_weight_scan_bytes,
    maximum_full_layers_under_weight_budget,
    maximum_resident_layers,
    resident_total_bytes,
)

NATIVE_4B_Q4_WEIGHT_BYTES = 2_000_000_000
P50_MULTIPLIER = 1.2
P95_MULTIPLIER = 1.5
SEGMENT_TOKENS = 339
REGISTERED_PARAMETER_COUNT = 405_849_243_648


def stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run() -> dict[str, Any]:
    inv = LlamaInventory()
    reserve = default_reserve(inv, SEGMENT_TOKENS)
    p50_budget = int(NATIVE_4B_Q4_WEIGHT_BYTES * P50_MULTIPLIER)
    p95_budget = int(NATIVE_4B_Q4_WEIGHT_BYTES * P95_MULTIPLIER)

    formats = [BF16, Q8_G128, Q4_IDEAL, Q4_G128]
    rows: list[dict[str, Any]] = []
    for quant in formats:
        max_layers_vram = maximum_resident_layers(inv, quant, SEGMENT_TOKENS, reserve)
        head_bytes = quant.stored_bytes(inv.lm_head_parameters)
        layer_bytes = quant.stored_bytes(inv.layer_parameters)
        one_layer_scan = draft_weight_scan_bytes(inv, quant, 1)
        rows.append(
            {
                "format": quant.name,
                "bytes_per_parameter": float(quant.bytes_per_parameter),
                "head_bytes": head_bytes,
                "layer_bytes": layer_bytes,
                "head_plus_one_layer_bytes": one_layer_scan,
                "head_plus_one_layer_to_native_4b_ratio": one_layer_scan / NATIVE_4B_Q4_WEIGHT_BYTES,
                "head_plus_one_layer_to_p50_budget_ratio": one_layer_scan / p50_budget,
                "maximum_full_layers_under_p50_weight_budget": maximum_full_layers_under_weight_budget(inv, quant, p50_budget),
                "maximum_resident_layers_under_8gib": max_layers_vram,
                "resident_total_at_max_layers": resident_total_bytes(inv, quant, max_layers_vram, SEGMENT_TOKENS, reserve),
            }
        )

    q4_ideal = next(row for row in rows if row["format"] == Q4_IDEAL.name)
    q4_real = next(row for row in rows if row["format"] == Q4_G128.name)

    integrity_failures: list[str] = []
    if inv.registered_parameters != REGISTERED_PARAMETER_COUNT:
        integrity_failures.append("registered_parameter_population_mismatch")
    if q4_ideal["maximum_full_layers_under_p50_weight_budget"] != 0:
        integrity_failures.append("ideal_q4_p50_layer_floor_unexpected")
    if q4_real["maximum_resident_layers_under_8gib"] != 3:
        integrity_failures.append("realistic_q4_static_layer_count_unexpected")

    # A full-vocabulary target-width slice requires the complete head and at least
    # one complete target layer for nontrivial causal state. Even an impossible
    # metadata-free 4-bit representation exceeds the 1.2x p50 weight-scan budget.
    decisive_floor = (
        q4_ideal["head_plus_one_layer_bytes"] > p50_budget
        and q4_real["head_plus_one_layer_bytes"] > p50_budget
    )

    if integrity_failures:
        decision = "INVALID_RESIDENT_SLICE_BUDGET_CONTROL_FAILURE"
    elif decisive_floor:
        decision = "REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE"
    else:
        decision = "SURVIVE_FULL_VOCABULARY_RESIDENT_SLICE_STATIC_GATE"

    result: dict[str, Any] = {
        "schema": "exp-104a-checkpoint-native-resident-slice-v1",
        "authoritative_arm": "REAL_EXECUTOR_ONLY_STATIC_AND_MEASURED_LEDGER",
        "decision": decision,
        "integrity_failures": integrity_failures,
        "inventory": {
            "hidden_size": inv.hidden_size,
            "intermediate_size": inv.intermediate_size,
            "vocab_size": inv.vocab_size,
            "layers": inv.layers,
            "kv_heads": inv.kv_heads,
            "head_dim": inv.head_dim,
            "layer_parameters": inv.layer_parameters,
            "embedding_parameters": inv.embedding_parameters,
            "lm_head_parameters": inv.lm_head_parameters,
            "registered_parameters": inv.registered_parameters,
        },
        "budgets": {
            "vram_bytes": 8 * GIB,
            "native_4b_q4_weight_bytes": NATIVE_4B_Q4_WEIGHT_BYTES,
            "p50_weight_scan_budget_bytes": p50_budget,
            "p95_weight_scan_budget_bytes": p95_budget,
            "segment_tokens": SEGMENT_TOKENS,
            "reserve": {
                "target_kv_bytes": reserve.target_kv_bytes,
                "target_weight_staging_bytes": reserve.target_weight_staging_bytes,
                "block_activation_bytes": reserve.block_activation_bytes,
                "allocator_fragmentation_bytes": reserve.allocator_fragmentation_bytes,
                "total": reserve.total,
            },
            "embedding_row_stream_bytes_per_draft_token": inv.hidden_size * 2,
            "embedding_row_stream_bytes_per_339_token_block": inv.hidden_size * 2 * SEGMENT_TOKENS,
        },
        "format_rows": rows,
        "decisive_facts": {
            "ideal_q4_head_plus_one_layer_bytes": q4_ideal["head_plus_one_layer_bytes"],
            "ideal_q4_head_plus_one_layer_native_ratio": q4_ideal["head_plus_one_layer_to_native_4b_ratio"],
            "real_q4_head_plus_one_layer_bytes": q4_real["head_plus_one_layer_bytes"],
            "real_q4_head_plus_one_layer_native_ratio": q4_real["head_plus_one_layer_to_native_4b_ratio"],
            "real_q4_max_layers_under_8gib": q4_real["maximum_resident_layers_under_8gib"],
            "draft_autoregressive_scans_amortized_by_acceptance": False,
            "reason": "each proposed token requires one resident draft pass; at 100% acceptance the per-committed-token draft scan is still head+slice",
        },
        "existing_real_checkpoint_control": {
            "source": "EXP-102A committed result",
            "same_family_full_draft_parameter_bytes": 269_030_016,
            "same_family_holdout_minimum_committed_tokens": 2,
            "cross_family_holdout_minimum_committed_tokens": 1,
            "interpretation": "diagnostic context only; the static p50 byte floor is the authoritative rejection",
        },
        "three_principles_screened": [
            {
                "name": "full_vocabulary_checkpoint_native_resident_slice",
                "status": "implemented_and_rejected_by_p50_weight_scan_floor",
                "premise_flip": "draft from a target-derived resident slice rather than an independently trained model",
            },
            {
                "name": "streaming_page_shadow_self_draft",
                "status": "rejected_before_implementation_as_jacobi_parareal_dependency_family",
                "premise_flip": "derive future candidates while exact pages are transferred",
            },
            {
                "name": "deferred_mismatch_state_closure",
                "status": "retained_auxiliary_only",
                "premise_flip": "materialize mismatch-token state in the following sweep",
            },
        ],
        "next_gate": {
            "name": "sub_full_head_or_multi_token_per_scan_transducer",
            "requirement": "either score less than the full vocabulary with a causal charged mechanism or generate more than one exact-useful candidate token per resident weight scan",
            "forbidden": [
                "another full-head full-layer slice",
                "unmeasured sub-4-bit quality",
                "free HBM residency scans",
                "configured K counted as accepted A",
            ],
        },
        "claim_boundary": {
            "public_checkpoint_slice_execution": "NOT_TESTED_BECAUSE_STATIC_P50_FLOOR_DECISIVE",
            "target_405b_execution": "NOT_TESTED",
            "physical_8gib": "DERIVED_LEDGER_ONLY",
            "same_machine_latency": "DERIVED_WEIGHT_SCAN_LOWER_BOUND",
        },
    }
    result["deterministic_core"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "integrity_failures": result["integrity_failures"],
        "deterministic_core": result["deterministic_core"],
    }, indent=2))


if __name__ == "__main__":
    main()
