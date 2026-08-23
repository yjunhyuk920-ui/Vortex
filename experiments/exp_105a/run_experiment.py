from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from vortex_runtime.progressive_head_tournament import (
    ROW_NORM_BYTES,
    ROW_STATE_BYTES_PER_ACTIVE_STAGE,
    exact_activation_ordered_tournament,
    full_q4_head_bytes,
    make_late_decision_control,
    make_random_q4_matrix,
    make_structured_positive_control,
)


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "maximum": float(arr.max(initial=0.0)),
    }


def stable_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def causal_dependency_gate(tokens: int = 8, layers: int = 4) -> dict[str, Any]:
    """Construct the exact autoregressive dependency chain and count serial scans."""
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    previous_head: str | None = None
    for token in range(tokens):
        embedding = f"t{token}:embedding"
        nodes.append(embedding)
        if previous_head is not None:
            edges.append((previous_head, embedding))
        previous = embedding
        for layer in range(layers):
            current = f"t{token}:layer{layer}"
            nodes.append(current)
            edges.append((previous, current))
            previous = current
        head = f"t{token}:head"
        nodes.append(head)
        edges.append((previous, head))
        previous_head = head
    distance: dict[str, int] = {node: 1 for node in nodes}
    incoming: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        incoming[target].append(source)
    for node in nodes:
        if incoming[node]:
            distance[node] = 1 + max(distance[source] for source in incoming[node])
    return {
        "tokens": tokens,
        "layers": layers,
        "weight_bearing_stages_per_token": layers + 1,
        "minimum_serial_weight_stages": tokens * (layers + 1),
        "longest_dependency_path_nodes": max(distance.values()),
        "conclusion": "without guessed branches or a precompiled transition operator, token t+1 cannot enter layer0 before token t head selection",
    }


def target_late_decision_ledger(config: dict[str, Any]) -> list[dict[str, Any]]:
    target = config["target"]
    vocab = int(target["vocab_size"])
    dimension = int(target["dimension"])
    head_bytes = float(target["real_q4_head_bytes"])
    layer_bytes = float(target["real_q4_one_layer_bytes"])
    budget = float(target["p50_budget_bytes"])
    rows: list[dict[str, Any]] = []
    for block_size in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        stages = math.ceil(dimension / block_size)
        selector_state = vocab * ROW_STATE_BYTES_PER_ACTIVE_STAGE * stages
        norm_bytes = vocab * ROW_NORM_BYTES
        activation_order = dimension * 5
        query_head = head_bytes + selector_state + norm_bytes + activation_order
        total = layer_bytes + query_head
        rows.append({
            "block_size": block_size,
            "stages": stages,
            "head_query_bytes": query_head,
            "total_draft_bytes_per_token": total,
            "ratio_to_p50_budget": total / budget,
            "passes_p50": total <= budget,
        })
    return rows


def run(config: dict[str, Any]) -> dict[str, Any]:
    sim = config["simulation"]
    rng = np.random.default_rng(int(config["random_seed"]))
    vocab = int(sim["vocab_size"])
    dimension = int(sim["dimension"])
    query_count = int(sim["queries"])
    block_sizes = [int(value) for value in sim["block_sizes"]]

    weights = make_random_q4_matrix(rng, vocab, dimension)
    random_rows: dict[str, list[dict[str, Any]]] = {str(size): [] for size in block_sizes}
    for query_index in range(query_count):
        activation = rng.integers(-127, 128, size=dimension, dtype=np.int16).astype(np.int8)
        for block_size in block_sizes:
            row = exact_activation_ordered_tournament(weights, activation, block_size).to_dict()
            row["query_index"] = query_index
            random_rows[str(block_size)].append(row)

    random_summary: dict[str, dict[str, Any]] = {}
    for block_size, rows in random_rows.items():
        random_summary[block_size] = {
            "queries": len(rows),
            "exact_all": all(bool(row["exact"]) for row in rows),
            "early_before_final_count": sum(
                bool(row["stopped_early"]) and int(row["stages_executed"]) < int(row["stage_count"])
                for row in rows
            ),
            "weight_element_fraction": quantiles([float(row["weight_element_fraction"]) for row in rows]),
            "head_query_fraction": quantiles([float(row["head_query_fraction"]) for row in rows]),
            "stages_executed": quantiles([float(row["stages_executed"]) for row in rows]),
        }

    positive_rows: dict[str, dict[str, Any]] = {}
    positive_weights, positive_activation, expected = make_structured_positive_control(
        rng, 2048, 1024, decisive_dimensions=64
    )
    for block_size in (32, 64, 128, 256):
        row = exact_activation_ordered_tournament(positive_weights, positive_activation, block_size).to_dict()
        row["expected_winner"] = expected
        positive_rows[str(block_size)] = row

    late_rows: dict[str, dict[str, Any]] = {}
    late_weights, late_activation = make_late_decision_control(rng, 2048, 2048, 128)
    for block_size in (64, 128, 256, 512, 1024, 2048):
        late_rows[str(block_size)] = exact_activation_ordered_tournament(
            late_weights, late_activation, block_size
        ).to_dict()

    target_rows = target_late_decision_ledger(config)
    best_target = min(target_rows, key=lambda row: float(row["total_draft_bytes_per_token"]))

    target = config["target"]
    static_bytes = (
        int(target["real_q4_head_bytes"])
        + int(target["real_q4_one_layer_bytes"])
        + int(target["reserve_bytes"])
        + int(target["vocab_size"]) * (ROW_NORM_BYTES + 20)
        + int(target["dimension"]) * 5
    )

    integrity_failures: list[str] = []
    if not all(summary["exact_all"] for summary in random_summary.values()):
        integrity_failures.append("random_exact_argmax_mismatch")
    if not all(bool(row["exact"]) and int(row["winner"]) == expected for row in positive_rows.values()):
        integrity_failures.append("positive_control_mismatch")
    if not all(bool(row["exact"]) for row in late_rows.values()):
        integrity_failures.append("late_decision_control_mismatch")

    random_best = min(
        random_summary.items(), key=lambda item: float(item[1]["head_query_fraction"]["p50"])
    )
    projected_random_total = (
        int(target["real_q4_one_layer_bytes"])
        + float(random_best[1]["head_query_fraction"]["p50"]) * int(target["real_q4_head_bytes"])
    )

    if integrity_failures:
        decision = "INVALID_EXP_105A_PROGRESSIVE_HEAD_TOURNAMENT_CONTROL_FAILURE"
    elif not bool(best_target["passes_p50"]):
        decision = "REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE"
    else:
        decision = "SURVIVE_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_GATE"

    result: dict[str, Any] = {
        "schema": config["schema"],
        "authoritative_arm": config["authoritative_arm"],
        "decision": decision,
        "round_decision": "NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE",
        "integrity_failures": integrity_failures,
        "config": config,
        "principles": [
            {
                "name": "activation_ordered_exact_residual_bound_head_index",
                "status": "implemented_and_rejected_with_minimum_complete_layer",
                "premise_flip": "score only query-relevant head columns/rows until exact residual bounds certify the winner",
            },
            {
                "name": "single_scan_multi_token_resident_transducer",
                "status": "rejected_before_backend_by_causal_dependency_gate",
                "premise_flip": "one resident scan emits multiple causally ordered tokens",
            },
            {
                "name": "cross_matrix_decision_bit_program",
                "status": "not_reopened_without_new_information_source",
                "premise_flip": "compile winner bits across all projections instead of materializing matrices",
                "closed_family": "self-contained exact program requires checkpoint information; query-time version must expose charged probes",
            },
        ],
        "exact_head_abi": {
            "weights": "signed Q4 integers [-8,7] with group-128 FP16 scale+zero bytes charged",
            "activation": "signed Q8 integers",
            "accumulator": "exact int64 dot product",
            "selector": "activation dimensions descending by absolute value; Cauchy tail bound with exact ceil integer sqrt",
            "tie_break": "lowest token id",
        },
        "random_dense_control": {
            "shape": [vocab, dimension],
            "queries": query_count,
            "summary": random_summary,
            "best_p50_block_size": int(random_best[0]),
            "best_p50_head_query_fraction": float(random_best[1]["head_query_fraction"]["p50"]),
            "projected_target_total_bytes_per_token": projected_random_total,
            "projected_target_ratio_to_p50_budget": projected_random_total / int(target["p50_budget_bytes"]),
            "provenance": "MEASURED small finite-word control; target bytes are PROJECTED",
        },
        "structured_positive_control": positive_rows,
        "late_decision_finite_word_control": late_rows,
        "target_scale_late_decision_ledger": {
            "rows": target_rows,
            "best": best_target,
            "head_only_fits_p50_byte_budget": float(best_target["head_query_bytes"]) <= int(target["p50_budget_bytes"]),
            "head_only_not_promoted_reason": "no implemented causal state source with measured A>=339; adding the minimum complete target-width layer exceeds p50",
            "provenance": "DERIVED exact bytes for a legal late-decision Q4 head/query under this algorithm",
        },
        "static_resident_ledger": {
            "bytes": static_bytes,
            "vram_bytes": int(target["vram_bytes"]),
            "fits": static_bytes <= int(target["vram_bytes"]),
            "meaning": "capacity passes; per-token scan traffic fails",
        },
        "causal_single_scan_gate": causal_dependency_gate(),
        "claim_boundary": {
            "public_checkpoint_head_test": "NOT_TESTED_BECAUSE_UNIVERSAL_REGISTERED_ALGORITHM_GATE_IS_DECISIVE",
            "target_405b_execution": "NOT_TESTED",
            "physical_8gib": "DERIVED_STATIC_LEDGER_ONLY",
            "target_latency": "NOT_TESTED",
        },
        "next_gate": "EXP-106A depth-complete width-thin checkpoint surrogate: derive a training-free target-native causal state source cheaper than one complete layer, freeze full bytes/operations, then require real public-checkpoint A>=339 before any backend",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    result["deterministic_core"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "integrity_failures": result["integrity_failures"],
        "deterministic_core": result["deterministic_core"],
        "best_random_head_fraction": result["random_dense_control"]["best_p50_head_query_fraction"],
        "best_target_late_decision_bytes": result["target_scale_late_decision_ledger"]["best"]["total_draft_bytes_per_token"],
    }, indent=2))


if __name__ == "__main__":
    main()
