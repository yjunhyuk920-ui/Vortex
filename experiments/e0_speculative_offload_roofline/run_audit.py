"""Deterministic E0 dual-roofline audit for exact speculative offloading.

The audit is intentionally favorable.  It assumes perfect overlap between the
compressed checkpoint stream and dense candidate verification, ignores every
attention/KV/draft/synchronization cost, and charges only the larger of the two
unavoidable terms.  A failure under this lower bound is decisive; a pass is not
an implementation or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

GIB = 1 << 30


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    return data


def compute_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    mission = config["mission"]

    parameters = int(mission["model_parameters"])
    if parameters <= 0:
        raise ValueError("model_parameters must be positive")

    exact_bits = _positive(
        mission["exact_weight_bits_per_parameter"],
        "exact_weight_bits_per_parameter",
    )
    compression = _positive(
        mission["favorable_lossless_compression_ratio"],
        "favorable_lossless_compression_ratio",
    )
    hot_budget_gib = _positive(mission["gpu_hot_budget_gib"], "gpu_hot_budget_gib")
    host_free_gib = _positive(
        mission["registered_host_free_storage_gib"],
        "registered_host_free_storage_gib",
    )
    bandwidth_gib_s = _positive(
        mission["registered_host_to_device_bandwidth_gib_per_s"],
        "registered_host_to_device_bandwidth_gib_per_s",
    )
    peak_ops_s = _positive(
        mission["favorable_device_peak_ops_per_s"],
        "favorable_device_peak_ops_per_s",
    )
    p50_multiplier = _positive(
        mission["p50_relative_multiplier"], "p50_relative_multiplier"
    )
    p95_multiplier = _positive(
        mission["p95_relative_multiplier"], "p95_relative_multiplier"
    )

    raw_weight_gib = parameters * exact_bits / 8.0 / GIB
    compressed_weight_gib = raw_weight_gib / compression
    target_sweep_io_s = compressed_weight_gib / bandwidth_gib_s

    # One dense matrix-vector-equivalent candidate column uses the optimistic
    # transformer parameter floor 2P scalar operations.  This gives the draft
    # tree every possible benefit from perfect GEMM utilization and overlap.
    dense_candidate_floor_s = 2.0 * parameters / peak_ops_s
    io_compute_crossover_nodes = target_sweep_io_s / dense_candidate_floor_s
    crossover_chain_tokens = math.ceil(io_compute_crossover_nodes)

    max_whole_model_bits_per_parameter_in_hot_budget = (
        hot_budget_gib * GIB * 8.0 / parameters
    )
    one_bit_substitute_gib = parameters / 8.0 / GIB
    four_bit_substitute_gib = parameters * 4.0 / 8.0 / GIB

    critical_native_4b_p50_ms = (
        dense_candidate_floor_s / p50_multiplier * 1000.0
    )
    critical_native_4b_p95_ms = (
        dense_candidate_floor_s / p95_multiplier * 1000.0
    )

    baseline_rows: list[dict[str, Any]] = []
    for baseline_ms_raw in config["conditional_native_4b_baselines_ms"]:
        baseline_ms = _positive(baseline_ms_raw, "conditional baseline")
        baseline_s = baseline_ms / 1000.0
        p50_allowance_s = p50_multiplier * baseline_s
        max_candidate_inflation = p50_allowance_s / dense_candidate_floor_s
        baseline_rows.append(
            {
                "native_4b_p50_ms": baseline_ms,
                "p50_target_allowance_ms": p50_allowance_s * 1000.0,
                "minimum_accepted_tokens_for_io": math.ceil(
                    target_sweep_io_s / p50_allowance_s
                ),
                "maximum_verified_nodes_per_accepted_token": max_candidate_inflation,
                "maximum_fine_dense_arithmetic_fraction_for_exact_chain": min(
                    1.0, max_candidate_inflation
                ),
                "unreduced_dense_chain_semantically_feasible": (
                    max_candidate_inflation >= 1.0
                ),
            }
        )

    scenarios = list(config["diagnostic_scenarios"])
    scenarios.append(
        {
            "id": "perfect_chain_at_io_compute_crossover",
            "verified_nodes": float(crossover_chain_tokens),
            "accepted_tokens": float(crossover_chain_tokens),
            "classification": "derived_oracle",
            "note": "Shortest integer perfect chain whose optimistic dense compute time reaches the compressed target-sweep I/O floor.",
        }
    )

    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        verified_nodes = _positive(scenario["verified_nodes"], "verified_nodes")
        accepted_tokens = _positive(scenario["accepted_tokens"], "accepted_tokens")
        if accepted_tokens > verified_nodes + 1e-12:
            raise ValueError(
                f"accepted_tokens cannot exceed verified_nodes in {scenario['id']}"
            )
        tree_inflation = verified_nodes / accepted_tokens
        compute_cycle_s = dense_candidate_floor_s * verified_nodes
        cycle_floor_s = max(target_sweep_io_s, compute_cycle_s)
        per_token_floor_s = cycle_floor_s / accepted_tokens
        scenario_rows.append(
            {
                "id": str(scenario["id"]),
                "classification": str(scenario["classification"]),
                "note": str(scenario["note"]),
                "verified_nodes": verified_nodes,
                "accepted_tokens": accepted_tokens,
                "verified_nodes_per_accepted_token": tree_inflation,
                "target_sweep_io_s": target_sweep_io_s,
                "dense_compute_cycle_s": compute_cycle_s,
                "cycle_floor_s": cycle_floor_s,
                "dominant_floor": (
                    "io" if target_sweep_io_s >= compute_cycle_s else "compute"
                ),
                "per_accepted_token_floor_s": per_token_floor_s,
                "required_native_4b_p50_ms": (
                    per_token_floor_s / p50_multiplier * 1000.0
                ),
                "required_native_4b_p95_ms": (
                    per_token_floor_s / p95_multiplier * 1000.0
                ),
            }
        )

    core: dict[str, Any] = {
        "schema_version": 1,
        "authoritative_decision": (
            "RETAIN_LOSSLESS_OFFLOAD_AND_OVERLAP_AS_AUXILIARY_"
            "REQUIRE_FINE_DENSE_ARITHMETIC_REDUCTION_FOR_CORE"
        ),
        "claim_boundary": {
            "equation_kind": "optimistic lower bound",
            "perfect_io_compute_overlap": True,
            "draft_attention_kv_sync_cost": "GRANTED_FREE",
            "actual_native_4b_baseline": "NOT_TESTED",
            "target_405b_execution": "NOT_TESTED",
            "exact_successor_state_integration": "NOT_TESTED",
        },
        "mission_inputs": {
            "model_parameters": parameters,
            "exact_weight_bits_per_parameter": exact_bits,
            "favorable_lossless_compression_ratio": compression,
            "gpu_hot_budget_gib": hot_budget_gib,
            "registered_host_free_storage_gib": host_free_gib,
            "registered_host_to_device_bandwidth_gib_per_s": bandwidth_gib_s,
            "favorable_device_peak_ops_per_s": peak_ops_s,
            "p50_relative_multiplier": p50_multiplier,
            "p95_relative_multiplier": p95_multiplier,
        },
        "target_floor": {
            "raw_exact_weight_gib": raw_weight_gib,
            "compressed_exact_weight_gib": compressed_weight_gib,
            "registered_host_storage_deficit_gib": max(
                0.0, compressed_weight_gib - host_free_gib
            ),
            "compressed_target_sweep_io_s": target_sweep_io_s,
            "dense_candidate_floor_s": dense_candidate_floor_s,
            "io_compute_crossover_verified_nodes": io_compute_crossover_nodes,
            "crossover_perfect_chain_tokens": crossover_chain_tokens,
            "critical_native_4b_p50_ms_for_unreduced_dense_chain": critical_native_4b_p50_ms,
            "critical_native_4b_p95_ms_for_unreduced_dense_chain": critical_native_4b_p95_ms,
        },
        "draft_residency_floor": {
            "full_model_four_bit_substitute_gib": four_bit_substitute_gib,
            "full_model_one_bit_substitute_gib": one_bit_substitute_gib,
            "maximum_whole_model_bits_per_parameter_in_8gib": max_whole_model_bits_per_parameter_in_hot_budget,
            "full_model_one_bit_substitute_fits_8gib": one_bit_substitute_gib
            <= hot_budget_gib,
        },
        "conditional_native_4b_rows": baseline_rows,
        "diagnostic_scenarios": scenario_rows,
        "literature": list(config["literature"]),
        "promotion_requirements": [
            "causal checkpoint-derived source available before target continuation",
            "exact token and successor-state equivalence under the frozen ABI",
            "measured accepted tokens A and target-verified nodes N",
            "compressed sweep condition S/(B*A) within the native-4B allowance",
            "fine arithmetic condition r*(N/A)*(2P/F) within the native-4B allowance",
            "complete 8-GiB hot-state and local-source-storage ledger",
            "no training, changed checkpoint, hidden compute, or uncharged fallback",
        ],
    }

    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    core["deterministic_core_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return core


def write_summary(summary: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compute_audit(load_config(args.config))
    write_summary(summary, args.output)


if __name__ == "__main__":
    main()
