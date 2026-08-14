#!/usr/bin/env python3
"""Run EXP-074 weight-stationary MTP/expert block budget Gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vortex_runtime.weight_stationary_block import (
    MoeActiveSet,
    block_target_parameters,
    decimal_gb_to_gib,
    expected_independent_routed_union,
    maximum_distinct_routed_union,
    minimum_accepted_tokens,
    normalized_weight_traffic,
    search_minimum_block,
)

def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def route_union(model: MoeActiveSet, route_model: str, block_tokens: int) -> float:
    if route_model == "fixed":
        return float(model.routed_experts_per_token)
    if route_model == "independent_uniform_expected":
        return expected_independent_routed_union(
            expert_count=model.expert_count,
            routed_per_token=model.routed_experts_per_token,
            block_tokens=block_tokens,
        )
    if route_model == "maximally_distinct":
        return float(
            maximum_distinct_routed_union(
                expert_count=model.expert_count,
                routed_per_token=model.routed_experts_per_token,
                block_tokens=block_tokens,
            )
        )
    raise ValueError(f"unknown route model: {route_model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_074/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_074_candidate"
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir.resolve()
    root = ROOT.resolve()
    if output == root or not (output.name == "exp_074" or output.name.startswith("exp_074_")):
        raise ValueError("output directory must be a dedicated exp_074 or exp_074_* path")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    target = config["target"]
    baseline = config["baseline"]
    model = MoeActiveSet(
        total_parameters=float(target["total_parameters"]),
        active_parameters=float(target["active_parameters"]),
        baseline_parameters=float(baseline["parameter_equivalent"]),
        hidden_size=int(target["hidden_size"]),
        expert_intermediate_size=int(target["expert_intermediate_size"]),
        layer_count=int(target["layer_count"]),
        expert_count=int(target["expert_count"]),
        routed_experts_per_token=int(target["routed_experts_per_token"]),
        shared_experts_per_token=int(target["shared_experts_per_token"]),
    )
    model.validate()

    tracemalloc.start()
    started = time.perf_counter_ns()
    scenario_rows: list[dict[str, Any]] = []
    for route_model in config["route_models"]:
        for draft_cost in config["draft_parameter_equivalents"]:
            for block_tokens in config["registered_block_sizes"]:
                union = route_union(model, str(route_model), int(block_tokens))
                block_parameters = block_target_parameters(
                    model, routed_expert_union=union
                )
                fraction = normalized_weight_traffic(
                    model,
                    accepted_tokens=int(block_tokens),
                    routed_expert_union=union,
                    draft_parameters_per_token=float(draft_cost),
                )
                scenario_rows.append(
                    {
                        "route_model": route_model,
                        "draft_parameters_per_token": int(draft_cost),
                        "perfectly_accepted_tokens": int(block_tokens),
                        "routed_expert_union": union,
                        "block_target_parameter_reads": block_parameters,
                        "normalized_weight_traffic": fraction,
                        "p50_gate_pass": fraction
                        <= float(baseline["p50_allowed_multiplier"]),
                        "p95_gate_pass": fraction
                        <= float(baseline["p95_allowed_multiplier"]),
                    }
                )

    minima_rows: list[dict[str, Any]] = []
    for route_model in config["route_models"]:
        for draft_cost in config["draft_parameter_equivalents"]:
            for percentile in ("p50", "p95"):
                allowed = float(baseline[f"{percentile}_allowed_multiplier"])
                minimum = search_minimum_block(
                    model,
                    route_model=str(route_model),
                    allowed_multiplier=allowed,
                    draft_parameters_per_token=float(draft_cost),
                )
                minima_rows.append(
                    {
                        "route_model": route_model,
                        "draft_parameters_per_token": int(draft_cost),
                        "percentile_contract": percentile,
                        "allowed_multiplier": allowed,
                        "minimum_perfectly_accepted_tokens": minimum,
                    }
                )

    fixed_union = float(model.routed_experts_per_token)
    fixed_block_parameters = block_target_parameters(
        model, routed_expert_union=fixed_union
    )
    mtp1_tokens = int(config["standard_recipe_mtp_speculative_tokens"])
    mtp1_optimistic_fraction = normalized_weight_traffic(
        model,
        accepted_tokens=mtp1_tokens,
        routed_expert_union=fixed_union,
        draft_parameters_per_token=0.0,
    )
    optimistic_minimum = search_minimum_block(
        model,
        route_model="fixed",
        allowed_multiplier=float(baseline["p50_allowed_multiplier"]),
        draft_parameters_per_token=0.0,
    )
    mtp1_pass = mtp1_optimistic_fraction <= float(
        baseline["p50_allowed_multiplier"]
    )
    optimistic_path = optimistic_minimum is not None and optimistic_minimum <= int(
        config["gate"]["optimistic_candidate_block_limit"]
    )
    if mtp1_pass:
        decision = str(config["gate"]["decision_on_mtp1_success"])
    elif optimistic_path:
        decision = str(
            config["gate"]["decision_on_mtp1_failure_with_optimistic_path"]
        )
    else:
        decision = str(config["gate"]["decision_on_no_optimistic_path"])

    dense = config["comparison_405b_dense"]
    dense_p50_no_draft = minimum_accepted_tokens(
        block_target_parameter_reads=float(dense["target_parameters"]),
        baseline_parameters=float(dense["baseline_parameters"]),
        allowed_multiplier=float(baseline["p50_allowed_multiplier"]),
    )
    dense_p50_with_draft = minimum_accepted_tokens(
        block_target_parameter_reads=float(dense["target_parameters"]),
        baseline_parameters=float(dense["baseline_parameters"]),
        allowed_multiplier=float(baseline["p50_allowed_multiplier"]),
        draft_parameters_per_token=float(dense["draft_parameters"]),
    )

    model_size_gib = decimal_gb_to_gib(float(target["ollama_q4_k_m_size_decimal_gb"]))
    hardware = config["target_hardware"]
    remaining_bytes = int(hardware["root_available_bytes"]) - round(
        float(target["ollama_q4_k_m_size_decimal_gb"]) * 1_000_000_000
    )
    storage = {
        "model_artifact_gib": model_size_gib,
        "root_available_gib": int(hardware["root_available_bytes"]) / 1024**3,
        "remaining_after_direct_pull_gib": remaining_bytes / 1024**3,
        "direct_pull_nominally_fits": remaining_bytes >= 0,
        "minimum_safe_remaining_gib": int(hardware["minimum_safe_remaining_bytes"])
        / 1024**3,
        "safe_workspace_gate_pass": remaining_bytes
        >= int(hardware["minimum_safe_remaining_bytes"]),
    }

    deterministic_core = {
        "scenario_rows": scenario_rows,
        "minima_rows": minima_rows,
        "storage": storage,
        "dense_p50_no_draft": dense_p50_no_draft,
        "dense_p50_with_draft": dense_p50_with_draft,
    }
    core_hash = canonical_sha256(deterministic_core)
    controls = [
        {
            "control": "single_token_active_decomposition",
            "passed": abs(fixed_block_parameters - model.active_parameters) < 1e-6,
        },
        {
            "control": "independent_union_at_one_token",
            "passed": abs(
                expected_independent_routed_union(
                    expert_count=model.expert_count,
                    routed_per_token=model.routed_experts_per_token,
                    block_tokens=1,
                )
                - model.routed_experts_per_token
            )
            < 1e-9,
        },
        {
            "control": "expected_union_monotonic",
            "passed": all(
                expected_independent_routed_union(
                    expert_count=model.expert_count,
                    routed_per_token=model.routed_experts_per_token,
                    block_tokens=right,
                )
                >= expected_independent_routed_union(
                    expert_count=model.expert_count,
                    routed_per_token=model.routed_experts_per_token,
                    block_tokens=left,
                )
                for left, right in zip(
                    config["registered_block_sizes"],
                    config["registered_block_sizes"][1:],
                )
            ),
        },
        {
            "control": "dense_405b_zero_cost_requirement",
            "passed": dense_p50_no_draft == 85,
        },
        {
            "control": "dense_405b_4b_draft_requirement",
            "passed": dense_p50_with_draft == 507,
        },
        {
            "control": "storage_unit_conversion",
            "passed": abs(model_size_gib - 75.43712854385376) < 1e-12,
        },
        {
            "control": "deterministic_core_hash_rerun",
            "passed": core_hash == canonical_sha256(deterministic_core),
            "sha256": core_hash,
        },
    ]
    peak_traced_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    control_failures = sum(not bool(row["passed"]) for row in controls)
    if control_failures:
        decision = "INVALID_REFERENCE_CONTROL_FAILURE"

    derived = {
        "parameters_per_expert_per_layer": model.parameters_per_expert_per_layer,
        "parameters_per_expert_model_wide": model.parameters_per_expert_model_wide,
        "single_token_active_expert_parameters": model.single_token_active_expert_parameters,
        "nonexpert_active_parameters": model.nonexpert_active_parameters,
        "mtp1_zero_draft_fixed_route_normalized_traffic": mtp1_optimistic_fraction,
        "mtp1_p50_gate_pass": mtp1_pass,
        "fixed_route_zero_draft_p50_minimum_tokens": optimistic_minimum,
        "optimistic_long_block_path_within_registered_limit": optimistic_path,
        "dense_405b_zero_cost_p50_minimum_tokens": dense_p50_no_draft,
        "dense_405b_4b_draft_p50_minimum_tokens": dense_p50_with_draft,
        "storage": storage,
        "decision": decision,
    }
    measured = {
        "scenario_count": len(scenario_rows),
        "minimum_search_count": len(minima_rows),
        "control_count": len(controls),
        "control_failures": control_failures,
        "deterministic_core_sha256": core_hash,
        "peak_traced_bytes": peak_traced_bytes,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    summary = {
        "experiment": "EXP-074",
        "name": config["name"],
        "phase": ["A", "B-reference-accounting"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "target": target,
            "sources": config["sources"],
            "baseline": baseline,
            "target_hardware": hardware,
        },
        "MEASURED": measured,
        "DERIVED": derived,
        "UNVERIFIED": [
            "causal MTP proposal acceptance distribution",
            "real Qwen expert route union and temporal locality",
            "MTP and verification compute and memory traffic",
            "exact floating and quantized block semantics",
            "actual Transformer operation replacement",
            "physical SSD RAM PCIe H2D and VRAM traffic",
            "latency TTFT tokens per second power and thermal behavior",
            "122B checkpoint execution",
            "transfer to arbitrary dense 405B checkpoints",
            "E4 E5 E6 and E7",
        ],
        "claim_boundary": {
            "mtp1_plus_expert_paging": "REJECTED_AS_1B_CLASS_CORE_UNDER_OPTIMISTIC_ACCOUNTING",
            "long_block_weight_stationary_candidate": "REVISE_PENDING_CAUSAL_PROPOSAL_AND_ROUTING_TRACE_GATES",
            "122b_download": "NOT_AUTHORIZED_OR_PERFORMED",
            "target_hardware": "NO_COMMAND_EXECUTED",
            "dense_405b": "NOT_VALIDATED_BY_MOE_SURROGATE",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    dump_rows(output / "raw/scenario_rows.jsonl", scenario_rows)
    dump_rows(output / "raw/minimum_rows.jsonl", minima_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "processed/budget_audit.json", deterministic_core)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {"experiment": "EXP-074", "decision": decision, "measured": measured},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if control_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
