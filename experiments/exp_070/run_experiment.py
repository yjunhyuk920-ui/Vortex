#!/usr/bin/env python3
"""Run EXP-070 pinned real-Q4 local-pattern table-circuit Gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp_059.run_experiment import (
    dump, dump_rows, exp057_q4_checksums, git_commit, resolve_snapshot,
    sha256_bytes, sha256_file, write_checksums,
)
from vortex_runtime.local_pattern_table import (
    analyze_local_pattern_plan, choose_joint_plan, registered_orders,
)
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]
EXP058_ROWS = ROOT / "results/exp_058/raw/matrix_rank_rows.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def target_index() -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in read_jsonl(EXP058_ROWS):
        if str(row["matrix_role"]) != "dense_projection":
            continue
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-058 dense projection: {key}")
        indexed[key] = row
    return indexed


def plan_summary(plan: Any) -> dict[str, Any]:
    row = plan.as_dict()
    row.pop("blocks", None)
    return row


def control_population(seed: int) -> tuple[list[dict[str, Any]], int]:
    rows = []
    failures = 0
    base_patterns = np.asarray([
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        [4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1],
        [1, 0, -1, 2, 1, 0, -1, 2, 1, 0, -1, 2, 1, 0, -1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)
    repeated = np.ascontiguousarray(np.tile(base_patterns, (256, 8)))
    repeated_plan = analyze_local_pattern_plan(
        repeated, block_width=16, order_name="natural", order=range(128)
    )
    passed = (
        repeated_plan.reconstruction_mismatches == 0
        and repeated_plan.operation_fraction <= 0.10
        and repeated_plan.query_byte_fraction <= 0.10
        and repeated_plan.static_representation_fraction <= 0.10
    )
    failures += int(not passed)
    rows.append({"control": "repeated_local_patterns", "passed": passed, **plan_summary(repeated_plan)})

    unique = np.asarray(
        [[(row * 3 + col * 5) % 15 - 7 for col in range(16)] for row in range(256)],
        dtype=np.int8,
    )
    for row in range(256):
        for offset in range(4):
            unique[row, offset] = ((row >> (offset * 2)) & 3) - 1
    unique_plan = analyze_local_pattern_plan(
        unique, block_width=16, order_name="natural", order=range(16)
    )
    passed = unique_plan.distinct_pattern_total == 256 and unique_plan.joint_fraction > 0.25
    failures += int(not passed)
    rows.append({"control": "forced_unique_patterns", "passed": passed, **plan_summary(unique_plan)})

    rng = np.random.default_rng(seed)
    random_matrix = rng.integers(-7, 8, size=(256, 64), dtype=np.int8)
    random_plans = []
    for width in (2, 3, 4, 6, 8, 12, 16):
        for order_name, order in registered_orders(
            random_matrix, ("natural", "bit_reversal", "lexicographic_signature")
        ):
            random_plans.append(analyze_local_pattern_plan(
                random_matrix, block_width=width, order_name=order_name, order=order
            ))
    random_plan = choose_joint_plan(random_plans)
    passed = random_plan.joint_fraction > 0.25
    failures += int(not passed)
    rows.append({"control": "dense_random_q4", "passed": passed, **plan_summary(random_plan)})

    mutated = repeated.copy()
    before = analyze_local_pattern_plan(
        repeated, block_width=16, order_name="natural", order=range(128)
    )
    mutated[1, 0] += 1
    after = analyze_local_pattern_plan(
        mutated, block_width=16, order_name="natural", order=range(128)
    )
    passed = after.distinct_pattern_total == before.distinct_pattern_total + 1
    failures += int(not passed)
    rows.append({"control": "one_nibble_mutation", "passed": passed})
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/exp_070/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/exp_070_candidate")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/exp_070_huggingface")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config["torch_num_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(config["torch_num_threads"]))

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(config["torch_num_threads"]))
    targets = target_index()
    expected_checksums = exp057_q4_checksums()
    widths = tuple(int(value) for value in config["block_widths"])
    order_names = tuple(str(value) for value in config["column_orders"])
    tensor_rows, plan_rows, selected_rows, model_rows = [], [], [], []
    checksum_mismatches = missing_tensors = reconstruction_mismatches = collision_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(model_id=model_id, revision=revision, cache_dir=arguments.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        state = model.state_dict()
        model_targets = {
            name: row for (candidate_model, name), row in targets.items()
            if candidate_model == model_id
        }
        local_selected = []
        role_counts = Counter()
        for tensor_name, source in sorted(model_targets.items()):
            tensor = state.get(tensor_name)
            if tensor is None:
                missing_tensors += 1
                continue
            floating = tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            quantized = symmetric_row_quantize(floating, bits=4)
            matrix = np.ascontiguousarray(quantized.values)
            checksum = sha256_bytes(matrix.tobytes())
            expected = expected_checksums.get((model_id, tensor_name))
            checksum_mismatches += int(
                expected is None or checksum != expected or checksum != str(source["q4_integer_sha256"])
            )
            role = str(source["matrix_role"])
            tensor_rows.append({
                "model_id": model_id, "revision": revision, "tensor_name": tensor_name,
                "matrix_role": role, "shape": list(matrix.shape),
                "q4_integer_sha256": checksum,
                "expected_q4_integer_sha256": expected,
                "checksum_match": checksum == expected == str(source["q4_integer_sha256"]),
            })
            plans = []
            tensor_plan_start = len(plan_rows)
            for order_name, order in registered_orders(matrix, order_names):
                for width in widths:
                    plan = analyze_local_pattern_plan(
                        matrix, block_width=width, order_name=order_name, order=order
                    )
                    reconstruction_mismatches += plan.reconstruction_mismatches
                    collision_mismatches += plan.hash_collision_mismatches
                    plans.append(plan)
                    plan_rows.append({
                        "model_id": model_id, "revision": revision,
                        "tensor_name": tensor_name, "matrix_role": role,
                        "selected": False, **plan_summary(plan),
                    })
            selected = choose_joint_plan(plans)
            for row in plan_rows[tensor_plan_start:]:
                if row["block_width"] == selected.block_width and row["order_name"] == selected.order_name:
                    row["selected"] = True
                    break
            selected_row = {
                "model_id": model_id, "revision": revision,
                "tensor_name": tensor_name, "matrix_role": role,
                **plan_summary(selected),
            }
            selected_rows.append(selected_row)
            local_selected.append(selected_row)
            role_counts[role] += 1
        model_rows.append({
            "model_id": model_id, "revision": revision,
            "dense_projection_count": len(local_selected),
            "role_distribution": dict(role_counts),
            "p50_operation_fraction": percentile([row["operation_fraction"] for row in local_selected], 0.50),
            "p90_operation_fraction": percentile([row["operation_fraction"] for row in local_selected], 0.90),
            "p50_query_byte_fraction": percentile([row["query_byte_fraction"] for row in local_selected], 0.50),
            "p90_query_byte_fraction": percentile([row["query_byte_fraction"] for row in local_selected], 0.90),
            "p50_static_representation_fraction": percentile([row["static_representation_fraction"] for row in local_selected], 0.50),
            "p90_static_representation_fraction": percentile([row["static_representation_fraction"] for row in local_selected], 0.90),
            "p50_joint_fraction": percentile([row["joint_fraction"] for row in local_selected], 0.50),
        })
        del model, state

    controls, control_failures = control_population(int(config["seed"]))
    operations = [float(row["operation_fraction"]) for row in selected_rows]
    query = [float(row["query_byte_fraction"]) for row in selected_rows]
    storage = [float(row["static_representation_fraction"]) for row in selected_rows]
    role_rows = defaultdict(list)
    for row in selected_rows:
        role_rows[str(row["matrix_role"])].append(row)
    role_p90 = {
        role: {
            "operation_fraction": percentile([row["operation_fraction"] for row in rows], 0.90),
            "query_byte_fraction": percentile([row["query_byte_fraction"] for row in rows], 0.90),
            "static_representation_fraction": percentile([row["static_representation_fraction"] for row in rows], 0.90),
        }
        for role, rows in sorted(role_rows.items())
    }
    model_joint = {row["model_id"]: float(row["p50_joint_fraction"]) for row in model_rows}
    best_model = min(model_joint.values())
    largest_id = str(config["models"][-1]["model_id"])
    largest_degradation = model_joint[largest_id] / best_model - 1.0 if best_model else math.inf
    gate = config["gate"]
    correctness_gate = (
        checksum_mismatches == 0 and missing_tensors == 0
        and reconstruction_mismatches == 0 and collision_mismatches == 0
        and control_failures == 0
    )
    population_gate = len(selected_rows) == int(gate["expected_dense_projection_count"])
    operation_gate = (
        percentile(operations, 0.50) <= float(gate["maximum_p50_operation_fraction"])
        and percentile(operations, 0.90) <= float(gate["maximum_p90_operation_fraction"])
    )
    query_gate = (
        percentile(query, 0.50) <= float(gate["maximum_p50_query_byte_fraction"])
        and percentile(query, 0.90) <= float(gate["maximum_p90_query_byte_fraction"])
    )
    storage_gate = (
        percentile(storage, 0.50) <= float(gate["maximum_p50_static_representation_fraction"])
        and percentile(storage, 0.90) <= float(gate["maximum_p90_static_representation_fraction"])
    )
    role_gate = all(
        value <= float(gate["maximum_required_role_p90_fraction"])
        for axes in role_p90.values() for value in axes.values()
    )
    trend_gate = largest_degradation <= float(gate["maximum_largest_model_degradation"])
    survives = all((correctness_gate, population_gate, operation_gate, query_gate, storage_gate, role_gate, trend_gate))
    decision = (
        "PROMOTE_EXACT_Q4_LOCAL_PATTERN_TABLE_TO_REPLAY_ORDER_GATE"
        if survives else str(config["failure_decision"])
    )
    measured = {
        "model_count": len(model_rows), "dense_projection_count": len(selected_rows),
        "plan_count": len(plan_rows), "checksum_mismatches": checksum_mismatches,
        "missing_tensors": missing_tensors, "reconstruction_mismatches": reconstruction_mismatches,
        "hash_collision_mismatches": collision_mismatches, "control_failures": control_failures,
        "p50_operation_fraction": percentile(operations, 0.50),
        "p90_operation_fraction": percentile(operations, 0.90),
        "p50_query_byte_fraction": percentile(query, 0.50),
        "p90_query_byte_fraction": percentile(query, 0.90),
        "p50_static_representation_fraction": percentile(storage, 0.50),
        "p90_static_representation_fraction": percentile(storage, 0.90),
        "minimum_joint_fraction": min(float(row["joint_fraction"]) for row in selected_rows),
        "maximum_joint_fraction": max(float(row["joint_fraction"]) for row in selected_rows),
        "selected_block_width_distribution": dict(Counter(str(row["block_width"]) for row in selected_rows)),
        "selected_order_distribution": dict(Counter(str(row["order_name"]) for row in selected_rows)),
        "role_p90_fractions": role_p90, "model_p50_joint_fraction": model_joint,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    summary = {
        "experiment": "EXP-070",
        "name": "pinned_real_q4_exact_local_pattern_table_circuit_gate",
        "phase": ["A", "B", "C-real-Q4-structure"], "evidence_level": "E1",
        "authoritative_decision": decision, "MEASURED": measured,
        "DERIVED": {
            "local_pattern_table_survives_gate": survives, "decision": decision,
            "correctness_gate_pass": correctness_gate, "population_gate_pass": population_gate,
            "operation_gate_pass": operation_gate, "query_byte_gate_pass": query_gate,
            "static_representation_gate_pass": storage_gate,
            "required_role_gate_pass": role_gate, "model_trend_gate_pass": trend_gate,
            "selection_rule": "minimize maximum of operation, query-byte, and static-representation fractions; no per-axis cherry-picking",
            "accounting_scope": "exact Q4 coefficient dictionaries, row pattern IDs, block offsets, non-natural column permutation indexes and moves, unique nonzero partial dots, fused per-row gather-adds, and row-scale storage/multiplication",
        },
        "UNVERIFIED": [
            "bitwise floating-point replay-order equivalence", "physical lookup/table kernel",
            "actual Transformer operation replacement", "405B execution",
            "8 GiB total runtime state", "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "q4_integer_pattern_structure": "MEASURED", "floating_point_replay_order": "NOT TESTED",
            "physical_kernel": "NOT TESTED", "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED", "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(), "config_sha256": sha256_file(arguments.config),
            "exp058_rows_sha256": sha256_file(EXP058_ROWS), "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/tensor_rows.jsonl", tensor_rows)
    dump_rows(output / "raw/plan_rows.jsonl", plan_rows)
    dump_rows(output / "raw/selected_rows.jsonl", selected_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(str(config["evidence_ceiling"]) + "\n", encoding="utf-8")
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
