#!/usr/bin/env python3
"""Run EXP-064 pinned real-Q4 exact output-row structure Gate."""
from __future__ import annotations

import argparse
from collections import Counter
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
    dump,
    dump_rows,
    exp057_q4_checksums,
    git_commit,
    matrix_role,
    resolve_snapshot,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from vortex_runtime.output_row_structure import (
    compile_output_row_plans,
    select_output_row_plan,
)
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def bias_name(weight_name: str) -> str | None:
    return weight_name[:-6] + "bias" if weight_name.endswith("weight") else None


def deployable(plans):
    selected = select_output_row_plan(plans)
    dense = next(plan for plan in plans if plan.mechanism == "dense")
    return (
        selected
        if selected.operation_fraction < 1.0
        and selected.query_byte_fraction < 1.0
        else dense
    )


def control_population(
    seed: int, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], int, list[float]]:
    base = (np.arange(32, dtype=np.int16) % 8) - 4
    identical = np.tile(base, (32, 1))
    sign = np.vstack([base, -base] * 16)
    sparse_delta = identical.copy()
    sparse_delta[np.arange(32), np.arange(32)] = np.clip(
        sparse_delta[np.arange(32), np.arange(32)] + 1, -7, 7
    )
    dense = np.random.default_rng(seed).integers(
        -7, 8, size=(32, 32), dtype=np.int16
    )
    unique = np.random.default_rng(seed + 1).integers(
        -7, 8, size=(32, 32), dtype=np.int16
    )
    unique[:, 0] = np.arange(-16, 16, dtype=np.int16)
    mutated = np.vstack([base, base.copy()])
    mutated[1, 7] += 1
    definitions = (
        ("identical_rows", identical),
        ("sign_rows", sign),
        ("sparse_delta", sparse_delta),
        ("dense_random", dense),
        ("forced_unique", unique),
        ("one_nibble_mutation", mutated),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    adversary: list[float] = []
    for name, matrix in definitions:
        plans = compile_output_row_plans(
            matrix,
            scales=np.linspace(
                0.1, 1.0, matrix.shape[0], dtype=np.float32
            ),
            bits_per_weight=int(config["bits_per_weight"]),
            activation_bytes=int(config["activation_bytes"]),
            has_bias=True,
            prototype_counts=tuple(config["prototype_counts"]),
            candidate_cap=int(config["prototype_candidate_cap"]),
        )
        selected = deployable(plans)
        identical_plan = next(
            plan for plan in plans if plan.mechanism == "identical_rows"
        )
        sign_plan = next(
            plan for plan in plans if plan.mechanism == "sign_canonical_rows"
        )
        expectation = {
            "identical_rows": selected.mechanism == "identical_rows"
            and selected.operation_fraction < 0.20,
            "sign_rows": selected.mechanism == "sign_canonical_rows"
            and selected.operation_fraction < 0.20,
            "sparse_delta": selected.mechanism.startswith(
                "prototype_sparse_delta"
            )
            and selected.residual_scalar_fraction < 0.10,
            "dense_random": selected.mechanism == "dense",
            "forced_unique": identical_plan.duplicate_row_count == 0
            and sign_plan.duplicate_row_count == 0,
            "one_nibble_mutation": identical_plan.duplicate_row_count == 0,
        }[name]
        exact = all(plan.reconstruction_mismatches == 0 for plan in plans)
        passed = expectation and exact
        failures += int(not passed)
        if name in {"dense_random", "forced_unique"}:
            adversary.append(selected.operation_fraction)
        rows.append(
            {
                "control": name,
                "shape": list(matrix.shape),
                "selected": selected.accounting(),
                "identical_prototype_count": identical_plan.prototype_count,
                "sign_prototype_count": sign_plan.prototype_count,
                "all_reconstruction_exact": exact,
                "passed": passed,
                "plans": [plan.accounting() for plan in plans],
            }
        )
    return rows, failures, adversary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_064/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_064_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_064_huggingface",
    )
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
    expected = exp057_q4_checksums()
    matrix_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    checksum_mismatches = 0
    missing_checksums = 0
    reconstruction_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        state = model.state_dict()
        local: list[dict[str, Any]] = []
        two_d = 0
        dense_count = 0
        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            two_d += 1
            role = matrix_role(tensor_name)
            dense_count += int(role == "dense_projection")
            floating = (
                tensor.detach()
                .cpu()
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            )
            quantized = symmetric_row_quantize(floating, bits=4)
            values = np.ascontiguousarray(quantized.values)
            checksum = sha256_bytes(values.tobytes())
            expected_checksum = expected.get((model_id, tensor_name))
            missing_checksums += int(expected_checksum is None)
            checksum_mismatches += int(
                expected_checksum is not None and expected_checksum != checksum
            )
            candidate_bias = bias_name(tensor_name)
            has_bias = bool(
                candidate_bias
                and candidate_bias in state
                and state[candidate_bias].ndim == 1
                and int(state[candidate_bias].numel()) == values.shape[0]
            )
            compile_started = time.perf_counter_ns()
            plans = compile_output_row_plans(
                values,
                scales=quantized.scales,
                bits_per_weight=int(config["bits_per_weight"]),
                activation_bytes=int(config["activation_bytes"]),
                has_bias=has_bias,
                prototype_counts=tuple(config["prototype_counts"]),
                candidate_cap=int(config["prototype_candidate_cap"]),
            )
            compile_ns = time.perf_counter_ns() - compile_started
            selected = deployable(plans)
            dense = next(plan for plan in plans if plan.mechanism == "dense")
            identical_plan = next(
                plan for plan in plans if plan.mechanism == "identical_rows"
            )
            sign_plan = next(
                plan for plan in plans if plan.mechanism == "sign_canonical_rows"
            )
            reconstruction_mismatches += sum(
                plan.reconstruction_mismatches for plan in plans
            )
            for plan in plans:
                plan_rows.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "tensor_name": tensor_name,
                        "matrix_role": role,
                        "q4_integer_sha256": checksum,
                        "selected": plan.mechanism == selected.mechanism,
                        **plan.accounting(),
                    }
                )
            row = {
                "model_id": model_id,
                "revision": revision,
                "tensor_name": tensor_name,
                "matrix_role": role,
                "row_count": int(values.shape[0]),
                "column_count": int(values.shape[1]),
                "has_bias": has_bias,
                "q4_integer_sha256": checksum,
                "expected_q4_integer_sha256": expected_checksum,
                "checksum_match": expected_checksum == checksum,
                "scale_sha256": sha256_bytes(
                    np.ascontiguousarray(quantized.scales).tobytes()
                ),
                "compile_elapsed_ns": compile_ns,
                "plan_count": len(plans),
                "selected_mechanism": selected.mechanism,
                "selected_operation_fraction": selected.operation_fraction,
                "selected_query_byte_fraction": selected.query_byte_fraction,
                "selected_static_storage_bytes": selected.static_storage_bytes,
                "dense_static_storage_bytes": dense.static_storage_bytes,
                "selected_storage_fraction": (
                    selected.static_storage_bytes / dense.static_storage_bytes
                ),
                "selected_residual_scalar_fraction": (
                    selected.residual_scalar_fraction
                ),
                "identical_duplicate_row_count": (
                    identical_plan.duplicate_row_count
                ),
                "sign_duplicate_row_count": sign_plan.duplicate_row_count,
                "sign_negative_row_count": sign_plan.negative_row_count,
                "reconstruction_mismatches": sum(
                    plan.reconstruction_mismatches for plan in plans
                ),
            }
            matrix_rows.append(row)
            if role == "dense_projection":
                local.append(row)
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "two_dimensional_tensor_count": two_d,
                "dense_projection_count": dense_count,
                "p50_operation_fraction": percentile(
                    [row["selected_operation_fraction"] for row in local], 0.50
                ),
                "p90_operation_fraction": percentile(
                    [row["selected_operation_fraction"] for row in local], 0.90
                ),
                "p50_query_byte_fraction": percentile(
                    [row["selected_query_byte_fraction"] for row in local], 0.50
                ),
                "p90_query_byte_fraction": percentile(
                    [row["selected_query_byte_fraction"] for row in local], 0.90
                ),
            }
        )
        del model, state

    control_rows, control_failures, adversary = control_population(
        int(config["seed"]), config
    )
    dense_rows = [
        row for row in matrix_rows if row["matrix_role"] == "dense_projection"
    ]
    operation = [row["selected_operation_fraction"] for row in dense_rows]
    byte = [row["selected_query_byte_fraction"] for row in dense_rows]
    source_storage = sum(
        row["dense_static_storage_bytes"] for row in dense_rows
    )
    selected_storage = sum(
        row["selected_static_storage_bytes"] for row in dense_rows
    )
    storage_fraction = selected_storage / source_storage
    projected = (
        float(config["projection"]["target_q4_bytes"]) * storage_fraction
    )
    model_p50 = {
        row["model_id"]: row["p50_operation_fraction"] for row in model_rows
    }
    largest_id = str(config["models"][-1]["model_id"])
    baseline_min = min(model_p50.values())
    largest_degradation = (
        model_p50[largest_id] / baseline_min - 1.0
        if baseline_min > 0
        else math.inf
    )
    gate = config["gate"]
    measured = {
        "model_count": len(model_rows),
        "two_dimensional_tensor_count": len(matrix_rows),
        "dense_projection_count": len(dense_rows),
        "plan_row_count": len(plan_rows),
        "checksum_mismatches": checksum_mismatches,
        "missing_checksums": missing_checksums,
        "reconstruction_mismatches": reconstruction_mismatches,
        "control_failures": control_failures,
        "p50_operation_fraction": percentile(operation, 0.50),
        "p90_operation_fraction": percentile(operation, 0.90),
        "p50_query_byte_fraction": percentile(byte, 0.50),
        "p90_query_byte_fraction": percentile(byte, 0.90),
        "minimum_operation_fraction": min(operation),
        "minimum_query_byte_fraction": min(byte),
        "dense_unique_control_p50": percentile(adversary, 0.50),
        "matrix_with_identical_duplicates": sum(
            row["identical_duplicate_row_count"] > 0 for row in dense_rows
        ),
        "matrix_with_sign_duplicates": sum(
            row["sign_duplicate_row_count"] > 0 for row in dense_rows
        ),
        "selected_mechanism_distribution": dict(
            Counter(row["selected_mechanism"] for row in dense_rows)
        ),
        "p50_selected_residual_scalar_fraction": percentile(
            [
                row["selected_residual_scalar_fraction"]
                for row in dense_rows
            ],
            0.50,
        ),
        "aggregate_static_storage_fraction": storage_fraction,
        "projected_405b_static_storage_bytes": projected,
        "model_p50_operation_fraction": model_p50,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    gates = {
        "correctness_gate_pass": checksum_mismatches == 0
        and missing_checksums == 0
        and reconstruction_mismatches == 0
        and control_failures == 0,
        "population_gate_pass": len(matrix_rows)
        == int(gate["expected_two_dimensional_tensors"])
        and len(dense_rows) == int(gate["expected_dense_projections"]),
        "operation_gate_pass": measured["p50_operation_fraction"]
        <= float(gate["maximum_p50_operation_fraction"])
        and measured["p90_operation_fraction"]
        <= float(gate["maximum_p90_operation_fraction"]),
        "query_byte_gate_pass": measured["p50_query_byte_fraction"]
        <= float(gate["maximum_p50_query_byte_fraction"])
        and measured["p90_query_byte_fraction"]
        <= float(gate["maximum_p90_query_byte_fraction"]),
        "adversary_gate_pass": measured["dense_unique_control_p50"]
        <= float(gate["maximum_dense_unique_control_p50"]),
        "storage_gate_pass": projected
        <= float(gate["maximum_projected_storage_bytes"]),
        "model_trend_gate_pass": largest_degradation
        <= float(gate["maximum_largest_model_degradation"]),
    }
    survives = all(gates.values())
    decision = (
        "PROMOTE_REAL_Q4_OUTPUT_ROW_PROTOTYPE_TO_OPERATION_REPLACEMENT_GATE"
        if survives
        else str(config["failure_decision"])
    )
    summary = {
        "experiment": "EXP-064",
        "name": "pinned_real_q4_exact_output_row_prototype_sparse_delta_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "output_row_structure_survives_gate": survives,
            "decision": decision,
            **gates,
            "accounting_scope": (
                "integer prototype dots, residual terms, output accumulator "
                "reads/copies/signs, per-row scales, all biases, mappings, "
                "indexes, activation reads and static bytes charged"
            ),
        },
        "UNVERIFIED": [
            "Q4 model-output preservation",
            "physical output-row kernel",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "q4_output_preservation": "NOT TESTED",
            "physical_kernel": "NOT TESTED",
            "real_transformer_operation_replacement": False,
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/matrix_rows.jsonl", matrix_rows)
    dump_rows(output / "raw/plan_rows.jsonl", plan_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gates["correctness_gate_pass"] or not gates["population_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
