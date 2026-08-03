#!/usr/bin/env python3
"""Run EXP-060 pinned real-Q4 exact zero-sparsity streaming Gate."""

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
from typing import Any

import numpy as np

from experiments.exp_059.run_experiment import (
    dump,
    dump_rows,
    exp057_q4_checksums,
    git_commit,
    matrix_role,
    percentile,
    resolve_snapshot,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from vortex_runtime.sparse_streaming import (
    compile_registered_sparse_formats,
    select_favorable_sparse_format,
)
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]


def validate_controls(*, seed: int) -> tuple[list[dict[str, Any]], int]:
    sparse = np.zeros((32, 32), dtype=np.int64)
    sparse[0, 0] = 1
    sparse[17, 19] = -2
    dense = np.random.default_rng(seed).integers(1, 8, size=(16, 16), dtype=np.int64)
    isolated = np.ones((16, 16), dtype=np.int64)
    isolated[::2, ::2] = 0
    block = np.zeros((16, 16), dtype=np.int64)
    block[:8, :8] = 3
    definitions = (
        ("highly_sparse", sparse),
        ("dense_random", dense),
        ("isolated_zero", isolated),
        ("block_zero", block),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    for name, matrix in definitions:
        plans = compile_registered_sparse_formats(matrix, materialize=True)
        reconstruction_mismatches = sum(
            int(not np.array_equal(plan.reconstruct(), matrix)) for plan in plans
        )
        selected = select_favorable_sparse_format(plans)
        expectations = {
            "highly_sparse": selected.operation_fraction < 0.10,
            "dense_random": selected.kind == "dense",
            "isolated_zero": next(
                plan for plan in plans if plan.kind == "bsr_8x8"
            ).operation_fraction == 1.0,
            "block_zero": next(
                plan for plan in plans if plan.kind == "bsr_8x8"
            ).operation_fraction == 0.25,
        }
        passed = reconstruction_mismatches == 0 and expectations[name]
        failures += int(not passed)
        rows.append(
            {
                "control": name,
                "shape": list(matrix.shape),
                "selected_format": selected.kind,
                "selected_operation_fraction": selected.operation_fraction,
                "selected_query_byte_fraction": selected.query_byte_fraction,
                "reconstruction_mismatches": reconstruction_mismatches,
                "passed": passed,
                "format_rows": [plan.accounting() for plan in plans],
            }
        )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_060/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_060_candidate"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/exp_060_huggingface"
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
    expected_checksums = exp057_q4_checksums()
    block_shapes = tuple(tuple(int(value) for value in shape) for shape in config["block_shapes"])
    matrix_rows: list[dict[str, Any]] = []
    format_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    checksum_mismatches = 0
    missing_checksums = 0
    unregistered_dense = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id, revision=revision, cache_dir=arguments.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        state = model.state_dict()
        model_2d = 0
        model_dense = 0
        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            model_2d += 1
            role = matrix_role(tensor_name)
            model_dense += int(role == "dense_projection")
            try:
                floating = tensor.detach().cpu().contiguous().numpy().astype(
                    np.float32, copy=False
                )
                q4 = symmetric_row_quantize(floating, bits=4)
                values = np.ascontiguousarray(q4.values)
                checksum = sha256_bytes(values.tobytes())
                expected = expected_checksums.get((model_id, tensor_name))
                missing_checksums += int(expected is None)
                checksum_mismatches += int(
                    expected is not None and expected != checksum
                )
                compile_started = time.perf_counter_ns()
                plans = compile_registered_sparse_formats(
                    values,
                    bits_per_scalar=int(config["bits_per_scalar"]),
                    block_shapes=block_shapes,
                    materialize=False,
                )
                compile_elapsed = time.perf_counter_ns() - compile_started
                selected = select_favorable_sparse_format(plans)
                for plan in plans:
                    row = {
                        "model_id": model_id,
                        "revision": revision,
                        "tensor_name": tensor_name,
                        "matrix_role": role,
                        "q4_integer_sha256": checksum,
                        "selected": plan.kind == selected.kind,
                        **plan.accounting(),
                    }
                    format_rows.append(row)
                matrix_rows.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "tensor_name": tensor_name,
                        "matrix_role": role,
                        "shape": list(values.shape),
                        "q4_integer_sha256": checksum,
                        "exp057_q4_integer_sha256": expected,
                        "q4_checksum_matches_exp057": expected == checksum,
                        "zero_scalar_count": selected.zero_scalar_count,
                        "zero_scalar_fraction": (
                            selected.zero_scalar_count / selected.direct_scalar_terms
                        ),
                        "format_count": len(plans),
                        "format_search_scalar_scans": len(plans) * values.size,
                        "format_compile_elapsed_ns": compile_elapsed,
                        "selected_format": selected.kind,
                        "selected_operation_fraction": selected.operation_fraction,
                        "selected_query_byte_fraction": selected.query_byte_fraction,
                    }
                )
            except Exception:
                if role == "dense_projection":
                    unregistered_dense += 1
                raise
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "two_dimensional_tensor_count": model_2d,
                "dense_projection_tensor_count": model_dense,
            }
        )
        del state
        del model

    controls, control_failures = validate_controls(seed=int(config["seed"]))
    primary = [
        row for row in matrix_rows if row["matrix_role"] == config["primary_matrix_role"]
    ]
    if not primary:
        raise RuntimeError("no primary dense projections registered")
    operation_fractions = [float(row["selected_operation_fraction"]) for row in primary]
    byte_fractions = [float(row["selected_query_byte_fraction"]) for row in primary]
    zero_fractions = [float(row["zero_scalar_fraction"]) for row in primary]
    p50_operations = percentile(operation_fractions, 0.50)
    p90_operations = percentile(operation_fractions, 0.90)
    p50_bytes = percentile(byte_fractions, 0.50)
    p90_bytes = percentile(byte_fractions, 0.90)
    model_p50_operations: dict[str, float] = {}
    for entry in config["models"]:
        model_id = str(entry["model_id"])
        model_p50_operations[model_id] = percentile(
            [
                float(row["selected_operation_fraction"])
                for row in primary
                if row["model_id"] == model_id
            ],
            0.50,
        )
    ordered = [model_p50_operations[str(entry["model_id"])] for entry in config["models"]]
    degradation = max(0.0, ordered[-1] / ordered[0] - 1.0)
    gate = config["early_gate"]
    correctness_pass = control_failures <= int(
        gate["maximum_reconstruction_or_control_mismatches"]
    )
    registration_pass = (
        unregistered_dense <= int(gate["maximum_unregistered_dense_projections"])
        and len(primary) == 144
    )
    checksum_pass = (
        missing_checksums == 0
        and checksum_mismatches
        <= int(gate["maximum_q4_checksum_mismatches_against_exp_057"])
    )
    operation_pass = (
        p50_operations <= float(gate["maximum_p50_operation_fraction"])
        and p90_operations <= float(gate["maximum_p90_operation_fraction"])
    )
    byte_pass = (
        p50_bytes <= float(gate["maximum_p50_query_byte_fraction"])
        and p90_bytes <= float(gate["maximum_p90_query_byte_fraction"])
    )
    trend_pass = degradation <= float(gate["maximum_model_size_degradation"])
    survives = all(
        (correctness_pass, registration_pass, checksum_pass, operation_pass, byte_pass, trend_pass)
    )
    decision = (
        "PROMOTE_REAL_Q4_EXACT_ZERO_SPARSITY_TO_PHYSICAL_KERNEL_GATE"
        if survives
        else str(config["failure_decision"])
    )
    distribution = Counter(str(row["selected_format"]) for row in primary)
    best = min(
        primary,
        key=lambda row: (
            float(row["selected_operation_fraction"]),
            float(row["selected_query_byte_fraction"]),
            str(row["model_id"]),
            str(row["tensor_name"]),
        ),
    )
    summary = {
        "experiment": "EXP-060",
        "name": "pinned_real_q4_exact_zero_sparsity_streaming_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "reconstruction_control_gate_pass": correctness_pass,
            "registration_gate_pass": registration_pass,
            "q4_checksum_gate_pass": checksum_pass,
            "operation_gate_pass": operation_pass,
            "query_byte_gate_pass": byte_pass,
            "model_size_trend_gate_pass": trend_pass,
            "real_q4_zero_sparsity_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "exact scalar zeros only; packed Q4 values, indexes, row pointers, "
                "run metadata, nonzero-block internal slots and edge padding charged"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "two_dimensional_tensor_count": len(matrix_rows),
            "dense_projection_matrix_count": len(primary),
            "registered_format_count": 3 + len(block_shapes),
            "format_row_count": len(format_rows),
            "q4_checksum_mismatches_against_exp_057": checksum_mismatches,
            "missing_exp_057_q4_checksum_count": missing_checksums,
            "unregistered_dense_projection_count": unregistered_dense,
            "reconstruction_or_control_mismatches": control_failures,
            "control_row_count": len(controls),
            "p50_zero_scalar_fraction": percentile(zero_fractions, 0.50),
            "p90_zero_scalar_fraction": percentile(zero_fractions, 0.90),
            "maximum_zero_scalar_fraction": max(zero_fractions),
            "p50_operation_fraction": p50_operations,
            "p90_operation_fraction": p90_operations,
            "p50_query_byte_fraction": p50_bytes,
            "p90_query_byte_fraction": p90_bytes,
            "best_real_matrix_operation_fraction": float(
                best["selected_operation_fraction"]
            ),
            "best_real_matrix_query_byte_fraction": float(
                best["selected_query_byte_fraction"]
            ),
            "best_real_matrix_model": best["model_id"],
            "best_real_matrix_tensor": best["tensor_name"],
            "best_real_matrix_format": best["selected_format"],
            "selected_format_distribution": dict(sorted(distribution.items())),
            "model_p50_operation_fraction": model_p50_operations,
            "model_size_degradation_fraction": degradation,
            "total_format_search_scalar_scans": sum(
                int(row["format_search_scalar_scans"]) for row in matrix_rows
            ),
            "total_format_compile_elapsed_ns": sum(
                int(row["format_compile_elapsed_ns"]) for row in matrix_rows
            ),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "UNVERIFIED": [
            "Q4 model output preservation",
            "physical CSR run or BSR kernels",
            "actual Transformer operation replacement",
            "70B or 405B zero sparsity",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "q4_output_preservation": "NOT TESTED",
            "physical_sparse_kernel": "NOT TESTED",
            "real_transformer_operation_replacement": False,
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp057_summary_sha256": sha256_file(ROOT / "results/exp_057/summary.json"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/matrix_rows.jsonl", matrix_rows)
    dump_rows(output / "raw/format_rows.jsonl", format_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(
        output / "artifacts/environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )
    (output / "artifacts/contract.txt").write_text(
        "Phase C observation only. Q4 output preservation, physical sparse "
        "kernels, 405B, 8 GiB and target hardware are NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
