#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vortex_runtime.differential_spanning_tree import (
    best_orientation_bound,
    certified_orientation_bound,
    exact_mst_hamming_weight,
)
from vortex_runtime.weight_structure import symmetric_row_quantize


CONFIG_PATH = ROOT / "experiments/exp_082a/config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_worktree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile population is empty")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def tensor_name(layer: int, family: str) -> str:
    base = f"model.language_model.layers.{layer}"
    if family in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"{base}.self_attn.{family}.weight"
    if family in {"gate_proj", "up_proj", "down_proj"}:
        return f"{base}.mlp.{family}.weight"
    raise ValueError(f"unregistered projection family: {family}")


def synthetic_controls(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = config["stage_1"]
    block_size = int(stage["coordinate_block_size"])
    chunk_rows = int(stage["pair_chunk_rows"])
    rng = np.random.default_rng(int(config["seed"]))
    rows: list[dict[str, Any]] = []

    cases: list[tuple[str, np.ndarray]] = [
        ("all_zero", np.zeros((5, 7), dtype=np.int16)),
        ("identity", np.eye(8, dtype=np.int16)),
        (
            "chain",
            np.tril(np.ones((9, 9), dtype=np.int16)),
        ),
        (
            "duplicates",
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                dtype=np.int16,
            ),
        ),
    ]
    for index in range(32):
        height = 2 + index % 8
        width = 2 + (index * 3) % 9
        cases.append(
            (
                f"random_{index:02d}",
                rng.integers(-7, 8, size=(height, width), dtype=np.int16),
            )
        )

    for name, matrix in cases:
        for orientation in ("rows", "columns"):
            bound = certified_orientation_bound(
                matrix,
                orientation=orientation,
                block_size=min(block_size, matrix.shape[1] if orientation == "rows" else matrix.shape[0]),
                chunk_rows=min(chunk_rows, max(matrix.shape) + 1),
            )
            exact = exact_mst_hamming_weight(matrix, orientation=orientation)
            rows.append(
                {
                    "case": name,
                    "orientation": orientation,
                    "shape": list(matrix.shape),
                    "input_sha256": hashlib.sha256(
                        np.ascontiguousarray(matrix).tobytes()
                    ).hexdigest(),
                    "certified_lower_bound": bound.mst_hamming_lower_bound,
                    "exact_mst_weight": exact,
                    "bound_nonnegative": bound.mst_hamming_lower_bound >= 0,
                    "bound_no_greater_than_exact": bound.mst_hamming_lower_bound
                    <= exact,
                }
            )
    return rows


def orientation_row(bound: Any) -> dict[str, Any]:
    return asdict(bound)


def aggregate_rows(
    matrix_rows: list[dict[str, Any]], *, target_fraction: float
) -> dict[str, Any]:
    dense = sum(int(row["dense_coefficients"]) for row in matrix_rows)
    lower = sum(int(row["selected_lower_bound"]) for row in matrix_rows)
    fractions = [float(row["operation_fraction_lower_bound"]) for row in matrix_rows]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        by_family[str(row["family"])].append(row)
    families: dict[str, dict[str, Any]] = {}
    for family, rows in sorted(by_family.items()):
        family_dense = sum(int(row["dense_coefficients"]) for row in rows)
        family_lower = sum(int(row["selected_lower_bound"]) for row in rows)
        families[family] = {
            "matrix_count": len(rows),
            "dense_coefficients": family_dense,
            "selected_lower_bound": family_lower,
            "weighted_operation_fraction_lower_bound": family_lower / family_dense,
            "p50_matrix_fraction_lower_bound": percentile(
                [float(row["operation_fraction_lower_bound"]) for row in rows], 0.50
            ),
        }
    weighted = lower / dense
    return {
        "matrix_count": len(matrix_rows),
        "dense_coefficients": dense,
        "selected_lower_bound": lower,
        "weighted_operation_fraction_lower_bound": weighted,
        "p50_matrix_fraction_lower_bound": percentile(fractions, 0.50),
        "p90_matrix_fraction_lower_bound": percentile(fractions, 0.90),
        "minimum_matrix_fraction_lower_bound": min(fractions),
        "maximum_matrix_fraction_lower_bound": max(fractions),
        "target_fraction": target_fraction,
        "gap_to_target": weighted / target_fraction,
        "stage_1_reject": weighted > target_fraction,
        "families": families,
    }


def write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
        for path in paths
    ]
    (output_dir / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_commit = git_commit()
    source_clean = git_worktree_clean()
    if not source_clean:
        raise RuntimeError("source worktree must be clean before evidence execution")

    started = time.perf_counter()
    checkpoint = config["checkpoint"]
    model_path = ROOT / checkpoint["local_path"]
    weight_path = model_path / "model.safetensors-00001-of-00001.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"pinned weight payload is missing: {weight_path}")
    actual_weight_sha256 = sha256_file(weight_path)
    if actual_weight_sha256 != checkpoint["weight_sha256"]:
        raise RuntimeError("pinned checkpoint weight SHA-256 mismatch")

    controls = synthetic_controls(config)
    control_failures = sum(
        not row["bound_nonnegative"] or not row["bound_no_greater_than_exact"]
        for row in controls
    )
    if control_failures:
        decision = config["decisions"]["control_failure"]
        raise RuntimeError(f"{decision}: {control_failures} failed controls")

    try:
        import torch
        from safetensors import safe_open
    except Exception as error:  # pragma: no cover - infrastructure path
        raise RuntimeError("torch/safetensors dependencies are unavailable") from error

    stage = config["stage_1"]
    population = config["population"]
    matrix_rows: list[dict[str, Any]] = []
    logs: list[str] = []
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for layer in population["layer_indices"]:
            for family in population["projection_names"]:
                name = tensor_name(int(layer), str(family))
                if name not in available:
                    raise RuntimeError(f"registered tensor is missing: {name}")
                floating = handle.get_tensor(name).float().cpu().numpy()
                quantized = symmetric_row_quantize(
                    floating, bits=int(population["weight_bits"])
                )
                values = np.ascontiguousarray(quantized.values)
                bound = best_orientation_bound(
                    values,
                    block_size=int(stage["coordinate_block_size"]),
                    chunk_rows=int(stage["pair_chunk_rows"]),
                )
                row = {
                    "tensor": name,
                    "family": str(family),
                    "layer": int(layer),
                    "shape": list(values.shape),
                    "dense_coefficients": bound.dense_coefficients,
                    "q4_integer_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "q4_zero_fraction": float(np.count_nonzero(values == 0) / values.size),
                    "quantization_maximum_absolute_error": quantized.maximum_absolute_error,
                    "quantization_mean_absolute_error": quantized.mean_absolute_error,
                    "rows": orientation_row(bound.rows),
                    "columns": orientation_row(bound.columns),
                    "selected_orientation": bound.selected_orientation,
                    "selected_lower_bound": bound.selected_lower_bound,
                    "operation_fraction_lower_bound": bound.operation_fraction_lower_bound,
                }
                matrix_rows.append(row)
                log_row = {
                    "tensor": name,
                    "shape": list(values.shape),
                    "selected_orientation": bound.selected_orientation,
                    "lower_bound_fraction": bound.operation_fraction_lower_bound,
                    "row_universal_blocks": bound.rows.universally_distinct_blocks,
                    "column_universal_blocks": bound.columns.universally_distinct_blocks,
                }
                logs.append(json.dumps(log_row, sort_keys=True))
                print(logs[-1], flush=True)
                del floating, values, quantized

    expected_count = len(population["layer_indices"]) * len(
        population["projection_names"]
    )
    if len(matrix_rows) != expected_count:
        raise RuntimeError(
            f"registered tensor count mismatch: {len(matrix_rows)} != {expected_count}"
        )

    target_fraction = float(stage["maximum_weighted_operation_fraction"])
    aggregate = aggregate_rows(matrix_rows, target_fraction=target_fraction)
    decision = (
        config["decisions"]["stage_1_reject"]
        if aggregate["stage_1_reject"]
        else config["decisions"]["stage_1_inconclusive"]
    )
    deterministic_core = canonical_sha256(
        {
            "config_sha256": sha256_file(CONFIG_PATH),
            "controls": controls,
            "matrices": matrix_rows,
            "aggregate": aggregate,
            "decision": decision,
        }
    )
    elapsed = time.perf_counter() - started
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }
    summary = {
        "experiment": config["experiment"],
        "name": config["name"],
        "decision": decision,
        "evidence_ceiling": config["evidence_ceiling"],
        "source_commit": source_commit,
        "source_worktree_clean": source_clean,
        "config_sha256": sha256_file(CONFIG_PATH),
        "weight_sha256": actual_weight_sha256,
        "control_count": len(controls),
        "control_failures": control_failures,
        "aggregate": aggregate,
        "deterministic_core_sha256": deterministic_core,
        "elapsed_seconds": elapsed,
        "not_tested": [
            "exact model-scale MST construction",
            "sparse-delta operation replacement",
            "BF16/Q4 model-output preservation",
            "physical CUDA, SSD, PCIe, H2D, VRAM, or latency",
            "122B and 405B execution",
            "E2-E7",
        ],
    }

    dump_rows(output_dir / "raw/control_rows.jsonl", controls)
    dump_rows(output_dir / "raw/matrix_rows.jsonl", matrix_rows)
    dump(output_dir / "processed/aggregate.json", aggregate)
    dump(output_dir / "artifacts/environment.json", environment)
    (output_dir / "artifacts/contract.txt").write_text(
        json.dumps(
            {
                "config": config,
                "config_sha256": sha256_file(CONFIG_PATH),
                "source_commit": source_commit,
                "weight_sha256": actual_weight_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "logs/run.log").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs/run.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    dump(output_dir / "summary.json", summary)
    write_checksums(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    try:
        summary = run(output_dir)
    except Exception as error:
        if output_dir.exists():
            dump(
                output_dir / "infrastructure_failure.json",
                {
                    "decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "weighted_operation_fraction_lower_bound": summary["aggregate"][
                    "weighted_operation_fraction_lower_bound"
                ],
                "target_fraction": summary["aggregate"]["target_fraction"],
                "deterministic_core_sha256": summary["deterministic_core_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
