#!/usr/bin/env python3
"""Run EXP-066 by reusing frozen validated EXP-065 rearrangement ranks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.tensor_train_rank import (
    TensorTrainModePlan,
    certify_mode_family,
    deinterleave_tensor,
    enumerate_mode_plans,
    select_favorable_tt_plan,
)
from vortex_runtime.tt_kronecker_reuse import (
    derive_reused_tt_plan,
    index_exp065_plan_rows,
    select_favorable_reused_plan,
    validate_cut_equivalence,
)

ROOT = Path(__file__).resolve().parents[2]
EXP065 = ROOT / "results/exp_065"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        normalized = relative.lstrip("* ").removeprefix("./")
        checksums[normalized] = digest
    return checksums


def verify_frozen_inputs() -> tuple[dict[str, Any], int, dict[str, str]]:
    checksum_path = EXP065 / "checksums.sha256"
    expected = parse_checksums(checksum_path)
    required = (
        "summary.json",
        "raw/matrix_rows.jsonl",
        "raw/plan_rows.jsonl",
        "raw/model_rows.jsonl",
        "raw/control_rows.jsonl",
    )
    mismatches = 0
    observed: dict[str, str] = {}
    for relative in required:
        path = EXP065 / relative
        digest = sha256_file(path)
        observed[relative] = digest
        mismatches += int(expected.get(relative) != digest)
    summary = json.loads((EXP065 / "summary.json").read_text(encoding="utf-8"))
    return summary, mismatches, observed


def dense_accounting(
    rows: int,
    columns: int,
    *,
    bits: int,
    activation_bytes: int,
    has_bias: bool,
) -> dict[str, int | float]:
    output_terms = rows + (rows if has_bias else 0)
    operations = rows * columns + output_terms
    storage = (
        math.ceil(rows * columns * bits / 8)
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    query_bytes = (
        math.ceil(rows * columns * bits / 8)
        + rows * columns * activation_bytes
        + rows * 4
        + (rows * 4 if has_bias else 0)
    )
    return {
        "baseline_operations": operations,
        "lower_bound_operations": operations,
        "operation_fraction": 1.0,
        "baseline_storage_bytes": storage,
        "lower_bound_storage_bytes": storage,
        "storage_fraction": 1.0,
        "baseline_query_bytes": query_bytes,
        "lower_bound_query_bytes": query_bytes,
        "query_byte_fraction": 1.0,
    }


def controls(seed: int) -> tuple[list[dict[str, Any]], int, list[float]]:
    rows: list[dict[str, Any]] = []
    failures = 0

    matrix = np.arange(64, dtype=np.int16).reshape(8, 8)
    plan = TensorTrainModePlan(
        variant="equivalence_control",
        row_schedule=(2, 2, 2),
        column_schedule=(2, 2, 2),
        mode_pairs=((2, 2), (2, 2), (2, 2)),
    )
    equivalence = all(
        validate_cut_equivalence(matrix, plan.mode_pairs, cut)
        for cut in range(1, len(plan.mode_pairs))
    )
    failures += int(not equivalence)
    rows.append(
        {
            "control": "tt_cut_equals_exp065_rearrangement",
            "passed": equivalence,
            "cut_count": len(plan.mode_pairs) - 1,
        }
    )

    vectors = (
        np.asarray([1, 2, 3, 5], dtype=np.int16),
        np.asarray([2, 1, 4, 3], dtype=np.int16),
        np.asarray([1, 1, 2, 1], dtype=np.int16),
    )
    tensor = np.einsum("i,j,k->ijk", *vectors, dtype=np.int64).astype(np.int16)
    rank_one_matrix = deinterleave_tensor(tensor, plan.mode_pairs)
    rank_one_plans = certify_mode_family(
        rank_one_matrix, primes=(251, 257), maximum_modes=(2, 4, 8)
    )
    rank_one_ok = bool(rank_one_plans) and min(
        item.maximum_bond_rank for item in rank_one_plans
    ) == 1
    failures += int(not rank_one_ok)
    rows.append(
        {
            "control": "direct_rank_one_mpo",
            "passed": rank_one_ok,
            "minimum_maximum_bond_rank": min(
                item.maximum_bond_rank for item in rank_one_plans
            ),
        }
    )

    adversary_fractions: list[float] = []
    for index in range(4):
        adversary = np.random.default_rng(seed + index).integers(
            -7, 8, size=(8, 8), dtype=np.int16
        )
        plans = certify_mode_family(
            adversary, primes=(251, 257), maximum_modes=(2, 4, 8)
        )
        selected = select_favorable_tt_plan(plans)
        passed = all(item.witness_mismatches == 0 for item in plans)
        failures += int(not passed)
        adversary_fractions.append(selected.operation_fraction)
        rows.append(
            {
                "control": f"dense_random_{index}",
                "passed": passed,
                "mode_plan_count": len(plans),
                "selected_operation_fraction": selected.operation_fraction,
                "selected_storage_fraction": selected.storage_fraction,
                "selected_maximum_bond_rank": selected.maximum_bond_rank,
            }
        )
    return rows, failures, adversary_fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_066/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_066_candidate",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    started = time.perf_counter_ns()
    source_summary, source_checksum_mismatches, source_hashes = (
        verify_frozen_inputs()
    )
    matrix_rows_065 = read_jsonl(EXP065 / "raw/matrix_rows.jsonl")
    plan_rows_065 = read_jsonl(EXP065 / "raw/plan_rows.jsonl")
    source_index = index_exp065_plan_rows(plan_rows_065)

    bits = int(config["bits_per_core"])
    activation_bytes = int(config["activation_bytes"])
    maximum_modes = tuple(int(value) for value in config["maximum_modes"])
    matrix_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    selected_cut_rows: list[dict[str, Any]] = []
    model_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_nontrivial_cuts = 0
    source_witness_mismatches = 0
    total_mapped_cuts = 0
    total_unit_boundary_cuts = 0
    dense_with_any_joint_survivor = 0
    total_joint_surviving_plans = 0

    for source_matrix in matrix_rows_065:
        model_id = str(source_matrix["model_id"])
        revision = str(source_matrix["revision"])
        tensor_name = str(source_matrix["tensor_name"])
        role = str(source_matrix["matrix_role"])
        rows = int(source_matrix["row_count"])
        columns = int(source_matrix["column_count"])
        has_bias = bool(source_matrix["has_bias"])
        mode_plans = tuple(
            plan
            for plan in enumerate_mode_plans(
                rows, columns, maximum_modes=maximum_modes
            )
            if len(plan.mode_pairs) >= 2
        )
        reused = tuple(
            derive_reused_tt_plan(
                model_id=model_id,
                tensor_name=tensor_name,
                rows=rows,
                columns=columns,
                has_bias=has_bias,
                mode_plan=mode_plan,
                exp065_index=source_index,
                bits_per_core=bits,
                activation_bytes=activation_bytes,
            )
            for mode_plan in mode_plans
        )
        if reused:
            selected = select_favorable_reused_plan(reused)
            selected_dict = selected.as_dict()
            survivors = sum(item.meets_joint_p90_screen for item in reused)
            dense_with_any_joint_survivor += int(
                role == "dense_projection" and survivors > 0
            )
            total_joint_surviving_plans += survivors
            missing_nontrivial_cuts += sum(
                item.missing_nontrivial_cut_count for item in reused
            )
            source_witness_mismatches += sum(
                item.source_witness_mismatches for item in reused
            )
            total_mapped_cuts += sum(item.mapped_cut_count for item in reused)
            total_unit_boundary_cuts += sum(
                item.unit_boundary_cut_count for item in reused
            )
            for item in reused:
                plan_rows.append(
                    {
                        "revision": revision,
                        "matrix_role": role,
                        "q4_integer_sha256": source_matrix["q4_integer_sha256"],
                        "selected": item.mode_plan.mode_pairs
                        == selected.mode_plan.mode_pairs,
                        **item.as_dict(),
                    }
                )
            selected_cut_rows.extend(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "tensor_name": tensor_name,
                    "matrix_role": role,
                    "selected_variant": selected.mode_plan.variant,
                    "selected_mode_pairs": [
                        list(pair) for pair in selected.mode_plan.mode_pairs
                    ],
                    **record,
                }
                for record in selected.cut_records
            )
            selected_variant = selected.mode_plan.variant
            selected_modes = [list(pair) for pair in selected.mode_plan.mode_pairs]
            selected_bonds = list(selected.bond_rank_lower_bounds)
            selected_mapped = selected.mapped_cut_count
            selected_units = selected.unit_boundary_cut_count
            selected_missing = selected.missing_nontrivial_cut_count
            selected_source_mismatches = selected.source_witness_mismatches
        else:
            selected_dict = dense_accounting(
                rows,
                columns,
                bits=bits,
                activation_bytes=activation_bytes,
                has_bias=has_bias,
            )
            survivors = 0
            selected_variant = "dense_fallback_no_internal_tt_cut"
            selected_modes = None
            selected_bonds = []
            selected_mapped = 0
            selected_units = 0
            selected_missing = 0
            selected_source_mismatches = 0

        row = {
            "model_id": model_id,
            "revision": revision,
            "tensor_name": tensor_name,
            "matrix_role": role,
            "row_count": rows,
            "column_count": columns,
            "has_bias": has_bias,
            "q4_integer_sha256": source_matrix["q4_integer_sha256"],
            "source_exp065_checksum_match": bool(source_matrix["checksum_match"]),
            "mode_plan_count": len(reused),
            "joint_p90_surviving_plan_count": survivors,
            "selected_variant": selected_variant,
            "selected_mode_pairs": selected_modes,
            "selected_bond_rank_lower_bounds": selected_bonds,
            "selected_mapped_cut_count": selected_mapped,
            "selected_unit_boundary_cut_count": selected_units,
            "selected_missing_nontrivial_cut_count": selected_missing,
            "selected_source_witness_mismatches": selected_source_mismatches,
            "selected_operation_fraction": float(
                selected_dict["operation_fraction"]
            ),
            "selected_storage_fraction": float(
                selected_dict["storage_fraction"]
            ),
            "selected_query_byte_fraction": float(
                selected_dict["query_byte_fraction"]
            ),
            "baseline_storage_bytes": int(
                selected_dict["baseline_storage_bytes"]
            ),
            "selected_lower_bound_storage_bytes": int(
                selected_dict["lower_bound_storage_bytes"]
            ),
        }
        matrix_rows.append(row)
        if role == "dense_projection":
            model_groups[(model_id, revision)].append(row)

    model_rows: list[dict[str, Any]] = []
    for (model_id, revision), rows in sorted(model_groups.items()):
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "dense_projection_count": len(rows),
                "p50_operation_fraction": percentile(
                    [item["selected_operation_fraction"] for item in rows], 0.50
                ),
                "p90_operation_fraction": percentile(
                    [item["selected_operation_fraction"] for item in rows], 0.90
                ),
                "p50_storage_fraction": percentile(
                    [item["selected_storage_fraction"] for item in rows], 0.50
                ),
                "p90_storage_fraction": percentile(
                    [item["selected_storage_fraction"] for item in rows], 0.90
                ),
            }
        )

    control_rows, control_failures, adversary_fractions = controls(
        int(config["seed"])
    )
    dense_rows = [
        row for row in matrix_rows if row["matrix_role"] == "dense_projection"
    ]
    operations = [row["selected_operation_fraction"] for row in dense_rows]
    storage = [row["selected_storage_fraction"] for row in dense_rows]
    query = [row["selected_query_byte_fraction"] for row in dense_rows]
    aggregate_baseline_storage = sum(
        row["baseline_storage_bytes"] for row in dense_rows
    )
    aggregate_lower_storage = sum(
        row["selected_lower_bound_storage_bytes"] for row in dense_rows
    )
    aggregate_storage_fraction = (
        aggregate_lower_storage / aggregate_baseline_storage
    )
    projected_storage = (
        float(config["projection"]["target_q4_bytes"])
        * aggregate_storage_fraction
    )
    model_p50 = {
        row["model_id"]: row["p50_operation_fraction"] for row in model_rows
    }
    largest_id = str(config["models"][-1]["model_id"])
    best_model = min(model_p50.values())
    largest_degradation = model_p50[largest_id] / best_model - 1.0
    gate = config["gate"]

    correctness_gate = (
        source_checksum_mismatches == 0
        and missing_nontrivial_cuts == 0
        and source_witness_mismatches == 0
        and control_failures == 0
        and all(row["source_exp065_checksum_match"] for row in matrix_rows)
    )
    population_gate = (
        len(matrix_rows) == int(gate["expected_two_dimensional_tensors"])
        and len(dense_rows) == int(gate["expected_dense_projections"])
    )
    operation_gate = (
        percentile(operations, 0.50)
        <= float(gate["maximum_p50_operation_fraction"])
        and percentile(operations, 0.90)
        <= float(gate["maximum_p90_operation_fraction"])
    )
    storage_gate = (
        percentile(storage, 0.50)
        <= float(gate["maximum_p50_storage_fraction"])
        and percentile(storage, 0.90)
        <= float(gate["maximum_p90_storage_fraction"])
        and projected_storage
        <= float(gate["maximum_projected_storage_bytes"])
    )
    adversary_gate = percentile(adversary_fractions, 0.50) <= float(
        gate["maximum_dense_random_control_p50"]
    )
    trend_gate = largest_degradation <= float(
        gate["maximum_largest_model_degradation"]
    )
    no_joint_survivor = dense_with_any_joint_survivor == 0
    survives = all(
        (
            correctness_gate,
            population_gate,
            operation_gate,
            storage_gate,
            adversary_gate,
            trend_gate,
        )
    )
    decision = (
        "RETAIN_REAL_Q4_TT_MPO_CANDIDATES_FOR_EXACT_RECONSTRUCTION_GATE"
        if survives
        else str(config["failure_decision"])
    )

    measured = {
        "source_exp065_workflow": source_summary["provenance"].get(
            "workflow_run_id", 30870558294
        ),
        "source_exp065_authoritative_decision": source_summary[
            "authoritative_decision"
        ],
        "source_plan_row_count": len(plan_rows_065),
        "source_checksum_mismatches": source_checksum_mismatches,
        "model_count": len(model_rows),
        "two_dimensional_tensor_count": len(matrix_rows),
        "dense_projection_count": len(dense_rows),
        "tt_mode_plan_row_count": len(plan_rows),
        "selected_cut_row_count": len(selected_cut_rows),
        "mapped_exp065_cut_count": total_mapped_cuts,
        "unit_boundary_lower_bound_cut_count": total_unit_boundary_cuts,
        "missing_nontrivial_exp065_cut_count": missing_nontrivial_cuts,
        "source_witness_mismatches": source_witness_mismatches,
        "control_failures": control_failures,
        "joint_p90_surviving_plan_count": total_joint_surviving_plans,
        "dense_with_any_joint_p90_survivor": dense_with_any_joint_survivor,
        "p50_operation_fraction": percentile(operations, 0.50),
        "p90_operation_fraction": percentile(operations, 0.90),
        "p50_storage_fraction": percentile(storage, 0.50),
        "p90_storage_fraction": percentile(storage, 0.90),
        "p50_query_byte_fraction": percentile(query, 0.50),
        "p90_query_byte_fraction": percentile(query, 0.90),
        "minimum_operation_fraction": min(operations),
        "minimum_storage_fraction": min(storage),
        "minimum_query_byte_fraction": min(query),
        "dense_random_control_p50": percentile(adversary_fractions, 0.50),
        "selected_variant_distribution": dict(
            Counter(row["selected_variant"] for row in dense_rows)
        ),
        "aggregate_lower_bound_storage_fraction": aggregate_storage_fraction,
        "projected_405b_lower_bound_storage_bytes": projected_storage,
        "model_p50_operation_fraction": model_p50,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    gates = {
        "correctness_gate_pass": correctness_gate,
        "population_gate_pass": population_gate,
        "operation_gate_pass": operation_gate,
        "storage_gate_pass": storage_gate,
        "adversary_gate_pass": adversary_gate,
        "model_trend_gate_pass": trend_gate,
        "no_joint_p90_survivor": no_joint_survivor,
    }
    summary = {
        "experiment": "EXP-066",
        "name": "frozen_exp065_derived_exact_tt_mpo_bond_rank_gate",
        "phase": ["A", "B", "C-derived-from-frozen-real-Q4-evidence"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "tt_mpo_lower_bound_survives_gate": survives,
            "decision": decision,
            **gates,
            "exact_integer_mpo_reconstruction_gate_pass": False,
            "derivation": (
                "Every interleaved TT/MPO prefix-suffix unfolding equals the "
                "EXP-065 Kronecker rearrangement whose factors are row and "
                "column prefix/suffix products. Nontrivial cut ranks reuse "
                "frozen validated EXP-065 rows; unit-boundary cuts receive "
                "the universally valid favorable lower bound one."
            ),
        },
        "PROJECTED": {
            "405b_lower_bound_storage_bytes": projected_storage,
            "source": "aggregate measured small-checkpoint storage fraction times 405B Q4 bytes",
        },
        "UNVERIFIED": [
            "exact integer MPO core reconstruction",
            "Q4 model-output preservation",
            "physical MPO contraction kernel",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "new_model_execution": False,
            "new_modular_rank_execution_on_real_weights": False,
            "frozen_real_q4_rank_evidence_reused": True,
            "exact_mpo_reconstruction": "NOT TESTED",
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
            "exp065_input_hashes": source_hashes,
            "exp065_summary_sha256": source_hashes["summary.json"],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    write_jsonl(output / "raw/matrix_rows.jsonl", matrix_rows)
    write_jsonl(output / "raw/plan_rows.jsonl", plan_rows)
    write_jsonl(output / "raw/selected_cut_rows.jsonl", selected_cut_rows)
    write_jsonl(output / "raw/model_rows.jsonl", model_rows)
    write_jsonl(output / "raw/control_rows.jsonl", control_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "processed/aggregate.json", summary)
    write_json(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    output_files = sorted(
        path for path in output.rglob("*") if path.is_file()
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in output_files
        if path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
