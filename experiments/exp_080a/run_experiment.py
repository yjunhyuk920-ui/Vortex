#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.hyperblock_fmm import (
    TensorFamily,
    evaluate_block,
    naive_matrix_multiply,
    strassen_matrix_multiply,
    target_equivalent_fraction,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_inputs(config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    registered = config["registered_inputs"]
    shape_path = ROOT / registered["target_shape_path"]
    if sha256_file(shape_path) != registered["target_shape_sha256"]:
        raise ValueError("target shape SHA-256 mismatch")
    paths["target_shapes"] = shape_path
    for name, row in registered["prior_block_summaries"].items():
        path = ROOT / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{name} summary SHA-256 mismatch")
        paths[name] = path
    return paths


def load_families(config: dict[str, Any], path: Path) -> tuple[TensorFamily, ...]:
    excluded = set(config["candidate"]["exclude_tensor_families"])
    families: list[TensorFamily] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["tensor"] in excluded:
            continue
        families.append(
            TensorFamily(
                name=row["tensor"],
                rows=int(row["rows"]),
                columns=int(row["columns"]),
                count=int(row["count"]),
            )
        )
    if not families:
        raise ValueError("registered shape population is empty")
    return tuple(families)


def run_exact_controls(config: dict[str, Any]) -> list[dict[str, Any]]:
    control = config["controls"]
    rng = random.Random(int(control["seed"]))
    rows: list[dict[str, Any]] = []
    for width in control["matrix_widths"]:
        width = int(width)
        leaves = []
        leaf = 1
        while leaf <= width:
            leaves.append(leaf)
            leaf *= 2
        for trial in range(int(control["trials_per_width"])):
            left = [
                [
                    rng.randint(
                        int(control["minimum_value"]),
                        int(control["maximum_value"]),
                    )
                    for _ in range(width)
                ]
                for _ in range(width)
            ]
            right = [
                [
                    rng.randint(
                        int(control["minimum_value"]),
                        int(control["maximum_value"]),
                    )
                    for _ in range(width)
                ]
                for _ in range(width)
            ]
            expected = naive_matrix_multiply(left, right)
            input_hash = canonical_sha256({"left": left, "right": right})
            expected_hash = canonical_sha256(expected)
            for leaf_size in leaves:
                actual = strassen_matrix_multiply(
                    left, right, leaf_size=leaf_size
                )
                rows.append(
                    {
                        "width": width,
                        "trial": trial,
                        "leaf_size": leaf_size,
                        "input_sha256": input_hash,
                        "expected_sha256": expected_hash,
                        "actual_sha256": canonical_sha256(actual),
                        "exact_match": actual == expected,
                    }
                )
    return rows


def prior_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    exp048 = json.loads(paths["exp_048"].read_text(encoding="utf-8"))
    exp049 = json.loads(paths["exp_049"].read_text(encoding="utf-8"))
    exp050 = json.loads(paths["exp_050"].read_text(encoding="utf-8"))
    return {
        "exp_048": {
            "decision": exp048["authoritative_decision"],
            "hard_jacobi_passes_per_32_tokens_p50": exp048["MEASURED"][
                "b2_p50_target_full_streams_per_32_tokens"
            ],
            "hard_jacobi_target_fraction_p50": exp048["MEASURED"][
                "b2_p50_target_equivalent_stream_fraction"
            ],
            "perfect_future_block_size": exp048["MEASURED"][
                "b1_oracle_block_size"
            ],
            "perfect_future_target_fraction": exp048["MEASURED"][
                "b1_oracle_target_equivalent_stream_fraction"
            ],
        },
        "exp_049": {
            "decision": exp049["authoritative_decision"],
            "oracle_best_prefix_p50": exp049["MEASURED"][
                "oracle_best_s1s2_p50_matching_prefix"
            ],
            "oracle_best_prefix_max": exp049["MEASURED"][
                "oracle_best_s1s2_max_matching_prefix"
            ],
            "triangular_barrier_observed": exp049["MEASURED"][
                "triangular_one_position_per_round_barrier_observed"
            ],
        },
        "exp_050": {
            "decision": exp050["authoritative_decision"],
            "external_draft_prefix_p50": exp050["MEASURED"][
                "favorable_pool_p50_matching_prefix"
            ],
            "external_draft_prefix_max": exp050["MEASURED"][
                "favorable_pool_maximum_matching_prefix"
            ],
            "required_perfect_4b_prefix": exp050["PROJECTED"][
                "perfect_4b_draft_minimum_exact_prefix"
            ],
        },
    }


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    started = time.perf_counter_ns()
    config_path = Path(arguments.config).resolve()
    output = Path(arguments.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_paths = validate_inputs(config)
    families = load_families(config, input_paths["target_shapes"])
    control_rows = run_exact_controls(config)
    control_mismatches = sum(not row["exact_match"] for row in control_rows)
    prior = prior_evidence(input_paths)

    contract = config["target_contract"]
    candidate = config["candidate"]
    p50_fraction = target_equivalent_fraction(
        baseline_billions=float(contract["baseline_billions"]),
        target_billions=float(contract["target_billions"]),
        latency_multiple=float(contract["p50_latency_multiple"]),
    )
    p95_fraction = target_equivalent_fraction(
        baseline_billions=float(contract["baseline_billions"]),
        target_billions=float(contract["target_billions"]),
        latency_multiple=float(contract["p95_latency_multiple"]),
    )
    if abs(p50_fraction - float(contract["p50_target_fraction"])) > 1e-15:
        raise ValueError("registered p50 fraction mismatch")
    if abs(p95_fraction - float(contract["p95_target_fraction"])) > 1e-15:
        raise ValueError("registered p95 fraction mismatch")

    block_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    scenarios = (
        ("perfect_future_zero_proposal", float(candidate["authoritative_proposal_billions"])),
        ("perfect_future_4b_draft_diagnostic", float(candidate["diagnostic_draft_billions"])),
    )
    for scenario, proposal_billions in scenarios:
        for block_length in candidate["block_lengths"]:
            evaluation, constructive_plans, omega_plans = evaluate_block(
                families,
                block_length=int(block_length),
                target_sweeps=float(candidate["authoritative_target_sweeps"]),
                proposal_billions=proposal_billions,
                target_billions=float(contract["target_billions"]),
                allowed_fraction=p50_fraction,
                peak_vram_bytes=int(contract["peak_vram_bytes"]),
                omega=float(candidate["theoretical_omega"]),
                q4_weight_bytes_per_scalar=float(
                    candidate["q4_weight_bytes_per_scalar"]
                ),
                activation_bytes_per_scalar=int(
                    candidate["activation_bytes_per_scalar"]
                ),
                scratch_bytes_per_scalar=int(candidate["scratch_bytes_per_scalar"]),
                scratch_square_tile_count=int(
                    candidate["scratch_square_tile_count"]
                ),
            )
            row = {"scenario": scenario, **evaluation.as_dict()}
            block_rows.append(row)
            for algorithm, plans in (
                ("constructive_strassen", constructive_plans),
                ("unit_constant_omega_oracle", omega_plans),
            ):
                for plan in plans:
                    family_rows.append(
                        {
                            "scenario": scenario,
                            "algorithm": algorithm,
                            **plan.as_dict(),
                        }
                    )
            line = json.dumps(
                {
                    "scenario": scenario,
                    "K": block_length,
                    "traffic_fraction": evaluation.traffic_fraction,
                    "constructive_fraction": evaluation.constructive_total_fraction,
                    "omega_oracle_fraction": evaluation.omega_oracle_total_fraction,
                    "constructive_speedup": 1.0 / evaluation.constructive_ratio,
                    "required_packing_factor": evaluation.required_constructive_packing_factor,
                    "workspace_gib": evaluation.favorable_workspace_bytes / 2**30,
                    "constructive_joint_pass": evaluation.constructive_joint_pass,
                    "omega_oracle_joint_pass": evaluation.omega_oracle_joint_pass,
                },
                sort_keys=True,
            )
            print(line, flush=True)
            log_lines.append(line)

    authoritative = [
        row for row in block_rows if row["scenario"] == "perfect_future_zero_proposal"
    ]
    diagnostic_draft = [
        row
        for row in block_rows
        if row["scenario"] == "perfect_future_4b_draft_diagnostic"
    ]
    constructive_passes = [row for row in authoritative if row["constructive_joint_pass"]]
    omega_passes = [row for row in authoritative if row["omega_oracle_joint_pass"]]
    draft_omega_passes = [row for row in diagnostic_draft if row["omega_oracle_joint_pass"]]
    best_constructive = min(
        authoritative, key=lambda row: row["constructive_total_fraction"]
    )
    if control_mismatches:
        decision = config["gate"]["decision_on_control_failure"]
    elif constructive_passes:
        decision = config["gate"]["decision_on_pass"]
    else:
        decision = config["gate"]["decision_on_scientific_failure"]

    shape_rows = [
        {
            "family": family.name,
            "rows": family.rows,
            "columns": family.columns,
            "count": family.count,
            "parameters": family.parameters,
        }
        for family in families
    ]
    deterministic_core = {
        "control_rows": control_rows,
        "shape_rows": shape_rows,
        "block_rows": block_rows,
        "prior_evidence": prior,
        "decision": decision,
    }
    core_hash = canonical_sha256(deterministic_core)
    source_commit = git_commit()
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": source_commit,
        "config_sha256": sha256_file(config_path),
    }
    summary = {
        "experiment": "EXP-080A",
        "name": config["name"],
        "phase": ["A-theory", "B-synthetic-reference"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "target_shape_path": config["registered_inputs"]["target_shape_path"],
            "target_shape_sha256": config["registered_inputs"][
                "target_shape_sha256"
            ],
            "prior_block_summaries": config["registered_inputs"][
                "prior_block_summaries"
            ],
        },
        "REUSED_COMMITTED_EVIDENCE": prior,
        "MEASURED": {
            "exact_control_case_count": len(control_rows),
            "exact_control_mismatches": control_mismatches,
            "wall_ns": time.perf_counter_ns() - started,
            "deterministic_core_sha256": core_hash,
        },
        "DERIVED": {
            "p50_target_fraction": p50_fraction,
            "p95_target_fraction": p95_fraction,
            "registered_family_count": len(families),
            "registered_dense_operation_parameters": sum(
                family.parameters for family in families
            ),
            "authoritative_block_rows": authoritative,
            "diagnostic_4b_draft_rows": diagnostic_draft,
            "best_constructive_row": best_constructive,
            "constructive_joint_pass_count": len(constructive_passes),
            "first_omega_oracle_pass_K": (
                min(row["block_length"] for row in omega_passes)
                if omega_passes
                else None
            ),
            "first_4b_draft_omega_oracle_pass_K": (
                min(row["block_length"] for row in draft_omega_passes)
                if draft_omega_passes
                else None
            ),
            "gate": {
                "controls_passed": control_mismatches == 0,
                "constructive_joint_p50_passed": bool(constructive_passes),
                "unit_constant_omega_has_theoretical_headroom": bool(omega_passes),
                "causal_future_source_passed": False,
            },
            "decision": decision,
        },
        "UNVERIFIED": [
            "any deployable causal source for the exact future activation block",
            "practical low-constant algorithm attaining omega 2.371552",
            "exact packed Q4 arithmetic and BF16 reduction-order compatibility",
            "nonlinear attention MLP normalization and KV block scheduling",
            "online first-token and inter-token latency without burst hiding",
            "physical CUDA PCIe SSD throughput power and peak VRAM",
            "122B and arbitrary dense 405B execution",
            "E2 through E7",
        ],
        "claim_boundary": {
            "future_information": "NON_DEPLOYABLE_PERFECT_FUTURE_ACTIVATION_ORACLE",
            "constructive_algorithm": "STANDARD_RECURSIVE_STRASSEN_COUNTS_ONLY",
            "omega_control": "UNIT_CONSTANT_NON_CONSTRUCTIVE_DIAGNOSTIC",
            "operation_replacement": "NOT_EXECUTED",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
            "model_payload": "NO_CHECKPOINT_LOADED_OR_DOWNLOADED",
        },
        "provenance": environment,
    }

    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump_rows(output / "raw/shape_rows.jsonl", shape_rows)
    dump_rows(output / "raw/block_rows.jsonl", block_rows)
    dump_rows(output / "raw/family_plan_rows.jsonl", family_rows)
    dump(output / "raw/prior_evidence.json", prior)
    dump(
        output / "processed/aggregate.json",
        {
            "authoritative": authoritative,
            "diagnostic_4b_draft": diagnostic_draft,
            "best_constructive": best_constructive,
            "constructive_passes": constructive_passes,
            "omega_passes": omega_passes,
            "draft_omega_passes": draft_omega_passes,
            "decision": decision,
        },
    )
    dump(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
    )
    dump(output / "summary.json", summary)
    write_checksums(output)
    print(
        json.dumps(
            {
                "decision": decision,
                "control_mismatches": control_mismatches,
                "constructive_joint_pass_count": len(constructive_passes),
                "best_constructive_row": best_constructive,
                "first_omega_oracle_pass_K": summary["DERIVED"][
                    "first_omega_oracle_pass_K"
                ],
                "first_4b_draft_omega_oracle_pass_K": summary["DERIVED"][
                    "first_4b_draft_omega_oracle_pass_K"
                ],
                "deterministic_core_sha256": core_hash,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
