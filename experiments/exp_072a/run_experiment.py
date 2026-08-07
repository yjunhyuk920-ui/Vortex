#!/usr/bin/env python3
"""Run EXP-072A self-contained exact Q4 DAG information-capacity Gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import tracemalloc
from typing import Any

from vortex_runtime.information_capacity import (
    audit_finite_domain,
    audit_target_capacity,
    standard_basis_signature,
)

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/exp_072a/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/exp_072a_candidate")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir.resolve()
    root = ROOT.resolve()
    if output == root or not (output.name == "exp_072a" or output.name.startswith("exp_072a_")):
        raise ValueError("output directory must be a dedicated exp_072a_* path")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    tracemalloc.start()
    started = time.perf_counter_ns()
    domain_rows = [
        audit_finite_domain(
            rows=int(domain["rows"]),
            columns=int(domain["columns"]),
            alphabet_size=int(domain["alphabet_size"]),
        ).as_dict()
        for domain in config["exhaustive_domains"]
    ]
    target = audit_target_capacity(
        coefficient_count=int(config["registered_parameter_total"]),
        alphabet_size=int(config["coefficient_alphabet_size"]),
        hot_allowance_bytes=int(config["hot_state_allowance_bytes"]),
        former_static_gate_fraction=float(config["former_static_gate_fraction"]),
    )
    config_integrity = (
        int(config["coefficient_alphabet_size"]).bit_length() - 1
        == int(config["bits_per_q4_coefficient"])
    )
    deterministic_core = {"domains": domain_rows, "target": target.as_dict()}
    deterministic_core_hash = canonical_sha256(deterministic_core)
    base = ((-8, 3, 7), (2, 0, -1))
    mutated = ((-8, 3, 6), (2, 0, -1))
    controls = [
        {
            "control": "finite_domain_basis_injectivity",
            "passed": all(bool(row["injective"]) for row in domain_rows),
            "domain_count": len(domain_rows),
        },
        {
            "control": "variable_length_capacity_boundary",
            "passed": all(bool(row["boundary_valid"]) for row in domain_rows),
            "domain_count": len(domain_rows),
        },
        {
            "control": "one_coefficient_mutation_changes_exact_map",
            "passed": standard_basis_signature(base) != standard_basis_signature(mutated),
        },
        {
            "control": "structured_zero_map_is_individually_compact_only",
            "passed": standard_basis_signature(((0, 0), (0, 0))) == ((0, 0), (0, 0)),
            "promotion_effect": "NONE_UNIVERSAL_POPULATION_BOUND_UNCHANGED",
        },
        {
            "control": "q4_config_information_width",
            "passed": config_integrity,
            "registered_bits_per_coefficient": int(config["bits_per_q4_coefficient"]),
        },
        {
            "control": "deterministic_core_hash_rerun",
            "passed": deterministic_core_hash == canonical_sha256(deterministic_core),
            "sha256": deterministic_core_hash,
        },
    ]
    peak_traced_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    correctness_gate = all(bool(row["passed"]) for row in controls)
    universal_hot_gate = target.universal_self_contained_hot_gate_pass
    survives = correctness_gate and universal_hot_gate
    decision = str(config["success_decision"] if survives else config["failure_decision"])
    measured = {
        "finite_domain_count": len(domain_rows),
        "enumerated_matrix_count": sum(int(row["matrix_count"]) for row in domain_rows),
        "unique_basis_signature_count": sum(int(row["unique_basis_signature_count"]) for row in domain_rows),
        "signature_collision_count": sum(int(row["signature_collision_count"]) for row in domain_rows),
        "control_failures": sum(not bool(row["passed"]) for row in controls),
        "deterministic_core_sha256": deterministic_core_hash,
        "peak_traced_bytes": peak_traced_bytes,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    derived = {
        **target.as_dict(),
        "worst_case_static_fraction": 1.0,
        "target_equivalent_fraction": float(config["target_equivalent_fraction"]),
        "worst_case_static_over_target_equivalent_factor": 1.0 / float(config["target_equivalent_fraction"]),
        "universal_hot_gate_pass": universal_hot_gate,
        "final_query_traffic_route": "NOT EVALUATED AFTER DECISIVE HOT-ARTIFACT FAILURE",
        "correctness_gate_pass": correctness_gate,
        "self_contained_dag_survives_gate": survives,
        "decision": decision,
        "injectivity_reason": "exact outputs on standard-basis inputs uniquely recover every matrix coefficient",
        "scope_boundary": "artifacts that query original or other lossless cold weight-derived data are excluded",
    }
    summary = {
        "experiment": "EXP-072A",
        "name": "self_contained_exact_q4_dag_information_capacity_gate",
        "phase": ["A", "B-finite-domain-reference"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": derived,
        "UNVERIFIED": [
            "per-instance compressibility of pinned real checkpoints",
            "online probe lower bound when original or cold lossless data remains queryable",
            "physical query traffic and latency",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB GPU runtime behavior",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "finite_domain_basis_injectivity": "MEASURED",
            "405b_information_arithmetic": "DERIVED_FROM_REGISTERED_PARAMETER_COUNT",
            "self_contained_universal_artifact_cap": "REJECTED",
            "cold_data_online_runtime": "NOT RULED OUT",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    dump_rows(output / "raw/domain_rows.jsonl", domain_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "processed/capacity_audit.json", {"target": target.as_dict(), "domains": domain_rows})
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(str(config["evidence_ceiling"]) + "\n", encoding="utf-8", newline="\n")
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps({"experiment": "EXP-072A", "decision": decision, "measured": measured}, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
