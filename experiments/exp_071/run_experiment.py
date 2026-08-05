#!/usr/bin/env python3
"""Run EXP-071 exact dense-runtime lower-bound applicability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np

from vortex_runtime.lower_bound_audit import (
    GIB,
    cgl2015_unit_constant_indicator,
    ckl2018_applicability,
    direct_sum_audit,
    exhaustive_binary_projection_reduction,
    llama_405b_tensor_plan,
    packed_q4_cells,
    parameter_total,
    q4_field_embedding_supported,
)

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def theorem_hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "source": "CGL15",
            "theorem": "Theorem 3",
            "problem": "static online n x n matrix-vector multiplication over finite field F",
            "representation": "general S cells of w bits",
            "query": "v in F^n unavailable during preprocessing",
            "error": "average error up to 1-|F|^(-n/4); exact runtime qualifies",
            "bound": "Omega(min(n log|F| / log(Sw/(n^2 log|F|)), n^2 log|F|/w)) probes",
            "rectangular_direct": False,
            "systematic_only": False,
            "finite_constant_available": False,
            "applies_to_word_ram": True,
        },
        {
            "source": "CKL18",
            "theorem": "Theorem 1.2",
            "problem": "systematic Boolean n x n matrix-vector multiplication",
            "representation": "read-only matrix plus r side bits",
            "query": "v in {0,1}^n",
            "error": "randomized success at least 1-1/n; exact runtime qualifies",
            "bound": "tr=Omega(n^3) for n<=r<=n^2/4; t=Omega(n^2) for r<n",
            "rectangular_direct": False,
            "systematic_only": True,
            "finite_constant_available": False,
            "applies_to_word_ram": True,
        },
        {
            "source": "CKL18",
            "theorem": "Theorem 1.3",
            "problem": "systematic F2 vector-matrix-vector, hence F2 matrix-vector corollary",
            "representation": "read-only matrix plus r side bits",
            "query": "u,v in F2^n",
            "error": "exact/deterministic theorem statement; exact runtime qualifies",
            "bound": "tr=Omega(n^3/log n) for n<=r<=n^2/4; t=Omega(n^2/log n) for r<n",
            "rectangular_direct": False,
            "systematic_only": True,
            "finite_constant_available": False,
            "applies_to_word_ram": True,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/exp_071/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/exp_071_candidate")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True)

    started = time.perf_counter_ns()
    side_bits = int(config["hot_side_information_bytes"]) * 8
    word_bits = int(config["word_bits"])
    field_size = int(config["strongest_direct_q4_field_prime"])
    reduction = exhaustive_binary_projection_reduction(int(config["exhaustive_binary_max_dimension"]))
    controls = [
        {"control": "binary_integer_to_boolean_and_f2", "passed": reduction["mismatches"] == 0, **reduction},
        {"control": "q4_symmetric_field_embedding", "field_size": field_size, "passed": q4_field_embedding_supported(field_size)},
    ]

    tensor_rows: list[dict[str, Any]] = []
    ckl_applicable_count = 0
    specs = llama_405b_tensor_plan()
    total_q4_cells = 0
    illegal_indicator_sum = 0.0
    for spec in specs:
        dimension = spec.square_subproblem_dimension
        ckl = ckl2018_applicability(dimension, side_bits)
        cgl = cgl2015_unit_constant_indicator(
            dimension=dimension,
            field_size=field_size,
            side_bits=side_bits,
            word_bits=word_bits,
        )
        dense_cells = packed_q4_cells(spec.rows, spec.columns, word_bits)
        indicator_fraction = cgl["minimum_term_probes_unit_constant"] / dense_cells
        total_q4_cells += spec.count * dense_cells
        illegal_indicator_sum += spec.count * cgl["minimum_term_probes_unit_constant"]
        ckl_applicable_count += int(ckl.applicable)
        tensor_rows.append({
            "tensor": spec.name,
            "count": spec.count,
            "rows": spec.rows,
            "columns": spec.columns,
            "parameters": spec.parameters,
            "square_subproblem_dimension": dimension,
            "rectangular_reduction": (
                "arbitrary min(rows,columns)^2 square matrix embeds in top-left; "
                "padding to max(rows,columns) is not a reverse lower-bound reduction"
            ),
            "ckl2018": ckl.as_dict(),
            "cgl2015_unit_constant_indicator": cgl,
            "packed_q4_cells_per_instance": dense_cells,
            "unit_constant_indicator_fraction_of_rectangular_q4_cells": indicator_fraction,
            "certified_finite_probe_bound": None,
        })

    direct_sum = direct_sum_audit()
    total_parameters = parameter_total(specs)
    correctness_gate = all(row["passed"] for row in controls)
    population_gate = total_parameters == int(config["expected_parameter_total"])
    theorem_source_gate = len(theorem_hypotheses()) == 3
    per_matrix_ckl_range_gate = ckl_applicable_count == len(specs)
    direct_sum_gate = bool(direct_sum["certified"])
    finite_constant_gate = False
    strong_closure = all((
        correctness_gate,
        population_gate,
        theorem_source_gate,
        per_matrix_ckl_range_gate,
        direct_sum_gate,
        finite_constant_gate,
    ))
    decision = (
        "CERTIFY_CONVENTIONAL_EXACT_ONLINE_DENSE_RUNTIME_LOWER_BOUND"
        if strong_closure
        else "INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY"
    )
    maximum_tradeoff = max(spec.square_subproblem_dimension ** 2 // 4 for spec in specs)
    measured = {
        "theorem_count": len(theorem_hypotheses()),
        "reduction_cases": reduction["cases"],
        "float32_replay_cases": reduction["float32_replay_cases"],
        "reduction_mismatches": reduction["mismatches"],
        "control_failures": sum(not row["passed"] for row in controls),
        "tensor_family_count": len(specs),
        "tensor_instance_count": sum(spec.count for spec in specs),
        "parameter_total": total_parameters,
        "q4_parameter_gib": total_parameters * 4 / 8 / GIB,
        "hot_side_information_bits": side_bits,
        "ckl_applicable_tensor_family_count": ckl_applicable_count,
        "ckl_out_of_range_tensor_family_count": len(specs) - ckl_applicable_count,
        "maximum_square_subproblem_dimension": max(spec.square_subproblem_dimension for spec in specs),
        "maximum_ckl_tradeoff_side_bits": maximum_tradeoff,
        "side_bits_over_maximum_ckl_tradeoff_range": side_bits / maximum_tradeoff,
        "illegal_direct_sum_unit_constant_indicator_fraction": illegal_indicator_sum / total_q4_cells,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    summary = {
        "experiment": "EXP-071",
        "name": "universal_exact_dense_runtime_lower_bound_applicability_audit",
        "phase": ["A", "B-theorem-and-reduction-audit"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "decision": decision,
            "strong_closure": strong_closure,
            "correctness_gate_pass": correctness_gate,
            "population_gate_pass": population_gate,
            "theorem_source_gate_pass": theorem_source_gate,
            "per_matrix_ckl_range_gate_pass": per_matrix_ckl_range_gate,
            "direct_sum_gate_pass": direct_sum_gate,
            "finite_constant_gate_pass": finite_constant_gate,
            "boolean_reduction_valid": correctness_gate,
            "rectangular_square_subproblem_rule": "dimension=min(rows,columns)",
            "side_information_division_across_matrices": "NOT JUSTIFIED",
            "cgl_numeric_rows": "UNIT-CONSTANT INDICATORS, NOT CERTIFIED FINITE LOWER BOUNDS",
        },
        "UNVERIFIED": [
            "direct-sum lower bound for jointly preprocessed Transformer matrices",
            "finite constants hidden by asymptotic Omega and soft-O notation",
            "physical mapping from a cell probe to target GPU/PCIe/SSD transactions",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB runtime behavior",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "primary_theorem_statements": "SOURCE-AUDITED",
            "small_domain_exact_reduction": "MEASURED",
            "per_tensor_formula_inputs": "DERIVED FROM REGISTERED PLAN",
            "model_wide_lower_bound": "NOT CERTIFIED",
            "impossibility_claim": "PROHIBITED",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "config_sha256": sha256_file(args.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    dump_rows(output / "raw/theorem_hypotheses.jsonl", theorem_hypotheses())
    dump_rows(output / "raw/tensor_rows.jsonl", tensor_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "processed/direct_sum_audit.json", direct_sum)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(str(config["evidence_ceiling"]) + "\n", encoding="utf-8")
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate or not theorem_source_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
