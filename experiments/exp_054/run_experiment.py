#!/usr/bin/env python3
"""Run EXP-054 exact reduced ordered decision-diagram Gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

from experiments.exp_053.run_experiment import (
    deterministic_spec,
    expected_classes,
    linear_growth,
)
from vortex_runtime.decision_diagram import (
    DiagramCompilation,
    ReducedDecisionDiagram,
    compile_reduced_decision_diagram,
    natural_variable_order,
    select_smaller_completed_compilation,
    weight_magnitude_variable_order,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile_from_histogram(histogram: Counter[int], probability: float) -> float:
    if not histogram:
        raise ValueError("histogram must not be empty")
    total = sum(histogram.values())
    target = max(1, math.ceil(probability * total))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return float(value)
    raise AssertionError("unreachable percentile")


def exhaustive_validate(
    *,
    diagram: ReducedDecisionDiagram,
    specification: Any,
    batch_size: int,
    np: Any,
) -> dict[str, Any]:
    total = 1 << specification.input_count
    mismatch_count = 0
    first_counterexample: dict[str, Any] | None = None
    probe_histogram: Counter[int] = Counter()
    elapsed_ns = 0
    variables = np.asarray([node.variable for node in diagram.nodes], dtype=np.uint64)
    lows = np.asarray([node.low for node in diagram.nodes], dtype=np.uint64)
    highs = np.asarray([node.high for node in diagram.nodes], dtype=np.uint64)

    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        values = np.arange(start, start + count, dtype=np.uint64)
        expected = expected_classes(specification, values, np)
        references = np.full(count, diagram.root, dtype=np.uint64)
        probes = np.zeros(count, dtype=np.uint16)
        query_start = time.perf_counter_ns()
        for _ in range(specification.input_count + 1):
            active = references >= specification.class_count
            if not bool(np.any(active)):
                break
            node_indexes = references[active] - specification.class_count
            selected_variables = variables[node_indexes]
            bits = (values[active] >> selected_variables) & np.uint64(1)
            references[active] = np.where(
                bits != 0,
                highs[node_indexes],
                lows[node_indexes],
            )
            probes[active] += 1
        else:
            raise RuntimeError("diagram query exceeded input width")
        elapsed_ns += time.perf_counter_ns() - query_start
        difference = references != expected
        batch_mismatches = int(np.count_nonzero(difference))
        mismatch_count += batch_mismatches
        if batch_mismatches and first_counterexample is None:
            local = int(np.flatnonzero(difference)[0])
            value = start + local
            first_counterexample = {
                "input_value": value,
                "reference_class": int(expected[local]),
                "diagram_class": int(references[local]),
            }
        counts = np.bincount(probes.astype(np.int64), minlength=specification.input_count + 1)
        for probe_count, frequency in enumerate(counts.tolist()):
            if frequency:
                probe_histogram[probe_count] += int(frequency)

    return {
        "validated_assignments": total,
        "mismatches": mismatch_count,
        "first_counterexample": first_counterexample,
        "path_probe_histogram": dict(sorted(probe_histogram.items())),
        "p50_path_probes": percentile_from_histogram(probe_histogram, 0.50),
        "p90_path_probes": percentile_from_histogram(probe_histogram, 0.90),
        "maximum_path_probes": float(max(probe_histogram)),
        "p50_query_probe_fraction": percentile_from_histogram(probe_histogram, 0.50)
        / specification.input_count,
        "p90_query_probe_fraction": percentile_from_histogram(probe_histogram, 0.90)
        / specification.input_count,
        "maximum_query_probe_fraction": max(probe_histogram) / specification.input_count,
        "query_elapsed_ns": elapsed_ns,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.name != "checksums.sha256"
    ):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compilation_row(
    *,
    case_id: str,
    order_name: str,
    compilation: DiagramCompilation,
    specification: Any,
    output: Path,
    batch_size: int,
    np: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "order_name": order_name,
        "variable_order": list(compilation.variable_order),
        "ceiling": compilation.ceiling,
        "ceiling_hit": compilation.ceiling_hit,
        "compile_state_visits": compilation.compile_state_visits,
        "memoized_state_count": compilation.memoized_state_count,
        "unique_node_count": compilation.unique_node_count,
        "compile_elapsed_ns": compilation.compile_elapsed_ns,
        "fallback_required": compilation.ceiling_hit,
    }
    if compilation.diagram is None:
        row.update(
            {
                "representation_kind": None,
                "contains_truth_table": False,
                "serialized_bytes": None,
                "diagram_sha256": None,
                "validation": None,
            }
        )
        return row

    diagram = compilation.diagram
    data = diagram.to_bytes()
    path = output / "raw" / "diagrams" / f"{case_id}-{order_name}.mtddbin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    validation = exhaustive_validate(
        diagram=diagram,
        specification=specification,
        batch_size=batch_size,
        np=np,
    )
    row.update(
        {
            "representation_kind": diagram.representation_kind,
            "contains_truth_table": diagram.contains_truth_table,
            "root_reference": diagram.root,
            "node_count": diagram.node_count,
            "serialized_bytes": len(data),
            "diagram_path": path.relative_to(output).as_posix(),
            "diagram_sha256": hashlib.sha256(data).hexdigest(),
            "validation": validation,
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_054/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_054_candidate"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True)

    import numpy as np

    shapes = [
        {**shape, "series": "input_scaling"} for shape in config["scaling_cases"]
    ] + [
        {**shape, "series": "class_sweep"} for shape in config["class_sweep_cases"]
    ]
    order_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    selected_probe_histogram: Counter[int] = Counter()
    total_mismatches = 0
    truth_table_representations = 0
    ceiling_rows = 0
    completed_order_validations = 0

    for family_index, family in enumerate(config["families"]):
        for shape in shapes:
            input_bits = int(shape["input_bits"])
            class_count = int(shape["class_count"])
            accumulator_bits = int(shape["accumulator_bits"])
            seed = (
                int(config["seed"])
                + family_index * 10000
                + input_bits * 100
                + class_count * 10
                + accumulator_bits
            )
            specification = deterministic_spec(
                family=str(family),
                input_bits=input_bits,
                class_count=class_count,
                accumulator_bits=accumulator_bits,
                seed=seed,
            )
            case_id = f"{family}-n{input_bits}-c{class_count}-w{accumulator_bits}"
            orders = {
                "natural": natural_variable_order(input_bits),
                "weight_magnitude": weight_magnitude_variable_order(specification),
            }
            compilations: dict[str, DiagramCompilation] = {}
            rows_by_order: dict[str, dict[str, Any]] = {}
            for order_name, order in orders.items():
                compilation = compile_reduced_decision_diagram(
                    specification,
                    variable_order=order,
                    compile_state_ceiling=int(config["compile_state_ceiling_per_order"]),
                )
                compilations[order_name] = compilation
                row = compilation_row(
                    case_id=case_id,
                    order_name=order_name,
                    compilation=compilation,
                    specification=specification,
                    output=output,
                    batch_size=int(config["validation_batch_size"]),
                    np=np,
                )
                rows_by_order[order_name] = row
                order_rows.append(row)
                ceiling_rows += int(compilation.ceiling_hit)
                if row["validation"] is not None:
                    completed_order_validations += 1
                    total_mismatches += int(row["validation"]["mismatches"])
                    truth_table_representations += int(row["contains_truth_table"])

            selected = select_smaller_completed_compilation(tuple(compilations.values()))
            total_visits = sum(item.compile_state_visits for item in compilations.values())
            allowed = float(config["projection"]["required_target_equivalent_fraction"])
            required_order_queries = math.ceil(
                total_visits / (allowed * input_bits * class_count)
            )
            if selected is None:
                selected_order_name = None
                selected_row = None
                selected_bytes = None
                selected_nodes = None
                selected_projection = math.inf
                selected_p50_fraction = math.inf
                selected_p90_fraction = math.inf
                fallback_required = True
            else:
                selected_order_name = next(
                    name for name, item in compilations.items() if item is selected
                )
                selected_row = rows_by_order[selected_order_name]
                selected_bytes = int(selected_row["serialized_bytes"])
                selected_nodes = int(selected_row["node_count"])
                selected_projection = (
                    selected_bytes
                    / specification.source_parameter_count
                    * int(config["projection"]["target_source_parameters"])
                )
                selected_p50_fraction = float(
                    selected_row["validation"]["p50_query_probe_fraction"]
                )
                selected_p90_fraction = float(
                    selected_row["validation"]["p90_query_probe_fraction"]
                )
                fallback_required = False
                for probes, frequency in selected_row["validation"]["path_probe_histogram"].items():
                    selected_probe_histogram[int(probes)] += int(frequency)

            case_rows.append(
                {
                    "case_id": case_id,
                    "series": shape["series"],
                    "family": family,
                    "seed": seed,
                    "input_bits": input_bits,
                    "class_count": class_count,
                    "accumulator_bits": accumulator_bits,
                    "finite_domain_size": 1 << input_bits,
                    "source_parameter_count": specification.source_parameter_count,
                    "nonzero_weight_count": specification.nonzero_weight_count,
                    "selected_order": selected_order_name,
                    "selected_node_count": selected_nodes,
                    "selected_serialized_bytes": selected_bytes,
                    "selected_projected_405b_bytes": selected_projection,
                    "selected_p50_query_probe_fraction": selected_p50_fraction,
                    "selected_p90_query_probe_fraction": selected_p90_fraction,
                    "both_order_compile_state_visits": total_visits,
                    "both_order_compile_elapsed_ns": sum(
                        item.compile_elapsed_ns for item in compilations.values()
                    ),
                    "required_order_search_queries": required_order_queries,
                    "fallback_required": fallback_required,
                    "ceiling_hit_orders": [
                        name for name, item in compilations.items() if item.ceiling_hit
                    ],
                }
            )

    growth_by_family: dict[str, dict[str, float]] = {}
    for family in config["families"]:
        rows = sorted(
            (
                row for row in case_rows
                if row["family"] == family and row["series"] == "input_scaling"
                and row["selected_node_count"] is not None
            ),
            key=lambda row: row["input_bits"],
        )
        if len(rows) >= 2:
            growth_by_family[str(family)] = linear_growth(
                [int(row["input_bits"]) for row in rows],
                [int(row["selected_node_count"]) for row in rows],
            )

    p50_probes = percentile_from_histogram(selected_probe_histogram, 0.50)
    p90_probes = percentile_from_histogram(selected_probe_histogram, 0.90)
    # Weighted fractions retain each query's own input-width denominator.
    fraction_histogram: Counter[int] = Counter()
    fraction_scale = 1_000_000
    for row in case_rows:
        if row["selected_order"] is None:
            continue
        validation = next(
            order_row["validation"] for order_row in order_rows
            if order_row["case_id"] == row["case_id"]
            and order_row["order_name"] == row["selected_order"]
        )
        for probes, frequency in validation["path_probe_histogram"].items():
            scaled = round(int(probes) / int(row["input_bits"]) * fraction_scale)
            fraction_histogram[scaled] += int(frequency)
    p50_fraction = percentile_from_histogram(fraction_histogram, 0.50) / fraction_scale
    p90_fraction = percentile_from_histogram(fraction_histogram, 0.90) / fraction_scale

    projected_values = [
        float(row["selected_projected_405b_bytes"])
        for row in case_rows if math.isfinite(float(row["selected_projected_405b_bytes"]))
    ]
    maximum_projection = max(projected_values, default=math.inf)
    maximum_growth = max(
        (
            value["multiplicative_growth_per_added_input_bit"]
            for family, value in growth_by_family.items()
            if family in {"low_rank_structured", "dense_random"}
        ),
        default=math.inf,
    )
    maximum_required_queries = max(
        int(row["required_order_search_queries"]) for row in case_rows
    )
    fallback_cases = sum(int(row["fallback_required"]) for row in case_rows)
    fallback_rate = fallback_cases / len(case_rows)

    gate = config["early_gate"]
    exact_gate_pass = total_mismatches <= int(gate["exact_mismatch_limit"])
    representation_gate_pass = truth_table_representations <= int(
        gate["truth_table_representation_limit"]
    )
    probe_gate_pass = (
        p50_fraction <= float(gate["maximum_p50_query_probe_fraction"])
        and p90_fraction <= float(gate["maximum_p90_query_probe_fraction"])
    )
    ceiling_gate_pass = fallback_rate <= float(gate["maximum_ceiling_or_fallback_rate"])
    storage_gate_pass = maximum_projection <= int(gate["maximum_projected_diagram_bytes"])
    growth_gate_pass = maximum_growth <= float(
        gate["maximum_adversarial_growth_per_added_bit"]
    )
    order_gate_pass = maximum_required_queries <= int(
        gate["maximum_required_order_search_queries"]
    )
    survives = all(
        (
            exact_gate_pass,
            representation_gate_pass,
            probe_gate_pass,
            ceiling_gate_pass,
            storage_gate_pass,
            growth_gate_pass,
            order_gate_pass,
        )
    )
    decision = (
        "CONTINUE_REDUCED_DECISION_DIAGRAM_TO_REAL_OPERATION_REPLACEMENT"
        if survives else str(gate["decision_on_failure"])
    )

    measured = {
        "case_count": len(case_rows),
        "order_row_count": len(order_rows),
        "completed_order_validations": completed_order_validations,
        "total_exhaustively_validated_assignments": sum(
            int(row["validation"]["validated_assignments"])
            for row in order_rows if row["validation"] is not None
        ),
        "exact_mismatches": total_mismatches,
        "truth_table_representations": truth_table_representations,
        "ceiling_hit_order_rows": ceiling_rows,
        "fallback_case_count": fallback_cases,
        "fallback_case_rate": fallback_rate,
        "selected_global_p50_path_probes": p50_probes,
        "selected_global_p90_path_probes": p90_probes,
        "selected_global_p50_query_probe_fraction": p50_fraction,
        "selected_global_p90_query_probe_fraction": p90_fraction,
        "maximum_projected_405b_diagram_bytes": maximum_projection,
        "maximum_projected_405b_diagram_tib": maximum_projection / 2**40,
        "maximum_adversarial_growth_per_added_input_bit": maximum_growth,
        "maximum_required_order_search_queries": maximum_required_queries,
        "growth_by_family": growth_by_family,
        "selected_order_distribution": dict(
            Counter(str(row["selected_order"]) for row in case_rows)
        ),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    derived = {
        "exact_gate_pass": exact_gate_pass,
        "representation_gate_pass": representation_gate_pass,
        "query_probe_gate_pass": probe_gate_pass,
        "ceiling_fallback_gate_pass": ceiling_gate_pass,
        "storage_projection_gate_pass": storage_gate_pass,
        "adversarial_growth_gate_pass": growth_gate_pass,
        "order_search_amortization_gate_pass": order_gate_pass,
        "reduced_decision_diagram_survives_gate": survives,
        "decision": decision,
        "query_fraction_scope": (
            "root-to-terminal input probes divided by declared binary activation width; "
            "not a real Transformer parameter-byte or hardware latency fraction"
        ),
    }
    summary = {
        "experiment": "EXP-054",
        "name": "exact_reduced_ordered_decision_diagram_gate",
        "phase": ["A", "B"],
        "evidence_level": "E1",
        "real_transformer_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "MEASURED": measured,
        "DERIVED": derived,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": 405_000_000_000 * 4 / 8 / 2**30,
            "baseline_q4_full_weight_gib_per_stream": 4_000_000_000 * 4 / 8 / 2**30,
            "allowed_1_2x_baseline_gib_per_token": 1.2 * 4_000_000_000 * 4 / 8 / 2**30,
            "required_target_equivalent_fraction": float(
                config["projection"]["required_target_equivalent_fraction"]
            ),
        },
        "UNVERIFIED": [
            "real quantized Transformer decision-diagram compilation",
            "real small-checkpoint operation replacement",
            "physical diagram lookup and storage traffic",
            "70B and 405B diagram scaling",
            "8 GiB total hot state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "authoritative_decision": decision,
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(args.config),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "raw_evidence": {
            "cases": "raw/cases.jsonl",
            "order_rows": "raw/order_rows.jsonl",
            "diagrams": "raw/diagrams/*.mtddbin",
        },
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "real_transformer_operation_replacement": False,
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
    }
    aggregate = {
        "growth_by_family": growth_by_family,
        "selected_order_distribution": measured["selected_order_distribution"],
        "global_probe_histogram": dict(sorted(selected_probe_histogram.items())),
        "gate": derived,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "peak_rss_kib": measured["peak_rss_kib"],
    }

    write_jsonl(output / "raw/cases.jsonl", case_rows)
    write_jsonl(output / "raw/order_rows.jsonl", order_rows)
    write_json(output / "processed/aggregate.json", aggregate)
    write_json(output / "summary.json", summary)
    write_json(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        "EXP-054 E1 bounded synthetic reduced decision-diagram audit. No real "
        "Transformer operation, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/s measured.\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {
                "decision": decision,
                "case_count": len(case_rows),
                "mismatches": total_mismatches,
                "p50_probe_fraction": p50_fraction,
                "p90_probe_fraction": p90_fraction,
                "maximum_projected_tib": maximum_projection / 2**40,
                "fallback_rate": fallback_rate,
            },
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
