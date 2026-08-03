#!/usr/bin/env python3
"""Run EXP-053 automatic bit-exact decision-circuit compiler Gate.

This Phase-A/B runner compiles bounded synthetic quantized linear top-1
operators from weights into structurally hashed AIG circuits. It exhaustively
validates every registered finite input domain. It does not replace a real
Transformer operation or measure target hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

from vortex_runtime.bit_circuit import (
    BinaryLinearTop1Spec,
    CompiledDecisionOperator,
    compile_binary_linear_top1,
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


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def deterministic_spec(
    *,
    family: str,
    input_bits: int,
    class_count: int,
    accumulator_bits: int,
    seed: int,
) -> BinaryLinearTop1Spec:
    randomizer = random.Random(seed)
    weights: list[tuple[int, ...]] = []

    if family == "sparse_structured":
        for class_index in range(class_count):
            row = [0] * input_bits
            row[class_index % input_bits] = 3
            if class_index + class_count < input_bits:
                row[class_index + class_count] = 1
            weights.append(tuple(row))
        biases = tuple(0 for _ in range(class_count))
    elif family == "low_rank_structured":
        base = [randomizer.choice((-2, -1, 1, 2)) for _ in range(input_bits)]
        coefficients = [
            2 * class_index - (class_count - 1)
            for class_index in range(class_count)
        ]
        for coefficient in coefficients:
            weights.append(
                tuple(
                    max(-7, min(7, coefficient * value)) for value in base
                )
            )
        biases = tuple((class_index % 3) - 1 for class_index in range(class_count))
    elif family == "dense_random":
        choices = (-3, -2, -1, 1, 2, 3)
        for _ in range(class_count):
            weights.append(
                tuple(randomizer.choice(choices) for _ in range(input_bits))
            )
        biases = tuple(
            randomizer.randint(-4, 4) for _ in range(class_count)
        )
    elif family == "late_bit":
        for class_index in range(class_count):
            row = [0] * input_bits
            if class_index == 1:
                row[-1] = 1
            weights.append(tuple(row))
        biases = tuple(
            0 if class_index < 2 else -(1 << (accumulator_bits - 2))
            for class_index in range(class_count)
        )
    else:
        raise ValueError(f"unregistered family {family}")

    specification = BinaryLinearTop1Spec(
        weights=tuple(weights),
        biases=tuple(int(value) for value in biases),
        accumulator_bits=accumulator_bits,
        family=family,
    )
    specification.validate()
    return specification


def packed_patterns(values: Any, input_bits: int, np: Any) -> tuple[int, ...]:
    patterns: list[int] = []
    for bit in range(input_bits):
        column = ((values >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
        patterns.append(
            int.from_bytes(np.packbits(column, bitorder="little").tobytes(), "little")
        )
    return tuple(patterns)


def expected_classes(
    specification: BinaryLinearTop1Spec, values: Any, np: Any
) -> Any:
    bit_indexes = np.arange(specification.input_count, dtype=np.uint64)
    activations = (
        (values[:, None] >> bit_indexes[None, :]) & np.uint64(1)
    ).astype(np.int64)
    weights = np.asarray(specification.weights, dtype=np.int64)
    biases = np.asarray(specification.biases, dtype=np.int64)
    exact_scores = activations @ weights.T + biases
    mask = (1 << specification.accumulator_bits) - 1
    sign = 1 << (specification.accumulator_bits - 1)
    unsigned = exact_scores & mask
    signed = np.where(
        (unsigned & sign) != 0,
        unsigned - (1 << specification.accumulator_bits),
        unsigned,
    )
    # np.argmax returns the first maximum, preserving lower-class tie break.
    return np.argmax(signed, axis=1).astype(np.uint64)


def exhaustive_validate(
    *,
    compiled: CompiledDecisionOperator,
    batch_size: int,
    np: Any,
) -> dict[str, Any]:
    specification = compiled.specification
    total = 1 << specification.input_count
    mismatch_count = 0
    first_counterexample: dict[str, Any] | None = None
    circuit_elapsed_ns = 0
    reference_elapsed_ns = 0

    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        values = np.arange(start, start + count, dtype=np.uint64)

        reference_start = time.perf_counter_ns()
        reference = expected_classes(specification, values, np)
        reference_elapsed_ns += time.perf_counter_ns() - reference_start

        input_patterns = packed_patterns(values, specification.input_count, np)
        circuit_start = time.perf_counter_ns()
        actual_patterns = compiled.circuit.evaluate_packed(
            input_patterns, assignment_count=count
        )
        circuit_elapsed_ns += time.perf_counter_ns() - circuit_start

        for output_bit, actual in enumerate(actual_patterns):
            expected_column = (
                (reference >> np.uint64(output_bit)) & np.uint64(1)
            ).astype(np.uint8)
            expected = int.from_bytes(
                np.packbits(expected_column, bitorder="little").tobytes(),
                "little",
            )
            difference = actual ^ expected
            if difference:
                mismatch_count += difference.bit_count()
                if first_counterexample is None:
                    local_index = (difference & -difference).bit_length() - 1
                    input_value = start + local_index
                    first_counterexample = {
                        "input_value": input_value,
                        "reference_class": specification.reference_top1(input_value),
                        "circuit_class": compiled.circuit.evaluate_scalar(input_value),
                        "output_bit": output_bit,
                    }

    return {
        "validated_assignments": total,
        "mismatch_output_bits": mismatch_count,
        "first_counterexample": first_counterexample,
        "circuit_query_elapsed_ns": circuit_elapsed_ns,
        "reference_elapsed_ns": reference_elapsed_ns,
    }


def linear_growth(input_bits: Sequence[int], nodes: Sequence[int]) -> dict[str, float]:
    if len(input_bits) != len(nodes) or len(nodes) < 2:
        raise ValueError("growth fit requires matching series")
    x_values = [float(value) for value in input_bits]
    y_values = [math.log2(float(value) + 1.0) for value in nodes]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    return {
        "log2_slope_per_input_bit": slope,
        "log2_intercept": intercept,
        "multiplicative_growth_per_added_input_bit": 2.0**slope,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "checksums.sha256"
    ):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_053/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_053_candidate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    import numpy as np

    case_shapes = [
        {**shape, "series": "input_scaling"}
        for shape in config["scaling_cases"]
    ] + [
        {**shape, "series": "class_sweep"}
        for shape in config["class_sweep_cases"]
    ]
    case_rows: list[dict[str, Any]] = []
    circuit_manifest: list[dict[str, Any]] = []
    total_mismatches = 0
    truth_table_representations = 0
    circuit_root = output / "raw" / "circuits"
    circuit_root.mkdir(parents=True, exist_ok=True)

    for family_index, family in enumerate(config["families"]):
        for shape_index, shape in enumerate(case_shapes):
            input_bits = int(shape["input_bits"])
            class_count = int(shape["class_count"])
            accumulator_bits = int(shape["accumulator_bits"])
            case_seed = (
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
                seed=case_seed,
            )
            compile_start = time.perf_counter_ns()
            compiled = compile_binary_linear_top1(specification)
            compile_elapsed_ns = time.perf_counter_ns() - compile_start
            validation = exhaustive_validate(
                compiled=compiled,
                batch_size=int(config["validation_batch_size"]),
                np=np,
            )
            total_mismatches += int(validation["mismatch_output_bits"])
            truth_table_representations += int(
                compiled.circuit.contains_truth_table
            )

            case_id = (
                f"{family}-n{input_bits}-c{class_count}-w{accumulator_bits}"
            )
            circuit_path = circuit_root / f"{case_id}.aigbin"
            circuit_bytes = compiled.circuit.to_bytes()
            circuit_path.write_bytes(circuit_bytes)
            reachable_nodes = compiled.circuit.reachable_and_node_count
            requested_nodes = compiled.circuit.requested_and_count
            logical_query_bytes = int(config["node_storage_bytes"]) * reachable_nodes
            logical_raw_bytes = int(config["node_storage_bytes"]) * requested_nodes
            logical_byte_fraction = (
                logical_query_bytes / logical_raw_bytes
                if logical_raw_bytes
                else 0.0
            )
            compile_equivalent_queries = (
                requested_nodes / max(1, reachable_nodes)
            )
            allowed_fraction = float(
                config["projection"]["required_target_equivalent_fraction"]
            )
            required_compile_reuse = (
                math.ceil(
                    requested_nodes
                    / (allowed_fraction * max(1, reachable_nodes))
                )
                if requested_nodes
                else 0
            )
            projected_bytes = (
                logical_query_bytes
                / specification.source_parameter_count
                * int(config["projection"]["target_source_parameters"])
            )
            spec_payload = json.dumps(
                {
                    "weights": specification.weights,
                    "biases": specification.biases,
                    "accumulator_bits": specification.accumulator_bits,
                    "family": specification.family,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            row = {
                "case_id": case_id,
                "series": shape["series"],
                "family": family,
                "seed": case_seed,
                "input_bits": input_bits,
                "class_count": class_count,
                "accumulator_bits": accumulator_bits,
                "finite_domain_size": 1 << input_bits,
                "source_parameter_count": specification.source_parameter_count,
                "nonzero_weight_count": specification.nonzero_weight_count,
                "specification_sha256": hashlib.sha256(spec_payload).hexdigest(),
                "representation_kind": compiled.circuit.representation_kind,
                "contains_truth_table": compiled.circuit.contains_truth_table,
                "requested_and_count": requested_nodes,
                "unique_and_node_count": compiled.circuit.and_node_count,
                "reachable_and_node_count": reachable_nodes,
                "structurally_unreachable_and_nodes": (
                    compiled.circuit.and_node_count - reachable_nodes
                ),
                "query_node_fraction": compiled.circuit.query_node_fraction,
                "logical_raw_bitblast_bytes": logical_raw_bytes,
                "logical_reachable_query_bytes": logical_query_bytes,
                "query_byte_fraction": logical_byte_fraction,
                "serialized_circuit_bytes": len(circuit_bytes),
                "serialized_circuit_sha256": hashlib.sha256(circuit_bytes).hexdigest(),
                "projected_405b_circuit_bytes": projected_bytes,
                "compile_elapsed_ns": compile_elapsed_ns,
                "compile_equivalent_queries": compile_equivalent_queries,
                "required_compile_reuse_for_1_185185_percent": required_compile_reuse,
                **validation,
            }
            case_rows.append(row)
            circuit_manifest.append(
                {
                    "case_id": case_id,
                    "path": circuit_path.relative_to(output).as_posix(),
                    "size_bytes": len(circuit_bytes),
                    "sha256": hashlib.sha256(circuit_bytes).hexdigest(),
                    "specification_sha256": row["specification_sha256"],
                }
            )

    growth_by_family: dict[str, dict[str, float]] = {}
    for family in config["families"]:
        rows = sorted(
            (
                row
                for row in case_rows
                if row["family"] == family and row["series"] == "input_scaling"
            ),
            key=lambda row: row["input_bits"],
        )
        growth_by_family[str(family)] = linear_growth(
            [int(row["input_bits"]) for row in rows],
            [int(row["reachable_and_node_count"]) for row in rows],
        )

    node_fractions = [float(row["query_node_fraction"]) for row in case_rows]
    byte_fractions = [float(row["query_byte_fraction"]) for row in case_rows]
    random_dense_fractions = [
        float(row["query_node_fraction"])
        for row in case_rows
        if row["family"] == "dense_random"
    ]
    projected_bytes = [float(row["projected_405b_circuit_bytes"]) for row in case_rows]
    compile_equivalent = [
        float(row["compile_equivalent_queries"]) for row in case_rows
    ]
    p50_node = statistics.median(node_fractions)
    p90_node = percentile(node_fractions, 0.90)
    p50_byte = statistics.median(byte_fractions)
    p90_byte = percentile(byte_fractions, 0.90)
    random_dense_p50 = statistics.median(random_dense_fractions)
    maximum_projected_bytes = max(projected_bytes)
    maximum_adversarial_growth = max(
        growth_by_family[family]["multiplicative_growth_per_added_input_bit"]
        for family in ("low_rank_structured", "dense_random")
    )
    maximum_compile_equivalent = max(compile_equivalent)

    gate = config["early_gate"]
    exact_gate_pass = total_mismatches <= int(gate["exact_mismatch_limit"])
    representation_gate_pass = truth_table_representations <= int(
        gate["truth_table_representation_limit"]
    )
    node_gate_pass = (
        p50_node <= float(gate["maximum_p50_query_node_fraction"])
        and p90_node <= float(gate["maximum_p90_query_node_fraction"])
    )
    byte_gate_pass = (
        p50_byte <= float(gate["maximum_p50_query_byte_fraction"])
        and p90_byte <= float(gate["maximum_p90_query_byte_fraction"])
    )
    storage_gate_pass = maximum_projected_bytes <= int(
        gate["maximum_projected_circuit_bytes"]
    )
    growth_gate_pass = maximum_adversarial_growth <= float(
        gate["maximum_adversarial_growth_per_added_bit"]
    )
    random_dense_gate_pass = random_dense_p50 <= float(
        gate["maximum_random_dense_p50_query_fraction"]
    )
    compile_gate_pass = maximum_compile_equivalent <= float(
        gate["maximum_compile_equivalent_queries"]
    )
    survives = all(
        (
            exact_gate_pass,
            representation_gate_pass,
            node_gate_pass,
            byte_gate_pass,
            storage_gate_pass,
            growth_gate_pass,
            random_dense_gate_pass,
            compile_gate_pass,
        )
    )
    decision = (
        "CONTINUE_BIT_EXACT_CIRCUIT_COMPILER_TO_REAL_OPERATION_REPLACEMENT"
        if survives
        else str(gate["decision_on_failure"])
    )

    measured = {
        "case_count": len(case_rows),
        "family_count": len(config["families"]),
        "total_exhaustively_validated_assignments": sum(
            int(row["validated_assignments"]) for row in case_rows
        ),
        "exact_mismatch_output_bits": total_mismatches,
        "truth_table_representations": truth_table_representations,
        "p50_query_node_fraction": p50_node,
        "p90_query_node_fraction": p90_node,
        "p50_query_byte_fraction": p50_byte,
        "p90_query_byte_fraction": p90_byte,
        "random_dense_p50_query_node_fraction": random_dense_p50,
        "maximum_projected_405b_circuit_bytes": maximum_projected_bytes,
        "maximum_projected_405b_circuit_tib": maximum_projected_bytes / 2**40,
        "maximum_adversarial_growth_per_added_input_bit": maximum_adversarial_growth,
        "maximum_compile_equivalent_queries": maximum_compile_equivalent,
        "maximum_required_compile_reuse": max(
            int(row["required_compile_reuse_for_1_185185_percent"])
            for row in case_rows
        ),
        "growth_by_family": growth_by_family,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    derived = {
        "exact_gate_pass": exact_gate_pass,
        "representation_gate_pass": representation_gate_pass,
        "query_node_gate_pass": node_gate_pass,
        "query_byte_gate_pass": byte_gate_pass,
        "storage_projection_gate_pass": storage_gate_pass,
        "adversarial_growth_gate_pass": growth_gate_pass,
        "random_dense_gate_pass": random_dense_gate_pass,
        "compile_amortization_gate_pass": compile_gate_pass,
        "bit_exact_circuit_compiler_survives_gate": survives,
        "decision": decision,
        "query_fraction_scope": (
            "relative to the same exact bit-blasted arithmetic before structural "
            "hashing; not a real Transformer or hardware byte fraction"
        ),
    }
    summary = {
        "experiment": "EXP-053",
        "name": "automatic_bit_exact_decision_circuit_compiler_gate",
        "phase": ["A", "B"],
        "evidence_level": "E1",
        "real_transformer_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "MEASURED": measured,
        "DERIVED": derived,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": 405_000_000_000
            * 4
            / 8
            / 2**30,
            "baseline_q4_full_weight_gib_per_stream": 4_000_000_000
            * 4
            / 8
            / 2**30,
            "allowed_1_2x_baseline_gib_per_token": 1.2
            * 4_000_000_000
            * 4
            / 8
            / 2**30,
            "required_target_equivalent_fraction": float(
                config["projection"]["required_target_equivalent_fraction"]
            ),
        },
        "UNVERIFIED": [
            "full quantized Transformer layer circuit compilation",
            "real small-checkpoint operation replacement",
            "physical circuit query bytes and latency",
            "70B and 405B circuit scaling",
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
            "circuit_manifest": "raw/circuit_manifest.json",
            "circuits": "raw/circuits/*.aigbin",
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
        "family_medians": {
            family: {
                "query_node_fraction": statistics.median(
                    float(row["query_node_fraction"])
                    for row in case_rows
                    if row["family"] == family
                ),
                "query_byte_fraction": statistics.median(
                    float(row["query_byte_fraction"])
                    for row in case_rows
                    if row["family"] == family
                ),
                "projected_405b_tib": statistics.median(
                    float(row["projected_405b_circuit_bytes"]) / 2**40
                    for row in case_rows
                    if row["family"] == family
                ),
            }
            for family in config["families"]
        },
        "gate": derived,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "peak_rss_kib": measured["peak_rss_kib"],
    }

    write_jsonl(output / "raw/cases.jsonl", case_rows)
    write_json(output / "raw/circuit_manifest.json", circuit_manifest)
    write_json(output / "processed/aggregate.json", aggregate)
    write_json(output / "summary.json", summary)
    write_json(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        "EXP-053 E1 bounded synthetic bit-exact AIG audit. No real Transformer "
        "operation, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/s measured.\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {
                "decision": decision,
                "case_count": len(case_rows),
                "mismatches": total_mismatches,
                "p50_node_fraction": p50_node,
                "p90_node_fraction": p90_node,
                "maximum_projected_tib": maximum_projected_bytes / 2**40,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
