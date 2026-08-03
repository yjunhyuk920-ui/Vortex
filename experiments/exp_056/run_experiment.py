#!/usr/bin/env python3
"""Run EXP-056 exact prototype plus sparse-residual dictionary Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
from typing import Any

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec
from vortex_runtime.prototype_residual import (
    PrototypeResidualPlan,
    compile_prototype_residual_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def wrap(value: int, width: int) -> int:
    unsigned = value & ((1 << width) - 1)
    sign = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned & sign else unsigned


def random_column(
    rng: random.Random, classes: int, limit: int
) -> tuple[int, ...]:
    bound = min(limit, 15)
    while True:
        result = tuple(rng.randint(-bound, bound) for _ in range(classes))
        if any(result):
            return result


def make_columns(
    family: str,
    n: int,
    classes: int,
    width: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    limit = (1 << (width - 1)) - 1
    if family == "repeated_columns":
        pool = tuple(random_column(rng, classes, limit) for _ in range(min(2, n)))
        return tuple(pool[index % len(pool)] for index in range(n))
    if family == "prototype_sparse_residual":
        pool = tuple(random_column(rng, classes, limit) for _ in range(min(2, n)))
        result = [pool[index % len(pool)] for index in range(n)]
        mutation_count = max(1, n // 16)
        for index in rng.sample(range(n), mutation_count):
            values = list(result[index])
            class_index = rng.randrange(classes)
            delta = 1 if values[class_index] < limit else -1
            values[class_index] += delta
            result[index] = tuple(values)
        return tuple(result)
    if family == "sign_clusters":
        pool = tuple(random_column(rng, classes, limit) for _ in range(min(2, n)))
        result = []
        for index in range(n):
            base = pool[index % len(pool)]
            result.append(base if (index // len(pool)) % 2 == 0 else tuple(-v for v in base))
        return tuple(result)
    if family == "sparse_columns":
        result = [(0,) * classes for _ in range(n)]
        for index in rng.sample(range(n), max(1, n // 5)):
            result[index] = random_column(rng, classes, limit)
        return tuple(result)
    if family == "low_rank_columns":
        left = random_column(rng, classes, max(1, limit // 6))
        right = random_column(rng, classes, max(1, limit // 6))
        coefficients = (
            (-3, -2), (-3, 1), (-2, 3), (-1, -3), (-1, 2),
            (1, -2), (1, 3), (2, -3), (2, 1), (3, -1), (3, 2),
        )
        result = []
        for index in range(n):
            a, b = coefficients[index % len(coefficients)]
            result.append(
                tuple(
                    max(-limit, min(limit, a * x + b * y))
                    for x, y in zip(left, right)
                )
            )
        return tuple(result)
    if family == "dense_random":
        return tuple(random_column(rng, classes, limit) for _ in range(n))
    if family == "forced_unique":
        observed: set[tuple[int, ...]] = set()
        result: list[tuple[int, ...]] = []
        while len(result) < n:
            candidate = random_column(rng, classes, limit)
            if candidate not in observed:
                observed.add(candidate)
                result.append(candidate)
        return tuple(result)
    raise ValueError(f"unknown family {family}")


def make_specification(
    family: str,
    n: int,
    classes: int,
    width: int,
    seed: int,
) -> BinaryLinearTop1Spec:
    columns = make_columns(family, n, classes, width, seed)
    rng = random.Random(seed ^ 0x56AA56AA)
    bias_limit = min(7, (1 << (width - 1)) - 1)
    specification = BinaryLinearTop1Spec(
        weights=tuple(
            tuple(columns[index][class_index] for index in range(n))
            for class_index in range(classes)
        ),
        biases=tuple(rng.randint(-bias_limit, bias_limit) for _ in range(classes)),
        accumulator_bits=width,
        family=family,
    )
    specification.validate()
    return specification


def source_scores(specification: BinaryLinearTop1Spec, input_value: int) -> tuple[int, ...]:
    return tuple(
        wrap(
            bias
            + sum(
                weight
                for index, weight in enumerate(row)
                if (input_value >> index) & 1
            ),
            specification.accumulator_bits,
        )
        for row, bias in zip(specification.weights, specification.biases)
    )


def prepared(plan: PrototypeResidualPlan) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    plan.validate()
    groups = tuple(
        (group.prototype, group.mask())
        for group in plan.active_prototype_groups
    )
    residuals = tuple(
        (
            residual.input_index,
            tuple((entry.class_index, entry.value) for entry in residual.entries),
        )
        for residual in plan.residual_columns
    )
    return groups, residuals


def plan_scores(
    specification: BinaryLinearTop1Spec,
    prepared_groups: tuple[Any, ...],
    prepared_residuals: tuple[Any, ...],
    input_value: int,
) -> tuple[int, ...]:
    scores = list(specification.biases)
    for prototype, mask in prepared_groups:
        count = (input_value & mask).bit_count()
        for class_index, weight in enumerate(prototype):
            if weight:
                scores[class_index] += count * weight
    for input_index, entries in prepared_residuals:
        active = (input_value >> input_index) & 1
        if active:
            for class_index, value in entries:
                scores[class_index] += value
    return tuple(wrap(value, specification.accumulator_bits) for value in scores)


def values_for(
    n: int, exhaustive_max: int, samples: int, seed: int
) -> tuple[tuple[int, ...], bool]:
    domain = 1 << n
    if n <= exhaustive_max:
        return tuple(range(domain)), True
    rng = random.Random(seed ^ 0xA656A656)
    values = {0, 1, domain - 2, domain - 1}
    while len(values) < samples:
        values.add(rng.randrange(domain))
    return tuple(sorted(values)), False


def validate_plan(
    specification: BinaryLinearTop1Spec,
    plan: PrototypeResidualPlan,
    inputs: tuple[int, ...],
) -> dict[str, Any]:
    groups, residuals = prepared(plan)
    score_mismatches = 0
    top1_mismatches = 0
    first_counterexample = None
    started = time.perf_counter_ns()
    for input_value in inputs:
        expected = source_scores(specification, input_value)
        actual = plan_scores(specification, groups, residuals, input_value)
        expected_class = max(range(len(expected)), key=lambda index: expected[index])
        actual_class = max(range(len(actual)), key=lambda index: actual[index])
        score_mismatches += int(expected != actual)
        top1_mismatches += int(expected_class != actual_class)
        if first_counterexample is None and (expected != actual or expected_class != actual_class):
            first_counterexample = {
                "input": input_value,
                "expected_scores": expected,
                "actual_scores": actual,
                "expected_class": expected_class,
                "actual_class": actual_class,
            }
    return {
        "validated_assignments": len(inputs),
        "score_mismatches": score_mismatches,
        "top1_mismatches": top1_mismatches,
        "first_counterexample": first_counterexample,
        "elapsed_ns": time.perf_counter_ns() - started,
    }


def validate_packed(
    specification: BinaryLinearTop1Spec,
    plan: PrototypeResidualPlan,
) -> int:
    assignment_count = 1 << specification.input_count
    patterns = tuple(
        sum(
            ((value >> bit) & 1) << assignment
            for assignment, value in enumerate(range(assignment_count))
        )
        for bit in range(specification.input_count)
    )
    outputs = plan.evaluate_packed(patterns, assignment_count=assignment_count)
    mismatches = 0
    for value in range(assignment_count):
        actual = sum(
            ((outputs[bit] >> value) & 1) << bit
            for bit in range(len(outputs))
        )
        scores = source_scores(specification, value)
        expected = max(range(len(scores)), key=lambda index: scores[index])
        mismatches += int(actual != expected)
    return mismatches


def accounting(plan: PrototypeResidualPlan) -> dict[str, Any]:
    specification = plan.specification
    scalar_bytes = math.ceil(specification.accumulator_bits / 8)
    input_index_bytes = max(1, math.ceil(math.log2(max(2, specification.input_count)) / 8))
    class_index_bytes = max(1, math.ceil(math.log2(max(2, specification.class_count)) / 8))
    active_groups = plan.active_prototype_groups
    membership_words = plan.membership_word_count
    prototype_scalar_slots = len(active_groups) * specification.class_count
    baseline_operations = specification.class_count * specification.input_count
    grouped_operations = (
        2 * membership_words
        + 2 * plan.prototype_scalar_count
        + plan.residual_column_count
        + plan.residual_scalar_count
    )
    baseline_bytes = (
        specification.class_count * specification.input_count * scalar_bytes
        + 8 * math.ceil(specification.input_count / 64)
    )
    prototype_bytes = prototype_scalar_slots * scalar_bytes
    membership_bytes = membership_words * 8
    residual_bytes = (
        plan.residual_column_count * (input_index_bytes + 1)
        + plan.residual_scalar_count * (class_index_bytes + scalar_bytes)
    )
    grouped_bytes = (
        specification.class_count * scalar_bytes
        + prototype_bytes
        + membership_bytes
        + residual_bytes
    )
    logical_storage = 32 + grouped_bytes
    source_weight_bytes = specification.class_count * specification.input_count * scalar_bytes
    return {
        "prototype_count": plan.prototype_count,
        "active_prototype_group_count": len(active_groups),
        "prototype_scalar_count": plan.prototype_scalar_count,
        "prototype_scalar_slots": prototype_scalar_slots,
        "membership_word_count": membership_words,
        "residual_column_count": plan.residual_column_count,
        "residual_scalar_count": plan.residual_scalar_count,
        "residual_scalar_fraction": plan.residual_scalar_count / specification.source_parameter_count,
        "baseline_operation_count": baseline_operations,
        "grouped_operation_count": grouped_operations,
        "operation_fraction": grouped_operations / baseline_operations,
        "baseline_query_bytes": baseline_bytes,
        "grouped_query_bytes": grouped_bytes,
        "query_byte_fraction": grouped_bytes / baseline_bytes,
        "logical_storage_bytes": logical_storage,
        "storage_fraction": logical_storage / source_weight_bytes,
        "reference_serialized_bytes": len(plan.to_bytes()),
        "compile_operation_count": plan.compile_operation_count,
        "contains_truth_table": plan.contains_truth_table,
    }


def finite_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_056/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_056_candidate",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    output = arguments.output_dir
    if output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    shapes = [
        {**shape, "series": "input_scaling"}
        for shape in config["scaling_cases"]
    ] + [
        {**shape, "series": "class_sweep"}
        for shape in config["class_sweep_cases"]
    ]
    case_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    totals = {
        "score": 0,
        "top1": 0,
        "packed": 0,
        "validated": 0,
        "exhaustive": 0,
        "sampled": 0,
        "truth_tables": 0,
    }
    experiment_started = time.perf_counter_ns()

    for family_index, family in enumerate(config["families"]):
        for shape_index, shape in enumerate(shapes):
            n = int(shape["input_bits"])
            classes = int(shape["class_count"])
            width = int(shape["accumulator_bits"])
            seed = (
                int(config["seed"])
                + family_index * 100000
                + shape_index * 1000
                + n * 100
                + classes * 10
                + width
            )
            specification = make_specification(family, n, classes, width, seed)
            case_id = f"{family}-n{n}-c{classes}-w{width}"
            inputs, exhaustive = values_for(
                n,
                int(config["exhaustive_max_input_bits"]),
                int(config["larger_domain_samples"]),
                seed,
            )
            variants: list[tuple[str, int, PrototypeResidualPlan, dict[str, Any]]] = []
            total_compile_operations = 0
            for strategy in config["strategies"]:
                for requested_count in config["prototype_counts"]:
                    started = time.perf_counter_ns()
                    plan = compile_prototype_residual_plan(
                        specification,
                        requested_prototype_count=int(requested_count),
                        strategy=strategy,
                    )
                    compile_elapsed_ns = time.perf_counter_ns() - started
                    validation = validate_plan(specification, plan, inputs)
                    info = accounting(plan)
                    data = plan.to_bytes()
                    plan_path = (
                        output
                        / "raw/plans"
                        / f"{case_id}-{strategy}-k{requested_count}.prbin"
                    )
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_path.write_bytes(data)
                    row = {
                        "case_id": case_id,
                        "strategy": strategy,
                        "requested_prototype_count": requested_count,
                        "compile_elapsed_ns": compile_elapsed_ns,
                        "plan_sha256": hashlib.sha256(data).hexdigest(),
                        "validation": validation,
                        **info,
                    }
                    plan_rows.append(row)
                    variants.append((strategy, int(requested_count), plan, info))
                    total_compile_operations += plan.compile_operation_count
                    totals["score"] += validation["score_mismatches"]
                    totals["top1"] += validation["top1_mismatches"]
                    totals["validated"] += validation["validated_assignments"]
                    totals["truth_tables"] += int(info["contains_truth_table"])
                    totals["exhaustive" if exhaustive else "sampled"] += validation[
                        "validated_assignments"
                    ]

            strategy, requested_count, selected, selected_info = min(
                variants,
                key=lambda item: (
                    item[3]["operation_fraction"],
                    item[3]["query_byte_fraction"],
                    item[3]["logical_storage_bytes"],
                    item[0],
                    item[1],
                ),
            )
            packed_mismatches = 0
            if n <= int(config["packed_validation_max_input_bits"]):
                packed_mismatches = validate_packed(specification, selected)
                totals["packed"] += packed_mismatches
            saved_operations = (
                selected_info["baseline_operation_count"]
                - selected_info["grouped_operation_count"]
            )
            amortization = (
                math.ceil(total_compile_operations / saved_operations)
                if saved_operations > 0
                else math.inf
            )
            projection = selected_info["storage_fraction"] * float(
                config["projection"]["target_q4_bytes"]
            )
            case_rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "series": shape["series"],
                    "seed": seed,
                    "input_bits": n,
                    "class_count": classes,
                    "accumulator_bits": width,
                    "validation_mode": "exhaustive" if exhaustive else "deterministic_sample",
                    "validated_assignments_per_plan": len(inputs),
                    "selected_strategy": strategy,
                    "selected_requested_prototype_count": requested_count,
                    "all_variant_compile_operations": total_compile_operations,
                    "required_compile_amortization_queries": amortization,
                    "packed_mismatches": packed_mismatches,
                    "selected_projected_405b_q4_storage_bytes": projection,
                    **{f"selected_{key}": value for key, value in selected_info.items()},
                }
            )

    operations = [float(row["selected_operation_fraction"]) for row in case_rows]
    bytes_fractions = [float(row["selected_query_byte_fraction"]) for row in case_rows]
    dense_unique = [
        float(row["selected_operation_fraction"])
        for row in case_rows
        if row["family"] in {"dense_random", "forced_unique"}
    ]
    projections = [
        float(row["selected_projected_405b_q4_storage_bytes"])
        for row in case_rows
    ]
    amortizations = [float(row["required_compile_amortization_queries"]) for row in case_rows]
    structured_families = (
        "repeated_columns",
        "prototype_sparse_residual",
        "sign_clusters",
    )
    structured: dict[str, list[float]] = {}
    for family in structured_families:
        rows = sorted(
            (
                row
                for row in case_rows
                if row["family"] == family and row["series"] == "input_scaling"
            ),
            key=lambda row: row["input_bits"],
        )
        structured[family] = [float(row["selected_operation_fraction"]) for row in rows]
    structured_non_degrading = all(
        all(current <= previous + 1e-12 for previous, current in zip(values, values[1:]))
        for values in structured.values()
    )

    p50_operations = percentile(operations, 0.5)
    p90_operations = percentile(operations, 0.9)
    p50_bytes = percentile(bytes_fractions, 0.5)
    p90_bytes = percentile(bytes_fractions, 0.9)
    dense_unique_p50 = percentile(dense_unique, 0.5)
    maximum_projection = max(projections)
    maximum_amortization = finite_max(amortizations)
    gates = config["early_gate"]
    checks = {
        "exact_gate_pass": (
            totals["score"] + totals["top1"] + totals["packed"]
            <= gates["maximum_exact_mismatches"]
        ),
        "operation_gate_pass": (
            p50_operations <= gates["maximum_p50_operation_fraction"]
            and p90_operations <= gates["maximum_p90_operation_fraction"]
        ),
        "byte_gate_pass": (
            p50_bytes <= gates["maximum_p50_byte_fraction"]
            and p90_bytes <= gates["maximum_p90_byte_fraction"]
        ),
        "dense_unique_gate_pass": (
            dense_unique_p50 <= gates["maximum_dense_unique_p50_fraction"]
        ),
        "storage_gate_pass": maximum_projection <= gates["maximum_projected_storage_bytes"],
        "compile_amortization_gate_pass": (
            maximum_amortization <= gates["maximum_compile_amortization_queries"]
        ),
        "structured_non_degrading_gate_pass": structured_non_degrading,
        "representation_gate_pass": totals["truth_tables"] == 0,
    }
    survives = all(checks.values())
    decision = (
        "PROMOTE_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_TO_REAL_WEIGHT_EXTRACTION"
        if survives
        else "REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY"
    )
    selected_strategy_distribution: dict[str, int] = {}
    selected_count_distribution: dict[str, int] = {}
    for row in case_rows:
        strategy = str(row["selected_strategy"])
        count = str(row["selected_requested_prototype_count"])
        selected_strategy_distribution[strategy] = selected_strategy_distribution.get(strategy, 0) + 1
        selected_count_distribution[count] = selected_count_distribution.get(count, 0) + 1

    summary = {
        "experiment": "EXP-056",
        "name": "exact_prototype_plus_sparse_residual_dictionary_gate",
        "evidence_level": "E1",
        "phase": ["A", "B"],
        "MEASURED": {
            "case_count": len(case_rows),
            "plan_count": len(plan_rows),
            "family_count": len(config["families"]),
            "total_scalar_validations": totals["validated"],
            "exhaustive_scalar_validations": totals["exhaustive"],
            "sampled_scalar_validations": totals["sampled"],
            "score_mismatches": totals["score"],
            "top1_mismatches": totals["top1"],
            "packed_mismatches": totals["packed"],
            "truth_table_representations": totals["truth_tables"],
            "p50_operation_fraction": p50_operations,
            "p90_operation_fraction": p90_operations,
            "p50_query_byte_fraction": p50_bytes,
            "p90_query_byte_fraction": p90_bytes,
            "dense_unique_p50_operation_fraction": dense_unique_p50,
            "maximum_projected_405b_q4_storage_bytes": maximum_projection,
            "maximum_projected_405b_q4_storage_tib": maximum_projection / 1024**4,
            "maximum_required_compile_amortization_queries": maximum_amortization,
            "infinite_amortization_case_count": sum(math.isinf(value) for value in amortizations),
            "structured_scaling_operation_fractions": structured,
            "selected_strategy_distribution": selected_strategy_distribution,
            "selected_requested_prototype_count_distribution": selected_count_distribution,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - experiment_started,
        },
        "DERIVED": {
            **checks,
            "prototype_residual_dictionary_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "logical membership-mask AND/popcount, nonzero prototype multiply/add, "
                "residual-column activation checks and exact residual scalar adds versus "
                "C*n conditional adds; not hardware latency"
            ),
        },
        "PROJECTED": config["projection"],
        "UNVERIFIED": [
            "real checkpoint prototype/residual sparsity",
            "real Transformer operation replacement",
            "physical dictionary kernel bytes and latency",
            "70B and 405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "authoritative_decision": decision,
        "real_transformer_operation_replacement": False,
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
            "real_transformer_operation_replacement": False,
        },
        "provenance": {
            "source_commit": commit(),
            "config_sha256": digest(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    dump_rows(output / "raw/plan_rows.jsonl", plan_rows)
    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        "EXP-056 Phase A/B E1 synthetic exact dictionary Gate.\n"
        "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/sec: NOT TESTED.\n"
    )
    checksum_lines = [
        f"{digest(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
