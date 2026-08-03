#!/usr/bin/env python3
"""Run EXP-055 exact column-signature popcount aggregation Gate."""

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
from vortex_runtime.column_signature import (
    ColumnSignaturePlan,
    compile_column_signature_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def wrap(value: int, width: int) -> int:
    unsigned = value & ((1 << width) - 1)
    sign = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned & sign else unsigned


def signature(rng: random.Random, classes: int, limit: int) -> tuple[int, ...]:
    bound = min(limit, 15)
    while True:
        value = tuple(rng.randint(-bound, bound) for _ in range(classes))
        if any(value):
            return value


def columns(
    family: str, n: int, classes: int, width: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    limit = (1 << (width - 1)) - 1
    if family == "repeated_columns":
        pool = tuple(signature(rng, classes, limit) for _ in range(min(2, n)))
        return tuple(pool[index % len(pool)] for index in range(n))
    if family == "sign_related_columns":
        pool = tuple(signature(rng, classes, limit) for _ in range(min(2, n)))
        return tuple(
            pool[index % len(pool)]
            if (index // len(pool)) % 2 == 0
            else tuple(-item for item in pool[index % len(pool)])
            for index in range(n)
        )
    if family == "sparse_columns":
        result = [(0,) * classes for _ in range(n)]
        for index in rng.sample(range(n), max(1, n // 5)):
            result[index] = signature(rng, classes, limit)
        return tuple(result)
    if family == "low_rank_columns":
        left = signature(rng, classes, max(1, limit // 4))
        right = signature(rng, classes, max(1, limit // 4))
        coefficients = (
            (-3, -2), (-3, 1), (-2, 3), (-1, -3), (-1, 2),
            (1, -2), (1, 3), (2, -3), (2, 1), (3, -1), (3, 2),
        )
        return tuple(
            tuple(
                max(-limit, min(limit, a * x + b * y))
                for x, y in zip(left, right)
            )
            for a, b in (
                coefficients[index % len(coefficients)] for index in range(n)
            )
        )
    if family == "dense_random":
        return tuple(signature(rng, classes, limit) for _ in range(n))
    if family == "forced_unique":
        result = tuple(
            tuple(index + 1 + cls * (n + 7) for cls in range(classes))
            for index in range(n)
        )
        if max(max(item) for item in result) > limit:
            raise RuntimeError("registered forced-unique case exceeds signed width")
        return result
    raise ValueError(f"unknown family {family}")


def specification(
    family: str, n: int, classes: int, width: int, seed: int
) -> BinaryLinearTop1Spec:
    source = columns(family, n, classes, width, seed)
    rng = random.Random(seed ^ 0x55AA55AA)
    bias_limit = min(7, (1 << (width - 1)) - 1)
    spec = BinaryLinearTop1Spec(
        weights=tuple(
            tuple(source[index][cls] for index in range(n))
            for cls in range(classes)
        ),
        biases=tuple(rng.randint(-bias_limit, bias_limit) for _ in range(classes)),
        accumulator_bits=width,
        family=family,
    )
    spec.validate()
    return spec


def active(plan: ColumnSignaturePlan) -> tuple[Any, ...]:
    return tuple(group for group in plan.groups if any(group.signature))


def accounting(plan: ColumnSignaturePlan) -> dict[str, Any]:
    spec = plan.specification
    scalar_bytes = math.ceil(spec.accumulator_bits / 8)
    groups = active(plan)
    words = sum(group.active_word_count() for group in groups)
    baseline_ops = spec.class_count * spec.input_count
    grouped_ops = 2 * words + 2 * spec.class_count * len(groups)
    baseline_bytes = (
        spec.source_parameter_count * scalar_bytes
        + 8 * math.ceil(spec.input_count / 64)
    )
    grouped_bytes = (
        spec.class_count * scalar_bytes
        + len(groups) * spec.class_count * scalar_bytes
        + 8 * words
    )
    storage = 32 + grouped_bytes
    return {
        "active_group_count": len(groups),
        "stored_group_count": plan.group_count,
        "membership_word_count": words,
        "baseline_operation_count": baseline_ops,
        "grouped_operation_count": grouped_ops,
        "operation_fraction": grouped_ops / baseline_ops,
        "baseline_query_bytes": baseline_bytes,
        "grouped_query_bytes": grouped_bytes,
        "query_byte_fraction": grouped_bytes / baseline_bytes,
        "logical_storage_bytes": storage,
        "reference_serialized_bytes": plan.serialized_bytes,
        "contains_truth_table": plan.contains_truth_table,
    }


def source_scores(spec: BinaryLinearTop1Spec, value: int) -> tuple[int, ...]:
    return tuple(
        wrap(
            bias
            + sum(
                weight
                for bit, weight in enumerate(row)
                if (value >> bit) & 1
            ),
            spec.accumulator_bits,
        )
        for row, bias in zip(spec.weights, spec.biases)
    )


def prepared(plan: ColumnSignaturePlan) -> tuple[Any, ...]:
    plan.validate()
    return tuple(
        (group.signature, group.positive_mask(), group.negative_mask())
        for group in active(plan)
    )


def plan_scores(
    spec: BinaryLinearTop1Spec, groups: tuple[Any, ...], value: int
) -> tuple[int, ...]:
    scores = list(spec.biases)
    for sig, positive, negative in groups:
        count = (value & positive).bit_count() - (value & negative).bit_count()
        for cls, weight in enumerate(sig):
            scores[cls] += count * weight
    return tuple(wrap(item, spec.accumulator_bits) for item in scores)


def values_for(
    n: int, exhaustive_max: int, samples: int, seed: int
) -> tuple[tuple[int, ...], bool]:
    domain = 1 << n
    if n <= exhaustive_max:
        return tuple(range(domain)), True
    rng = random.Random(seed ^ 0xA5A5A5A5)
    values = {0, 1, domain - 2, domain - 1}
    while len(values) < samples:
        values.add(rng.randrange(domain))
    return tuple(sorted(values)), False


def validate(
    spec: BinaryLinearTop1Spec,
    plan: ColumnSignaturePlan,
    values: tuple[int, ...],
) -> dict[str, Any]:
    groups = prepared(plan)
    score_mismatches = 0
    top1_mismatches = 0
    first = None
    started = time.perf_counter_ns()
    for value in values:
        expected = source_scores(spec, value)
        actual = plan_scores(spec, groups, value)
        expected_class = max(
            range(len(expected)), key=lambda index: expected[index]
        )
        actual_class = max(
            range(len(actual)), key=lambda index: actual[index]
        )
        score_mismatches += int(expected != actual)
        top1_mismatches += int(expected_class != actual_class)
        if first is None and (
            expected != actual or expected_class != actual_class
        ):
            first = {
                "input": value,
                "expected_scores": expected,
                "actual_scores": actual,
                "expected_class": expected_class,
                "actual_class": actual_class,
            }
    return {
        "validated_assignments": len(values),
        "score_mismatches": score_mismatches,
        "top1_mismatches": top1_mismatches,
        "first_counterexample": first,
        "elapsed_ns": time.perf_counter_ns() - started,
    }


def validate_packed(spec: BinaryLinearTop1Spec, plan: ColumnSignaturePlan) -> int:
    count = 1 << spec.input_count
    patterns = tuple(
        sum(((value >> bit) & 1) << value for value in range(count))
        for bit in range(spec.input_count)
    )
    outputs = plan.evaluate_packed(patterns, assignment_count=count)
    mismatches = 0
    for value in range(count):
        actual = sum(
            ((outputs[bit] >> value) & 1) << bit
            for bit in range(len(outputs))
        )
        scores = source_scores(spec, value)
        expected = max(range(len(scores)), key=lambda index: scores[index])
        mismatches += int(actual != expected)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_055/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_055_candidate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir
    if output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    shapes = [
        {**item, "series": "input_scaling"}
        for item in config["scaling_cases"]
    ] + [
        {**item, "series": "class_sweep"}
        for item in config["class_sweep_cases"]
    ]
    cases: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
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
            spec = specification(family, n, classes, width, seed)
            case_id = f"{family}-n{n}-c{classes}-w{width}"
            inputs, exhaustive = values_for(
                n,
                int(config["exhaustive_max_input_bits"]),
                int(config["larger_domain_samples"]),
                seed,
            )
            variants: list[tuple[str, ColumnSignaturePlan, int]] = []
            for mode, canonical in (
                ("identical_only", False),
                ("sign_canonical", True),
            ):
                started = time.perf_counter_ns()
                plan = compile_column_signature_plan(
                    spec, sign_canonical=canonical
                )
                compile_ns = time.perf_counter_ns() - started
                measured = validate(spec, plan, inputs)
                info = accounting(plan)
                data = plan.to_bytes()
                plan_path = (
                    output
                    / "raw"
                    / "plans"
                    / f"{case_id}-{mode}.colbin"
                )
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_bytes(data)
                plans.append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "compile_elapsed_ns": compile_ns,
                        "plan_sha256": hashlib.sha256(data).hexdigest(),
                        "validation": measured,
                        **info,
                    }
                )
                variants.append((mode, plan, compile_ns))
                totals["score"] += measured["score_mismatches"]
                totals["top1"] += measured["top1_mismatches"]
                totals["validated"] += measured["validated_assignments"]
                totals["truth_tables"] += int(info["contains_truth_table"])
                totals[
                    "exhaustive" if exhaustive else "sampled"
                ] += measured["validated_assignments"]

            mode, selected, _ = min(
                variants,
                key=lambda item: (
                    accounting(item[1])["operation_fraction"],
                    accounting(item[1])["query_byte_fraction"],
                    accounting(item[1])["logical_storage_bytes"],
                    item[0],
                ),
            )
            info = accounting(selected)
            packed_mismatches = 0
            if n <= int(config["packed_validation_max_input_bits"]):
                packed_mismatches = validate_packed(spec, selected)
                totals["packed"] += packed_mismatches
            compile_ns = sum(item[2] for item in variants)
            compile_ops = 2 * classes * n + 2 * n
            saved_ops = (
                info["baseline_operation_count"]
                - info["grouped_operation_count"]
            )
            amortization = (
                math.ceil(compile_ops / saved_ops)
                if saved_ops > 0
                else math.inf
            )
            source_bytes = classes * n * math.ceil(width / 8)
            storage_fraction = info["logical_storage_bytes"] / source_bytes
            projection = storage_fraction * float(
                config["projection"]["target_q4_bytes"]
            )
            cases.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "series": shape["series"],
                    "seed": seed,
                    "input_bits": n,
                    "class_count": classes,
                    "accumulator_bits": width,
                    "validation_mode": (
                        "exhaustive"
                        if exhaustive
                        else "deterministic_sample"
                    ),
                    "validated_assignments_per_plan": len(inputs),
                    "selected_mode": mode,
                    "packed_mismatches": packed_mismatches,
                    "both_compile_elapsed_ns": compile_ns,
                    "required_compile_amortization_queries": amortization,
                    "selected_projected_405b_q4_storage_bytes": projection,
                    **{
                        f"selected_{key}": value
                        for key, value in info.items()
                    },
                }
            )

    operations = [
        float(row["selected_operation_fraction"]) for row in cases
    ]
    byte_fractions = [
        float(row["selected_query_byte_fraction"]) for row in cases
    ]
    dense_unique = [
        float(row["selected_operation_fraction"])
        for row in cases
        if row["family"] in {"dense_random", "forced_unique"}
    ]
    projections = [
        float(row["selected_projected_405b_q4_storage_bytes"])
        for row in cases
    ]
    amortizations = [
        float(row["required_compile_amortization_queries"])
        for row in cases
    ]
    structured = {}
    for family in ("repeated_columns", "sign_related_columns"):
        rows = sorted(
            (
                row
                for row in cases
                if row["family"] == family
                and row["series"] == "input_scaling"
            ),
            key=lambda row: row["input_bits"],
        )
        structured[family] = [
            float(row["selected_operation_fraction"]) for row in rows
        ]
    non_degrading = all(
        all(
            current <= previous + 1e-12
            for previous, current in zip(items, items[1:])
        )
        for items in structured.values()
    )

    p50_ops, p90_ops = pct(operations, 0.5), pct(operations, 0.9)
    p50_bytes, p90_bytes = (
        pct(byte_fractions, 0.5),
        pct(byte_fractions, 0.9),
    )
    dense_p50 = pct(dense_unique, 0.5)
    maximum_projection = max(projections)
    maximum_amortization = max(amortizations)
    gate = config["early_gate"]
    checks = {
        "exact_gate_pass": (
            totals["score"] + totals["top1"] + totals["packed"]
            <= gate["maximum_exact_mismatches"]
        ),
        "operation_gate_pass": (
            p50_ops <= gate["maximum_p50_operation_fraction"]
            and p90_ops <= gate["maximum_p90_operation_fraction"]
        ),
        "byte_gate_pass": (
            p50_bytes <= gate["maximum_p50_byte_fraction"]
            and p90_bytes <= gate["maximum_p90_byte_fraction"]
        ),
        "dense_unique_gate_pass": (
            dense_p50 <= gate["maximum_dense_unique_p50_fraction"]
        ),
        "storage_gate_pass": (
            maximum_projection
            <= gate["maximum_projected_storage_bytes"]
        ),
        "compile_amortization_gate_pass": (
            maximum_amortization
            <= gate["maximum_compile_amortization_queries"]
        ),
        "structured_non_degrading_gate_pass": non_degrading,
        "representation_gate_pass": totals["truth_tables"] == 0,
    }
    survives = all(checks.values())
    decision = (
        "PROMOTE_EXACT_COLUMN_SIGNATURE_AGGREGATION_TO_REAL_WEIGHT_EXTRACTION"
        if survives
        else "REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY"
    )
    summary = {
        "experiment": "EXP-055",
        "name": "exact_column_signature_popcount_aggregation_gate",
        "evidence_level": "E1",
        "phase": ["A", "B"],
        "MEASURED": {
            "case_count": len(cases),
            "plan_count": len(plans),
            "family_count": len(config["families"]),
            "total_scalar_validations": totals["validated"],
            "exhaustive_scalar_validations": totals["exhaustive"],
            "sampled_scalar_validations": totals["sampled"],
            "score_mismatches": totals["score"],
            "top1_mismatches": totals["top1"],
            "packed_mismatches": totals["packed"],
            "truth_table_representations": totals["truth_tables"],
            "p50_operation_fraction": p50_ops,
            "p90_operation_fraction": p90_ops,
            "p50_query_byte_fraction": p50_bytes,
            "p90_query_byte_fraction": p90_bytes,
            "dense_unique_p50_operation_fraction": dense_p50,
            "maximum_projected_405b_q4_storage_bytes": maximum_projection,
            "maximum_projected_405b_q4_storage_tib": (
                maximum_projection / 1024**4
            ),
            "maximum_required_compile_amortization_queries": (
                maximum_amortization
            ),
            "infinite_amortization_case_count": sum(
                math.isinf(item) for item in amortizations
            ),
            "structured_scaling_operation_fractions": structured,
            "peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - experiment_started,
        },
        "DERIVED": {
            **checks,
            "column_signature_aggregation_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "logical 64-bit membership-mask AND/popcount plus exact "
                "per-class multiply/add versus C*n conditional adds; "
                "not hardware latency"
            ),
        },
        "PROJECTED": config["projection"],
        "UNVERIFIED": [
            "real checkpoint weight-column repetition",
            "real Transformer operation replacement",
            "physical grouped-kernel bytes and latency",
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
            "config_sha256": digest(args.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    dump_rows(output / "raw" / "plan_rows.jsonl", plans)
    dump_rows(output / "raw" / "case_rows.jsonl", cases)
    dump(output / "processed" / "aggregate.json", summary)
    dump(output / "summary.json", summary)
    dump(
        output / "artifacts" / "environment.json",
        summary["provenance"],
    )
    (output / "artifacts" / "contract.txt").write_text(
        "EXP-055 Phase A/B E1 synthetic exact grouping Gate.\n"
        "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/sec: NOT TESTED.\n"
    )
    checksums = [
        f"{digest(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
