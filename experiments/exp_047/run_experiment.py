#!/usr/bin/env python3
"""Run EXP-047 Phase A/B measurements in the current CPU environment."""

from __future__ import annotations

import argparse
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

from vortex_runtime.cptc import CPTCConfig, certify_sum_sign, exact_reference


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "exp_047"


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    config = json.loads(raw)
    config["_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def independent_alpha(delta: float, n: int) -> float:
    return delta * 6.0 / (math.pi * math.pi * n * n)


def independent_radius(n_total: int, n_sample: int, low: float, high: float, delta: float) -> float:
    if n_sample == n_total or low == high:
        return 0.0
    fpc = 1.0 - (n_sample - 1.0) / n_total
    return n_total * (high - low) * math.sqrt(
        fpc * math.log(2.0 / delta) / (2.0 * n_sample)
    )


def positive_control(size: int) -> list[float]:
    positive = int(size * 0.75)
    return [1.0] * positive + [-0.2] * (size - positive)


def negative_control(size: int) -> list[float]:
    return [-value for value in positive_control(size)]


def zero_margin(size: int) -> list[float]:
    values = [1.0, -1.0] * (size // 2)
    return values[:size]


def one_dominant(size: int) -> list[float]:
    values = [1.0 / max(size - 1, 1)] * size
    values[-1] = -1.0
    return values


def misleading_prefix(size: int, *, seed: int, fraction: float) -> list[float]:
    order = list(range(size))
    random.Random(seed).shuffle(order)
    values = [-1.0] * size
    for index in order[: math.ceil(size * fraction)]:
        values[index] = 0.2
    return values


def timed_case(
    *,
    name: str,
    values: list[float],
    seed: int,
    delta: float,
    fraction: float,
    low: float,
    high: float,
    base_margin: float = 0.0,
) -> dict[str, Any]:
    ref_start = time.perf_counter_ns()
    reference = exact_reference(values, base_margin=base_margin)
    ref_elapsed = time.perf_counter_ns() - ref_start

    config = CPTCConfig(
        delta=delta,
        min_samples=2,
        max_sample_fraction=fraction,
        seed=seed,
    )
    opt_start = time.perf_counter_ns()
    result = certify_sum_sign(
        values,
        value_min=low,
        value_max=high,
        base_margin=base_margin,
        config=config,
    )
    opt_elapsed = time.perf_counter_ns() - opt_start

    independent_match = True
    if result.certified:
        n = result.sampled_before_decision
        delta_n = independent_alpha(delta, n)
        radius = independent_radius(len(values), n, low, high, delta_n)
        estimate = base_margin + len(values) * result.sampled_sum / n
        independent_match = (
            math.isclose(estimate - radius, result.lower_bound, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(estimate + radius, result.upper_bound, rel_tol=1e-12, abs_tol=1e-12)
        )

    wrong_accept = bool(result.certified and result.decision != reference.decision)
    fallback_mismatch = bool(result.fallback and result.decision != reference.decision)
    return {
        "name": name,
        "population_size": len(values),
        "seed": seed,
        "reference_total": reference.total,
        "reference_decision": reference.decision,
        "decision": result.decision,
        "certified": result.certified,
        "fallback": result.fallback,
        "sampled_before_decision": result.sampled_before_decision,
        "total_tiles_evaluated": result.total_tiles_evaluated,
        "sampled_fraction_before_decision": result.sampled_fraction_before_decision,
        "evaluated_fraction": result.evaluated_fraction,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "delta_at_stop": result.delta_spent_at_stop,
        "wrong_accept": wrong_accept,
        "fallback_mismatch": fallback_mismatch,
        "independent_bound_match": independent_match,
        "reference_elapsed_ns": ref_elapsed,
        "optimized_elapsed_ns": opt_elapsed,
    }


def ensure_dirs() -> dict[str, Path]:
    paths = {
        name: RESULT_ROOT / name
        for name in ["raw", "processed", "logs", "artifacts"]
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_checksums() -> None:
    records: list[str] = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.relative_to(RESULT_ROOT).as_posix()}")
    (RESULT_ROOT / "checksums.sha256").write_text("\n".join(records) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "exp_047" / "config.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_dirs()

    delta = float(config["delta_per_decision"])
    fraction = float(config["max_sample_fraction"])
    low = float(config["declared_value_min"])
    high = float(config["declared_value_max"])
    base_seed = int(config["seed"])
    sizes = [int(value) for value in config["population_sizes"]]
    random_cases = int(config["random_cases_per_size"])

    cases: list[dict[str, Any]] = []
    for size in sizes:
        cases.append(
            timed_case(
                name="positive_cancellation",
                values=positive_control(size),
                seed=base_seed + size,
                delta=delta,
                fraction=fraction,
                low=low,
                high=high,
            )
        )
        cases.append(
            timed_case(
                name="negative_cancellation",
                values=negative_control(size),
                seed=base_seed + size,
                delta=delta,
                fraction=fraction,
                low=low,
                high=high,
            )
        )
        cases.append(
            timed_case(
                name="zero_margin",
                values=zero_margin(size),
                seed=base_seed + 2 * size,
                delta=delta,
                fraction=fraction,
                low=low,
                high=high,
            )
        )
        cases.append(
            timed_case(
                name="one_dominant",
                values=one_dominant(size),
                seed=base_seed + 3 * size,
                delta=delta,
                fraction=fraction,
                low=low,
                high=high,
            )
        )
        misleading_seed = base_seed + 4 * size
        cases.append(
            timed_case(
                name="misleading_prefix",
                values=misleading_prefix(size, seed=misleading_seed, fraction=fraction),
                seed=misleading_seed,
                delta=delta,
                fraction=fraction,
                low=low,
                high=high,
            )
        )

        rng = random.Random(base_seed + 10_000 + size)
        for case_index in range(random_cases):
            values = [rng.uniform(low, high) for _ in range(size)]
            base_margin = rng.uniform(-0.1 * size, 0.1 * size)
            cases.append(
                timed_case(
                    name="random_uniform",
                    values=values,
                    seed=base_seed + size * 1_000 + case_index,
                    delta=delta,
                    fraction=fraction,
                    low=low,
                    high=high,
                    base_margin=base_margin,
                )
            )

    raw_path = paths["raw"] / "cases.jsonl"
    raw_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )

    wrong_accepts = sum(int(case["wrong_accept"]) for case in cases)
    fallback_mismatches = sum(int(case["fallback_mismatch"]) for case in cases)
    bound_mismatches = sum(int(not case["independent_bound_match"]) for case in cases)
    certified = [case for case in cases if case["certified"]]
    fallback = [case for case in cases if case["fallback"]]

    max_size = max(sizes)
    max_positive = next(
        case
        for case in cases
        if case["name"] == "positive_cancellation" and case["population_size"] == max_size
    )
    adversarial = [
        case
        for case in cases
        if case["name"] in {"zero_margin", "one_dominant", "misleading_prefix"}
    ]
    adversarial_fallback_pass = all(case["fallback"] for case in adversarial)

    scaling: list[dict[str, Any]] = []
    for size in sizes:
        bucket = [case for case in cases if case["population_size"] == size]
        scaling.append(
            {
                "population_size": size,
                "cases": len(bucket),
                "certified_fraction": sum(int(case["certified"]) for case in bucket) / len(bucket),
                "fallback_fraction": sum(int(case["fallback"]) for case in bucket) / len(bucket),
                "mean_evaluated_fraction": sum(case["evaluated_fraction"] for case in bucket) / len(bucket),
                "mean_optimized_elapsed_ns": sum(case["optimized_elapsed_ns"] for case in bucket) / len(bucket),
                "mean_reference_elapsed_ns": sum(case["reference_elapsed_ns"] for case in bucket) / len(bucket),
            }
        )
    (paths["processed"] / "scaling.json").write_text(
        json.dumps(scaling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    target_parameters = int(config["target_parameters"])
    baseline_parameters = int(config["baseline_parameters"])
    bits_per_weight = int(config["bits_per_weight"])
    target_multiplier = float(config["target_traffic_multiplier"])
    target_full_gib = target_parameters * bits_per_weight / 8.0 / (2**30)
    baseline_full_gib = baseline_parameters * bits_per_weight / 8.0 / (2**30)
    allowed_gib = target_multiplier * baseline_full_gib
    required_fraction = allowed_gib / target_full_gib

    gate_passes = (
        wrong_accepts == int(config["wrong_accept_limit"])
        and fallback_mismatches == 0
        and bound_mismatches == 0
        and bool(max_positive["certified"])
        and max_positive["sampled_fraction_before_decision"]
        <= float(config["positive_control_max_fraction"])
        and adversarial_fallback_pass
    )

    summary = {
        "experiment": "EXP-047",
        "name": "causal_probabilistic_tile_certificate",
        "phase": ["A", "B"],
        "evidence_level": "E1" if gate_passes else "E0",
        "git_commit": git_commit(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "config_sha256": config["_sha256"],
        "checkpoint": None,
        "future_information_used": False,
        "phase_d_status": "NOT TESTED",
        "MEASURED": {
            "case_count": len(cases),
            "certified_count": len(certified),
            "fallback_count": len(fallback),
            "wrong_accepts": wrong_accepts,
            "fallback_mismatches": fallback_mismatches,
            "independent_bound_mismatches": bound_mismatches,
            "max_size_positive_control": max_positive,
            "adversarial_case_count": len(adversarial),
            "adversarial_fallback_pass": adversarial_fallback_pass,
            "scaling": scaling,
        },
        "DERIVED": {
            "alpha_spending_total_upper_bound": delta,
            "certificate_contract": "fixed-n Serfling intervals plus alpha-spending union bound over adaptive stops",
            "exact_fallback_contract": "all remaining tiles evaluated before commit",
            "synthetic_positive_fraction_over_required_405b_fraction": (
                max_positive["sampled_fraction_before_decision"] / required_fraction
            ),
        },
        "PROJECTED": {
            "target_parameters": target_parameters,
            "baseline_parameters": baseline_parameters,
            "bits_per_weight": bits_per_weight,
            "target_q4_full_weight_gib_per_full_pass": target_full_gib,
            "baseline_q4_full_weight_gib_per_full_pass": baseline_full_gib,
            "allowed_1_2x_baseline_weight_gib_per_token": allowed_gib,
            "required_average_target_weight_fraction_before_selector_and_fallback": required_fraction,
            "symbolic_weight_gib_by_fraction": {
                str(value): target_full_gib * value
                for value in [0.01, 0.01185, 0.025, 0.25, 0.5, 1.0]
            },
        },
        "UNVERIFIED": [
            "sound per-tile bounds derived from real Transformer weights and activations",
            "real-model certified tile omission",
            "model-wide nonlinear error propagation",
            "held-out prompt coverage",
            "selector overhead on accelerator hardware",
            "70B/405B scaling",
            "8 GiB VRAM execution",
            "PCIe and SSD traffic",
            "4B-class TTFT or tokens per second",
        ],
        "gate": {
            "passes": gate_passes,
            "positive_control_threshold": config["positive_control_max_fraction"],
            "wrong_accept_limit": config["wrong_accept_limit"],
            "decision": "PROMOTE_TO_PHASE_C_DESIGN" if gate_passes else "REVISE_OR_REJECT_PHASE_B",
        },
    }

    summary_path = RESULT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (paths["logs"] / "run.log").write_text(
        json.dumps(
            {
                "gate_passes": gate_passes,
                "case_count": len(cases),
                "wrong_accepts": wrong_accepts,
                "fallback_mismatches": fallback_mismatches,
                "bound_mismatches": bound_mismatches,
                "phase_d_status": "NOT TESTED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths["artifacts"] / "certificate_contract.txt").write_text(
        "Phase B only. Fixed-time Serfling total bounds use alpha spending "
        "delta_n = delta * 6/(pi^2 n^2). Failure to certify triggers an exact "
        "full-tile fallback. Phase D is NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
