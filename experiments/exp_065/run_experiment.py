#!/usr/bin/env python3
"""Run EXP-065 pinned real-Q4 exact Kronecker-rank Gate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp_059.run_experiment import (
    dump,
    dump_rows,
    exp057_q4_checksums,
    git_commit,
    matrix_role,
    resolve_snapshot,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from vortex_runtime.kronecker_rank import (
    certify_all_factorizations,
    certify_kronecker_plan,
    inverse_rearrangement,
    rearrange_kronecker,
    select_favorable_kronecker_plan,
)
from vortex_runtime.modular_rank import rank_certificate_mod_prime
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def bias_name(weight_name: str) -> str | None:
    return weight_name[:-6] + "bias" if weight_name.endswith("weight") else None


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


def control_population(
    *, seed: int, primes: tuple[int, ...]
) -> tuple[list[dict[str, Any]], int, list[float]]:
    left = np.asarray([[1, 2], [3, 5]], dtype=np.int16)
    right = np.asarray(
        [[2, 0, 1, 3], [1, 4, 2, 0], [3, 1, 0, 2], [1, 2, 4, 3]],
        dtype=np.int16,
    )
    rank_one = np.kron(left, right)

    left_a = np.asarray([[1, 0], [0, 1]], dtype=np.int16)
    left_b = np.asarray([[0, 1], [1, 0]], dtype=np.int16)
    right_a = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    right_b = np.asarray([[2, 0], [1, 3]], dtype=np.int16)
    rank_two = np.kron(left_a, right_a) + np.kron(left_b, right_b)

    mutated = rank_one.copy()
    mutated[0, 0] += 1
    round_trip = inverse_rearrangement(
        rearrange_kronecker(rank_one, m1=2, m2=4, n1=2, n2=4),
        m1=2,
        m2=4,
        n1=2,
        n2=4,
    )

    rank_one_plan = certify_kronecker_plan(
        rank_one,
        m1=2,
        m2=4,
        n1=2,
        n2=4,
        primes=primes,
    )
    rank_two_plan = certify_kronecker_plan(
        rank_two,
        m1=2,
        m2=2,
        n1=2,
        n2=2,
        primes=primes,
    )
    mutation_plan = certify_kronecker_plan(
        mutated,
        m1=2,
        m2=4,
        n1=2,
        n2=4,
        primes=primes,
    )

    rows: list[dict[str, Any]] = []
    failures = 0
    definitions = (
        (
            "rank_one",
            rank_one_plan.rank_lower_bound == 1
            and rank_one_plan.witness_mismatches == 0,
            rank_one_plan.as_dict(),
        ),
        (
            "rank_two",
            rank_two_plan.rank_lower_bound == 2
            and rank_two_plan.witness_mismatches == 0,
            rank_two_plan.as_dict(),
        ),
        (
            "one_scalar_mutation",
            mutation_plan.rank_lower_bound >= 2
            and mutation_plan.witness_mismatches == 0,
            mutation_plan.as_dict(),
        ),
        (
            "rearrangement_round_trip",
            np.array_equal(round_trip, rank_one),
            {"shape": list(rank_one.shape)},
        ),
    )
    for name, passed, detail in definitions:
        failures += int(not passed)
        rows.append({"control": name, "passed": bool(passed), "detail": detail})

    adversary_fractions: list[float] = []
    for index in range(4):
        matrix = np.random.default_rng(seed + index).integers(
            -7, 8, size=(8, 8), dtype=np.int16
        )
        plans = certify_all_factorizations(matrix, primes=primes)
        selected = select_favorable_kronecker_plan(plans)
        passed = (
            selected.witness_mismatches == 0
            and max(plan.rank_lower_bound for plan in plans) >= 8
        )
        failures += int(not passed)
        adversary_fractions.append(selected.operation_fraction)
        rows.append(
            {
                "control": f"dense_random_{index}",
                "passed": passed,
                "selected": selected.as_dict(),
                "factorization_count": len(plans),
                "maximum_rank_lower_bound": max(
                    plan.rank_lower_bound for plan in plans
                ),
            }
        )
    return rows, failures, adversary_fractions


def selected_certificate_rows(
    *,
    model_id: str,
    tensor_name: str,
    matrix: np.ndarray,
    factors: tuple[int, int, int, int],
    primes: tuple[int, ...],
) -> list[dict[str, Any]]:
    m1, m2, n1, n2 = factors
    rearranged = rearrange_kronecker(
        matrix, m1=m1, m2=m2, n1=n1, n2=n2
    )
    rows: list[dict[str, Any]] = []
    for prime in primes:
        certificate = rank_certificate_mod_prime(rearranged, prime=prime)
        certificate.validate(rearranged)
        rows.append(
            {
                "model_id": model_id,
                "tensor_name": tensor_name,
                "factors": list(factors),
                "rearranged_shape": list(rearranged.shape),
                "prime": prime,
                "rank": certificate.rank,
                "minimum_dimension": certificate.minimum_dimension,
                "full_rank": certificate.full_rank,
                "pivot_rows": list(certificate.pivot_rows),
                "pivot_columns": list(certificate.pivot_columns),
                "certified_minor_determinant": (
                    certificate.certified_minor_determinant
                ),
                "used_leading_minor_fast_path": (
                    certificate.used_leading_minor_fast_path
                ),
                "verified": True,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_065/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_065_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_065_huggingface",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config["torch_num_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(config["torch_num_threads"]))

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(config["torch_num_threads"]))
    primes = tuple(int(value) for value in config["primes"])
    bits = int(config["bits_per_factor"])
    activation_bytes = int(config["activation_bytes"])
    expected = exp057_q4_checksums()

    matrix_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    checksum_mismatches = 0
    missing_checksums = 0
    witness_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        state = model.state_dict()
        local: list[dict[str, Any]] = []
        two_d = 0
        dense_count = 0
        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            two_d += 1
            role = matrix_role(tensor_name)
            dense_count += int(role == "dense_projection")
            floating = (
                tensor.detach()
                .cpu()
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            )
            quantized = symmetric_row_quantize(floating, bits=4)
            values = np.ascontiguousarray(quantized.values)
            checksum = sha256_bytes(values.tobytes())
            expected_checksum = expected.get((model_id, tensor_name))
            missing_checksums += int(expected_checksum is None)
            checksum_mismatches += int(
                expected_checksum is not None and expected_checksum != checksum
            )
            candidate_bias = bias_name(tensor_name)
            has_bias = bool(
                candidate_bias
                and candidate_bias in state
                and state[candidate_bias].ndim == 1
                and int(state[candidate_bias].numel()) == values.shape[0]
            )
            compile_started = time.perf_counter_ns()
            plans = certify_all_factorizations(
                values,
                primes=primes,
                bits_per_factor=bits,
                activation_bytes=activation_bytes,
                has_bias=has_bias,
            )
            compile_ns = time.perf_counter_ns() - compile_started
            if plans:
                selected = select_favorable_kronecker_plan(plans)
                selected_dict = selected.as_dict()
                selected_factors = selected.factors
                selected_rank = selected.rank_lower_bound
                selected_full = selected.full_rearrangement_rank_proven
                witness_mismatches += sum(
                    plan.witness_mismatches for plan in plans
                )
                certificate_rows.extend(
                    selected_certificate_rows(
                        model_id=model_id,
                        tensor_name=tensor_name,
                        matrix=values,
                        factors=selected.factors,
                        primes=primes,
                    )
                )
                for plan in plans:
                    plan_rows.append(
                        {
                            "model_id": model_id,
                            "revision": revision,
                            "tensor_name": tensor_name,
                            "matrix_role": role,
                            "q4_integer_sha256": checksum,
                            "selected": plan.factors == selected.factors,
                            **plan.as_dict(),
                        }
                    )
            else:
                selected_dict = dense_accounting(
                    int(values.shape[0]),
                    int(values.shape[1]),
                    bits=bits,
                    activation_bytes=activation_bytes,
                    has_bias=has_bias,
                )
                selected_factors = None
                selected_rank = None
                selected_full = False
            row = {
                "model_id": model_id,
                "revision": revision,
                "tensor_name": tensor_name,
                "matrix_role": role,
                "row_count": int(values.shape[0]),
                "column_count": int(values.shape[1]),
                "has_bias": has_bias,
                "q4_integer_sha256": checksum,
                "expected_q4_integer_sha256": expected_checksum,
                "checksum_match": expected_checksum == checksum,
                "scale_sha256": sha256_bytes(
                    np.ascontiguousarray(quantized.scales).tobytes()
                ),
                "factorization_count": len(plans),
                "compile_elapsed_ns": compile_ns,
                "selected_factors": (
                    list(selected_factors) if selected_factors else None
                ),
                "selected_rank_lower_bound": selected_rank,
                "selected_full_rearrangement_rank_proven": selected_full,
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
                local.append(row)
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "two_dimensional_tensor_count": two_d,
                "dense_projection_count": dense_count,
                "p50_operation_fraction": percentile(
                    [row["selected_operation_fraction"] for row in local], 0.50
                ),
                "p90_operation_fraction": percentile(
                    [row["selected_operation_fraction"] for row in local], 0.90
                ),
                "p50_storage_fraction": percentile(
                    [row["selected_storage_fraction"] for row in local], 0.50
                ),
                "p90_storage_fraction": percentile(
                    [row["selected_storage_fraction"] for row in local], 0.90
                ),
            }
        )
        del model, state

    control_rows, control_failures, adversary = control_population(
        seed=int(config["seed"]), primes=primes
    )
    dense_rows = [
        row for row in matrix_rows if row["matrix_role"] == "dense_projection"
    ]
    operations = [row["selected_operation_fraction"] for row in dense_rows]
    storage = [row["selected_storage_fraction"] for row in dense_rows]
    query_bytes = [row["selected_query_byte_fraction"] for row in dense_rows]
    aggregate_source_storage = sum(
        row["baseline_storage_bytes"] for row in dense_rows
    )
    aggregate_lower_storage = sum(
        row["selected_lower_bound_storage_bytes"] for row in dense_rows
    )
    aggregate_storage_fraction = (
        aggregate_lower_storage / aggregate_source_storage
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
    largest_degradation = (
        model_p50[largest_id] / best_model - 1.0
        if best_model > 0
        else math.inf
    )
    gate = config["gate"]
    measured = {
        "model_count": len(model_rows),
        "two_dimensional_tensor_count": len(matrix_rows),
        "dense_projection_count": len(dense_rows),
        "plan_row_count": len(plan_rows),
        "selected_certificate_row_count": len(certificate_rows),
        "checksum_mismatches": checksum_mismatches,
        "missing_checksums": missing_checksums,
        "witness_mismatches": witness_mismatches,
        "control_failures": control_failures,
        "dense_without_nontrivial_factorization": sum(
            row["factorization_count"] == 0 for row in dense_rows
        ),
        "p50_operation_fraction": percentile(operations, 0.50),
        "p90_operation_fraction": percentile(operations, 0.90),
        "p50_storage_fraction": percentile(storage, 0.50),
        "p90_storage_fraction": percentile(storage, 0.90),
        "p50_query_byte_fraction": percentile(query_bytes, 0.50),
        "p90_query_byte_fraction": percentile(query_bytes, 0.90),
        "minimum_operation_fraction": min(operations),
        "minimum_storage_fraction": min(storage),
        "minimum_query_byte_fraction": min(query_bytes),
        "dense_random_control_p50": percentile(adversary, 0.50),
        "selected_full_rearrangement_rank_count": sum(
            row["selected_full_rearrangement_rank_proven"] for row in dense_rows
        ),
        "selected_rank_distribution": dict(
            Counter(
                str(row["selected_rank_lower_bound"])
                for row in dense_rows
            )
        ),
        "aggregate_lower_bound_storage_fraction": (
            aggregate_storage_fraction
        ),
        "projected_405b_lower_bound_storage_bytes": projected_storage,
        "model_p50_operation_fraction": model_p50,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    gates = {
        "correctness_gate_pass": checksum_mismatches == 0
        and missing_checksums == 0
        and witness_mismatches == 0
        and control_failures == 0,
        "population_gate_pass": len(matrix_rows)
        == int(gate["expected_two_dimensional_tensors"])
        and len(dense_rows) == int(gate["expected_dense_projections"]),
        "operation_gate_pass": measured["p50_operation_fraction"]
        <= float(gate["maximum_p50_operation_fraction"])
        and measured["p90_operation_fraction"]
        <= float(gate["maximum_p90_operation_fraction"]),
        "storage_gate_pass": measured["p50_storage_fraction"]
        <= float(gate["maximum_p50_storage_fraction"])
        and measured["p90_storage_fraction"]
        <= float(gate["maximum_p90_storage_fraction"])
        and projected_storage
        <= float(gate["maximum_projected_storage_bytes"]),
        "adversary_gate_pass": measured["dense_random_control_p50"]
        <= float(gate["maximum_dense_random_control_p50"]),
        "model_trend_gate_pass": largest_degradation
        <= float(gate["maximum_largest_model_degradation"]),
    }
    lower_bound_survives = all(gates.values())
    decision = (
        "RETAIN_REAL_Q4_KRONECKER_CANDIDATES_FOR_EXACT_RECONSTRUCTION_GATE"
        if lower_bound_survives
        else str(config["failure_decision"])
    )
    summary = {
        "experiment": "EXP-065",
        "name": "pinned_real_q4_exact_kronecker_rearrangement_rank_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "kronecker_lower_bound_survives_gate": lower_bound_survives,
            "decision": decision,
            **gates,
            "exact_integer_factor_reconstruction_gate_pass": False,
            "accounting_scope": (
                "two-prime exact rearrangement-rank witnesses and favorable "
                "lower bounds for 4-bit factors, reshape multiplies, "
                "intermediates, scales, biases, metadata and query bytes"
            ),
        },
        "UNVERIFIED": [
            "exact integer Kronecker factor reconstruction",
            "Q4 model-output preservation",
            "physical Kronecker kernel",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "exact_factor_reconstruction": "NOT TESTED",
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
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/matrix_rows.jsonl", matrix_rows)
    dump_rows(output / "raw/plan_rows.jsonl", plan_rows)
    dump_rows(output / "raw/certificate_rows.jsonl", certificate_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gates["correctness_gate_pass"] or not gates["population_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
