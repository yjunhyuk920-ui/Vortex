#!/usr/bin/env python3
"""Run EXP-067 pinned real-Q4 joint multi-projection exact-reuse Gate."""
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
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.exp_059.run_experiment import (
    dump,
    dump_rows,
    exp057_q4_checksums,
    git_commit,
    resolve_snapshot,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from vortex_runtime.joint_projection_reuse import (
    analyze_joint_rows,
    exact_repeated_block_stats,
)
from vortex_runtime.modular_rank import rank_certificate_mod_prime
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]
EXP058_ROWS = ROOT / "results/exp_058/raw/matrix_rank_rows.jsonl"

GROUP_SPECS: dict[str, tuple[str, ...]] = {
    "attention_qkv": (
        "attn.attention.q_proj.weight",
        "attn.attention.k_proj.weight",
        "attn.attention.v_proj.weight",
    ),
    "mlp_gate_up": (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ),
}


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
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
    return rows


def discover_projection_groups(names: Iterable[str]) -> list[dict[str, Any]]:
    population = set(str(name) for name in names)
    groups: list[dict[str, Any]] = []
    for group_type, suffixes in GROUP_SPECS.items():
        anchor = suffixes[0]
        for name in sorted(population):
            if not name.endswith(anchor):
                continue
            prefix = name[: -len(anchor)]
            members = tuple(prefix + suffix for suffix in suffixes)
            groups.append(
                {
                    "group_type": group_type,
                    "group_id": prefix.rstrip(".") + ":" + group_type,
                    "member_names": members,
                    "complete": all(
                        member in population for member in members
                    ),
                    "missing_members": [
                        member
                        for member in members
                        if member not in population
                    ],
                }
            )
    return groups


def exp058_index() -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(EXP058_ROWS):
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-058 row: {key}")
        indexed[key] = row
    return indexed


def common_right_rank(
    *,
    model_id: str,
    member_names: Sequence[str],
    matrices: Sequence[np.ndarray],
    rank_index: dict[tuple[str, str], dict[str, Any]],
    primes: Sequence[int],
) -> tuple[int, str, list[int], int]:
    columns = int(matrices[0].shape[1])
    member_ranks: list[int] = []
    evidence_mismatches = 0
    for name, matrix in zip(member_names, matrices, strict=True):
        source = rank_index.get((model_id, name))
        if source is None:
            evidence_mismatches += 1
            member_ranks.append(0)
            continue
        shape = tuple(int(value) for value in source["shape"])
        evidence_mismatches += int(
            shape != tuple(int(value) for value in matrix.shape)
        )
        rank = int(source["certificate"]["rank_lower_bound"])
        member_ranks.append(rank)
        evidence_mismatches += int(
            not bool(source["certificate"]["full_integer_rank_proven"])
        )
    if member_ranks and max(member_ranks) == columns:
        return (
            columns,
            "EXP-058 member full-column-rank witness",
            member_ranks,
            evidence_mismatches,
        )

    stacked = np.ascontiguousarray(np.concatenate(matrices, axis=0))
    ranks = []
    for prime in primes:
        certificate = rank_certificate_mod_prime(stacked, prime=int(prime))
        certificate.validate(stacked)
        ranks.append(certificate.rank)
    return (
        max(ranks),
        "direct stacked modular-rank witness",
        member_ranks,
        evidence_mismatches,
    )


def control_population(
    seed: int,
) -> tuple[list[dict[str, Any]], int, float]:
    rows: list[dict[str, Any]] = []
    failures = 0
    first = np.asarray(
        [[1, 2, 0], [2, 4, 0], [3, 0, 1]], dtype=np.int8
    )
    second = np.asarray(
        [[-1, -2, 0], [0, 0, 0], [3, 0, 1]], dtype=np.int8
    )
    plan = analyze_joint_rows((first, second))
    passed = plan.primitive_class_count == 2 and plan.reusable_rows == 3
    failures += int(not passed)
    rows.append(
        {
            "control": "duplicate_sign_proportional",
            "passed": passed,
            **plan.as_dict(),
        }
    )

    mutated = second.copy()
    mutated[0, 0] += 1
    mutation = analyze_joint_rows((first, mutated))
    passed = mutation.reusable_rows < plan.reusable_rows
    failures += int(not passed)
    rows.append(
        {
            "control": "one_nibble_mutation",
            "passed": passed,
            **mutation.as_dict(),
        }
    )

    rng = np.random.default_rng(seed)
    random_a = rng.integers(-7, 8, size=(64, 32), dtype=np.int8)
    random_b = rng.integers(-7, 8, size=(64, 32), dtype=np.int8)
    random_plan = analyze_joint_rows((random_a, random_b))
    passed = random_plan.reusable_row_fraction <= 0.01
    failures += int(not passed)
    rows.append(
        {
            "control": "dense_random",
            "passed": passed,
            **random_plan.as_dict(),
        }
    )

    block = np.arange(16, dtype=np.int8).reshape(4, 4)
    block_stats = exact_repeated_block_stats(
        (np.vstack((block, block)), block), block_rows=4
    )
    passed = int(block_stats["reusable_block_count"]) == 2
    failures += int(not passed)
    rows.append(
        {"control": "repeated_block", "passed": passed, **block_stats}
    )
    return rows, failures, random_plan.reusable_row_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_067/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_067_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_067_huggingface",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault(
        "OMP_NUM_THREADS", str(config["torch_num_threads"])
    )
    os.environ.setdefault(
        "MKL_NUM_THREADS", str(config["torch_num_threads"])
    )

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(config["torch_num_threads"]))
    expected = exp057_q4_checksums()
    ranks = exp058_index()
    primes = tuple(int(value) for value in config["primes"])
    block_sizes = tuple(int(value) for value in config["block_rows"])

    group_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    checksum_mismatches = 0
    missing_checksums = 0
    rank_evidence_mismatches = 0
    hash_collision_mismatches = 0
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
            snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        state = model.state_dict()
        groups = discover_projection_groups(state.keys())
        relevant = {
            name
            for group in groups
            for name in group["member_names"]
            if group["complete"]
        }
        quantized_values: dict[str, np.ndarray] = {}
        two_d_count = 0
        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            two_d_count += 1
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
                expected_checksum is not None
                and checksum != expected_checksum
            )
            tensor_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "tensor_name": tensor_name,
                    "shape": list(values.shape),
                    "q4_integer_sha256": checksum,
                    "expected_q4_integer_sha256": expected_checksum,
                    "checksum_match": checksum == expected_checksum,
                    "analyzed_in_joint_group": tensor_name in relevant,
                }
            )
            if tensor_name in relevant:
                quantized_values[tensor_name] = values

        local: list[dict[str, Any]] = []
        for group in groups:
            if not group["complete"]:
                incomplete_rows.append(
                    {"model_id": model_id, "revision": revision, **group}
                )
                continue
            member_names = tuple(
                str(value) for value in group["member_names"]
            )
            matrices = tuple(
                quantized_values[name] for name in member_names
            )
            plan = analyze_joint_rows(matrices)
            hash_collision_mismatches += plan.hash_collision_mismatches
            (
                right_rank,
                right_source,
                member_ranks,
                mismatches,
            ) = common_right_rank(
                model_id=model_id,
                member_names=member_names,
                matrices=matrices,
                rank_index=ranks,
                primes=primes,
            )
            rank_evidence_mismatches += mismatches
            block_stats = [
                exact_repeated_block_stats(matrices, block_rows=size)
                for size in block_sizes
            ]
            row = {
                "model_id": model_id,
                "revision": revision,
                **group,
                **plan.as_dict(),
                "common_right_rank": right_rank,
                "common_right_rank_fraction": right_rank / plan.columns,
                "common_right_rank_source": right_source,
                "member_rank_lower_bounds": member_ranks,
                "block_stats": block_stats,
            }
            group_rows.append(row)
            local.append(row)
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "two_dimensional_tensor_count": two_d_count,
                "complete_group_count": len(local),
                "group_type_distribution": dict(
                    Counter(row["group_type"] for row in local)
                ),
                "p50_operation_fraction": percentile(
                    [row["operation_fraction"] for row in local], 0.50
                ),
                "p90_operation_fraction": percentile(
                    [row["operation_fraction"] for row in local], 0.90
                ),
                "p50_storage_fraction": percentile(
                    [row["storage_fraction"] for row in local], 0.50
                ),
                "p90_storage_fraction": percentile(
                    [row["storage_fraction"] for row in local], 0.90
                ),
                "maximum_reusable_row_fraction": max(
                    row["reusable_row_fraction"] for row in local
                ),
            }
        )
        del model, state

    control_rows, control_failures, random_reuse = control_population(
        int(config["seed"])
    )
    operations = [row["operation_fraction"] for row in group_rows]
    storage = [row["storage_fraction"] for row in group_rows]
    reusable = [row["reusable_row_fraction"] for row in group_rows]
    right_fractions = [
        row["common_right_rank_fraction"] for row in group_rows
    ]
    model_p50 = {
        row["model_id"]: row["p50_operation_fraction"]
        for row in model_rows
    }
    best_model = min(model_p50.values())
    largest_id = str(config["models"][-1]["model_id"])
    largest_degradation = (
        model_p50[largest_id] / best_model - 1.0
        if best_model
        else math.inf
    )
    type_distribution = Counter(row["group_type"] for row in group_rows)
    gate = config["gate"]

    correctness_gate = (
        checksum_mismatches == 0
        and missing_checksums == 0
        and rank_evidence_mismatches == 0
        and hash_collision_mismatches == 0
        and control_failures == 0
    )
    population_gate = (
        len(group_rows) == int(gate["expected_complete_groups"])
        and int(type_distribution.get("attention_qkv", 0))
        == int(gate["expected_attention_qkv_groups"])
        and int(type_distribution.get("mlp_gate_up", 0))
        == int(gate["expected_mlp_gate_up_groups"])
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
    )
    common_factor_gate = (
        percentile(right_fractions, 0.50)
        <= float(gate["maximum_p50_common_right_rank_fraction"])
        and percentile(right_fractions, 0.90)
        <= float(gate["maximum_p90_common_right_rank_fraction"])
    )
    random_gate = random_reuse <= float(
        gate["maximum_random_reusable_row_fraction"]
    )
    trend_gate = largest_degradation <= float(
        gate["maximum_largest_model_degradation"]
    )
    survives = all(
        (
            correctness_gate,
            population_gate,
            operation_gate,
            storage_gate,
            common_factor_gate,
            random_gate,
            trend_gate,
        )
    )
    decision = (
        "RETAIN_REAL_Q4_EXACT_JOINT_ROW_REUSE_FOR_REPLAY_GATE"
        if survives
        else str(config["failure_decision"])
    )
    measured = {
        "model_count": len(model_rows),
        "two_dimensional_tensor_count": len(tensor_rows),
        "complete_group_count": len(group_rows),
        "incomplete_group_count": len(incomplete_rows),
        "group_type_distribution": dict(type_distribution),
        "checksum_mismatches": checksum_mismatches,
        "missing_checksums": missing_checksums,
        "rank_evidence_mismatches": rank_evidence_mismatches,
        "hash_collision_mismatches": hash_collision_mismatches,
        "control_failures": control_failures,
        "p50_operation_fraction": percentile(operations, 0.50),
        "p90_operation_fraction": percentile(operations, 0.90),
        "p50_storage_fraction": percentile(storage, 0.50),
        "p90_storage_fraction": percentile(storage, 0.90),
        "p50_reusable_row_fraction": percentile(reusable, 0.50),
        "p90_reusable_row_fraction": percentile(reusable, 0.90),
        "maximum_reusable_row_fraction": max(reusable),
        "total_reusable_rows": sum(
            int(row["reusable_rows"]) for row in group_rows
        ),
        "total_rows": sum(int(row["total_rows"]) for row in group_rows),
        "p50_common_right_rank_fraction": percentile(
            right_fractions, 0.50
        ),
        "p90_common_right_rank_fraction": percentile(
            right_fractions, 0.90
        ),
        "random_control_reusable_row_fraction": random_reuse,
        "model_p50_operation_fraction": model_p50,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    gates = {
        "correctness_gate_pass": correctness_gate,
        "population_gate_pass": population_gate,
        "operation_gate_pass": operation_gate,
        "storage_gate_pass": storage_gate,
        "common_right_factor_gate_pass": common_factor_gate,
        "random_control_gate_pass": random_gate,
        "model_trend_gate_pass": trend_gate,
    }
    summary = {
        "experiment": "EXP-067",
        "name": "pinned_real_q4_joint_multi_projection_exact_reuse_gate",
        "phase": ["A", "B", "C-weight-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "joint_exact_reuse_survives_gate": survives,
            "decision": decision,
            **gates,
            "exact_joint_replay_gate_pass": False,
            "accounting_scope": (
                "one primitive integer dot product per exact proportional row "
                "class, per-row scale/multiplier corrections, canonical maps, "
                "metadata, and exact shared-input rank lower bounds"
            ),
        },
        "UNVERIFIED": [
            "bitwise floating-point replay equivalence",
            "joint physical kernel",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "q4_integer_domain_reuse": "MEASURED",
            "floating_point_output_preservation": "NOT TESTED",
            "physical_kernel": "NOT TESTED",
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp058_rows_sha256": sha256_file(EXP058_ROWS),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/tensor_rows.jsonl", tensor_rows)
    dump_rows(output / "raw/group_rows.jsonl", group_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(
        output / "raw/incomplete_group_rows.jsonl", incomplete_rows
    )
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
