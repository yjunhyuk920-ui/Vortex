#!/usr/bin/env python3
"""Run EXP-058 pinned real-Q4 exact algebraic-rank certificate Gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from vortex_runtime.modular_rank import (
    certify_integer_rank,
    factorization_lower_bounds,
)
from vortex_runtime.weight_structure import symmetric_row_quantize

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dump_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def matrix_role(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("wte", "wpe", "embed", "embedding")):
        return "embedding"
    if "lm_head" in lowered or "output_projection" in lowered:
        return "output_head"
    return "dense_projection"


def tensor_checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + contiguous.tobytes())


def resolve_snapshot(
    *, model_id: str, revision: str, cache_dir: Path
) -> Path:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model_id, revision=revision)
    if str(info.sha) != revision:
        raise RuntimeError(
            f"resolved revision mismatch for {model_id}: {info.sha} != {revision}"
        )
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=list(MODEL_PATTERNS),
        )
    )


def exp057_q4_checksums() -> dict[tuple[str, str], str]:
    path = ROOT / "results/exp_057/raw/representation_rows.jsonl"
    if not path.exists():
        raise RuntimeError("frozen EXP-057 representation rows are missing")
    result: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["representation"] != "q4_row_symmetric":
            continue
        key = (str(row["model_id"]), str(row["tensor_name"]))
        value = str(row["quantization"]["integer_sha256"])
        if key in result and result[key] != value:
            raise RuntimeError("EXP-057 contains conflicting Q4 checksums")
        result[key] = value
    return result


def certificate_to_dict(certificate: Any) -> dict[str, Any]:
    value = asdict(certificate)
    value["prime_certificates"] = [
        asdict(item) for item in certificate.prime_certificates
    ]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_058/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_058_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_058_huggingface",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
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
    exp057_checksums = exp057_q4_checksums()
    matrix_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    certificate_cache: dict[str, dict[str, Any]] = {}
    q4_checksum_mismatches = 0
    missing_exp057_q4_checksums = 0
    unregistered_dense_projections = 0
    certificate_or_control_mismatches = 0
    experiment_started = time.perf_counter_ns()

    for model_index, model_entry in enumerate(config["models"]):
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
        model_matrix_count = 0
        model_dense_count = 0
        control_candidates: list[tuple[int, str, np.ndarray, str]] = []

        for tensor_index, (name, tensor) in enumerate(sorted(state.items())):
            if tensor.ndim != 2:
                continue
            model_matrix_count += 1
            role = matrix_role(name)
            if role == "dense_projection":
                model_dense_count += 1
            try:
                matrix = tensor.detach().cpu().contiguous().numpy().astype(
                    np.float32, copy=False
                )
                tensor_sha = tensor_checksum(matrix)
                q4 = symmetric_row_quantize(matrix, bits=4)
                q4_sha = sha256_bytes(np.ascontiguousarray(q4.values).tobytes())
                expected_q4_sha = exp057_checksums.get((model_id, name))
                if expected_q4_sha is None:
                    missing_exp057_q4_checksums += 1
                elif expected_q4_sha != q4_sha:
                    q4_checksum_mismatches += 1

                cache_key = sha256_bytes(
                    json.dumps(
                        {
                            "shape": list(q4.values.shape),
                            "q4_sha256": q4_sha,
                            "primes": primes,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                )
                if cache_key in certificate_cache:
                    cached = certificate_cache[cache_key]
                    certificate_dict = cached["certificate"]
                    bounds = cached["bounds"]
                    elapsed_ns = 0
                    reused = True
                else:
                    started = time.perf_counter_ns()
                    certificate = certify_integer_rank(
                        q4.values, primes=primes
                    )
                    certificate.validate(q4.values)
                    elapsed_ns = time.perf_counter_ns() - started
                    certificate_dict = certificate_to_dict(certificate)
                    bounds = factorization_lower_bounds(
                        rows=q4.values.shape[0],
                        columns=q4.values.shape[1],
                        rank_lower_bound=certificate.rank_lower_bound,
                    )
                    certificate_cache[cache_key] = {
                        "certificate": certificate_dict,
                        "bounds": bounds,
                    }
                    reused = False
                row = {
                    "model_id": model_id,
                    "revision": revision,
                    "tensor_name": name,
                    "matrix_role": role,
                    "shape": list(matrix.shape),
                    "source_tensor_sha256": tensor_sha,
                    "q4_integer_sha256": q4_sha,
                    "exp057_q4_integer_sha256": expected_q4_sha,
                    "q4_checksum_matches_exp057": expected_q4_sha == q4_sha,
                    "q4_quantization": {
                        "maximum_absolute_error": q4.maximum_absolute_error,
                        "mean_absolute_error": q4.mean_absolute_error,
                        "zero_row_count": q4.zero_row_count,
                        "clipped_value_count": q4.clipped_value_count,
                    },
                    "certificate": certificate_dict,
                    "factorization_lower_bounds": bounds,
                    "certificate_elapsed_ns": elapsed_ns,
                    "certificate_reused": reused,
                }
                matrix_rows.append(row)
                if (
                    role == "dense_projection"
                    and matrix.shape[0] >= 8
                    and matrix.shape[1] >= 8
                ):
                    control_candidates.append(
                        (int(matrix.size), name, q4.values.copy(), q4_sha)
                    )
            except Exception:
                if role == "dense_projection":
                    unregistered_dense_projections += 1
                raise

        for control_index, (_, name, q4_values, q4_sha) in enumerate(
            sorted(control_candidates, reverse=True)[
                : int(config["control_matrix_per_model"])
            ]
        ):
            seed = int(config["seed"]) + model_index * 100 + control_index
            rng = np.random.default_rng(seed)
            row_permutation = rng.permutation(q4_values.shape[0])
            column_permutation = rng.permutation(q4_values.shape[1])
            original = certify_integer_rank(q4_values, primes=primes)
            permuted = certify_integer_rank(
                q4_values[row_permutation][:, column_permutation],
                primes=primes,
            )
            rank_matches = (
                original.rank_lower_bound == permuted.rank_lower_bound
                and original.full_integer_rank_proven
                == permuted.full_integer_rank_proven
            )
            if not rank_matches:
                certificate_or_control_mismatches += 1
            control_rows.append(
                {
                    "model_id": model_id,
                    "tensor_name": name,
                    "q4_integer_sha256": q4_sha,
                    "original_rank_lower_bound": original.rank_lower_bound,
                    "permuted_rank_lower_bound": permuted.rank_lower_bound,
                    "original_full_rank": original.full_integer_rank_proven,
                    "permuted_full_rank": permuted.full_integer_rank_proven,
                    "row_column_permutation_rank_matches": rank_matches,
                }
            )

        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "two_dimensional_tensor_count": model_matrix_count,
                "dense_projection_tensor_count": model_dense_count,
            }
        )
        del state
        del model

    primary = [
        row
        for row in matrix_rows
        if row["matrix_role"] == config["primary_matrix_role"]
    ]
    if not primary:
        raise RuntimeError("no primary dense projections were certified")
    operation_lower_bounds = [
        float(
            row["factorization_lower_bounds"][
                "operation_fraction_lower_bound"
            ]
        )
        for row in primary
    ]
    storage_lower_bounds = [
        float(
            row["factorization_lower_bounds"][
                "storage_fraction_lower_bound"
            ]
        )
        for row in primary
    ]
    full_rank_count = sum(
        bool(row["certificate"]["full_integer_rank_proven"])
        for row in primary
    )
    break_even_exceeded_count = sum(
        bool(
            row["factorization_lower_bounds"][
                "rank_lower_bound_exceeds_break_even"
            ]
        )
        for row in primary
    )
    ten_percent_exceeded_count = sum(
        bool(
            row["factorization_lower_bounds"][
                "rank_lower_bound_exceeds_10_percent_budget"
            ]
        )
        for row in primary
    )
    twenty_five_percent_exceeded_count = sum(
        bool(
            row["factorization_lower_bounds"][
                "rank_lower_bound_exceeds_25_percent_budget"
            ]
        )
        for row in primary
    )
    p50_operation = percentile(operation_lower_bounds, 0.5)
    p90_operation = percentile(operation_lower_bounds, 0.9)
    p50_storage = percentile(storage_lower_bounds, 0.5)
    p90_storage = percentile(storage_lower_bounds, 0.9)

    parameter_counts = {
        row["model_id"]: int(row["parameter_count"]) for row in model_rows
    }
    model_p50: dict[str, float] = {}
    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        values = [
            float(
                row["factorization_lower_bounds"][
                    "operation_fraction_lower_bound"
                ]
            )
            for row in primary
            if row["model_id"] == model_id
        ]
        model_p50[model_id] = percentile(values, 0.5)
    ordered_models = sorted(model_p50, key=lambda item: parameter_counts[item])
    smallest = model_p50[ordered_models[0]]
    largest = model_p50[ordered_models[-1]]
    size_degradation = (
        0.0
        if largest <= smallest
        else (math.inf if smallest == 0.0 else largest / smallest - 1.0)
    )

    gates = config["early_gate"]
    checks = {
        "certificate_control_gate_pass": certificate_or_control_mismatches
        <= gates["maximum_certificate_or_control_mismatches"],
        "registration_gate_pass": unregistered_dense_projections
        <= gates["maximum_unregistered_dense_projections"],
        "q4_checksum_gate_pass": (
            q4_checksum_mismatches + missing_exp057_q4_checksums
            <= gates["maximum_q4_checksum_mismatches_against_exp_057"]
        ),
        "operation_lower_bound_gate_pass": (
            p50_operation
            <= gates["maximum_p50_operation_lower_bound_fraction"]
            and p90_operation
            <= gates["maximum_p90_operation_lower_bound_fraction"]
        ),
        "storage_lower_bound_gate_pass": (
            p50_storage
            <= gates["maximum_p50_storage_lower_bound_fraction"]
            and p90_storage
            <= gates["maximum_p90_storage_lower_bound_fraction"]
        ),
        "model_size_trend_gate_pass": size_degradation
        <= gates["maximum_model_size_degradation"],
    }
    survives = all(checks.values())
    decision = (
        "PROMOTE_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_TO_CONSTRUCTIVE_FACTOR_GATE"
        if survives
        else config["failure_decision"]
    )

    certificate_prime_distribution: dict[str, int] = {}
    fast_path_count = 0
    for row in primary:
        prime = row["certificate"]["certificate_prime"]
        key = "none" if prime is None else str(prime)
        certificate_prime_distribution[key] = (
            certificate_prime_distribution.get(key, 0) + 1
        )
        if row["certificate"]["prime_certificates"][0][
            "used_leading_minor_fast_path"
        ]:
            fast_path_count += 1

    summary = {
        "experiment": "EXP-058",
        "name": "pinned_real_q4_exact_algebraic_rank_certificate_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "MEASURED": {
            "model_count": len(model_rows),
            "two_dimensional_tensor_count": len(matrix_rows),
            "dense_projection_matrix_count": len(primary),
            "unregistered_dense_projection_count": unregistered_dense_projections,
            "q4_checksum_mismatches_against_exp_057": q4_checksum_mismatches,
            "missing_exp_057_q4_checksum_count": missing_exp057_q4_checksums,
            "certificate_or_control_mismatches": certificate_or_control_mismatches,
            "control_row_count": len(control_rows),
            "full_integer_rank_proven_count": full_rank_count,
            "full_integer_rank_proven_fraction": full_rank_count / len(primary),
            "rank_lower_bound_exceeds_10_percent_budget_count": ten_percent_exceeded_count,
            "rank_lower_bound_exceeds_25_percent_budget_count": twenty_five_percent_exceeded_count,
            "rank_lower_bound_exceeds_break_even_count": break_even_exceeded_count,
            "p50_exact_factor_operation_lower_bound_fraction": p50_operation,
            "p90_exact_factor_operation_lower_bound_fraction": p90_operation,
            "p50_exact_factor_storage_lower_bound_fraction": p50_storage,
            "p90_exact_factor_storage_lower_bound_fraction": p90_storage,
            "model_p50_exact_factor_operation_lower_bound_fraction": model_p50,
            "model_size_degradation_fraction": size_degradation,
            "certificate_prime_distribution": certificate_prime_distribution,
            "leading_minor_fast_path_count": fast_path_count,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - experiment_started,
        },
        "DERIVED": {
            **checks,
            "real_q4_exact_low_rank_survives_gate": survives,
            "decision": decision,
            "rank_logic": (
                "rank modulo a prime is a rigorous lower bound on integer/rational "
                "rank; full rank under one prime proves full integer/rational rank"
            ),
            "factor_accounting_scope": (
                "conventional exact two-factor W=A@B lower bound r*n + m*r "
                "scalar terms and r*(m+n) factor scalars; factor scalar bitwidth "
                "and metadata can only increase physical cost"
            ),
        },
        "UNVERIFIED": [
            "Q4 model output preservation",
            "constructive exact factor arithmetic and bitwidth",
            "factor-kernel execution",
            "actual Transformer operation replacement",
            "70B or 405B algebraic rank",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "authoritative_decision": decision,
        "real_transformer_operation_replacement": False,
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "q4_output_preservation": "NOT TESTED",
            "factor_kernel_execution": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
            "real_transformer_operation_replacement": False,
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp057_summary_sha256": sha256_file(
                ROOT / "results/exp_057/summary.json"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/matrix_rank_rows.jsonl", matrix_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        "EXP-058 exact modular-rank observation on pinned real Q4 matrices.\n"
        "Q4 output preservation, factor kernel, operation replacement: NOT TESTED.\n"
        "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/sec: NOT TESTED.\n"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
