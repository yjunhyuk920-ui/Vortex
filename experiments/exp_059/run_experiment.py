#!/usr/bin/env python3
"""Run EXP-059 pinned real-Q4 exact shift-displacement rank Gate."""

from __future__ import annotations

import argparse
from collections import Counter
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

from vortex_runtime.displacement_rank import (
    REGISTERED_OPERATORS,
    certify_registered_displacements,
    select_favorable_displacement,
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_rows(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def resolve_snapshot(*, model_id: str, revision: str, cache_dir: Path) -> Path:
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
        checksum = str(row["quantization"]["integer_sha256"])
        if key in result and result[key] != checksum:
            raise RuntimeError("EXP-057 contains conflicting Q4 checksums")
        result[key] = checksum
    return result


def certificate_dict(certificate: Any) -> dict[str, Any]:
    value = asdict(certificate)
    value["prime_certificates"] = [
        asdict(item) for item in certificate.prime_certificates
    ]
    return value


def toeplitz(rows: int, columns: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(-7, 8, size=rows + columns - 1, dtype=np.int64)
    return np.asarray(
        [[values[column - row + rows - 1] for column in range(columns)] for row in range(rows)],
        dtype=np.int64,
    )


def hankel(rows: int, columns: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(-7, 8, size=rows + columns - 1, dtype=np.int64)
    return np.asarray(
        [[values[row + column] for column in range(columns)] for row in range(rows)],
        dtype=np.int64,
    )


def circulant(size: int, *, seed: int) -> np.ndarray:
    first = np.random.default_rng(seed).integers(-7, 8, size=size, dtype=np.int64)
    return np.asarray(
        [[first[(row - column) % size] for column in range(size)] for row in range(size)],
        dtype=np.int64,
    )


def control_rows(*, primes: tuple[int, ...], seed: int) -> tuple[list[dict[str, Any]], int]:
    controls = (
        ("toeplitz", toeplitz(12, 16, seed=seed), "zero_fill_diagonal", 2, "at_most"),
        ("hankel", hankel(12, 16, seed=seed + 1), "zero_fill_antidiagonal", 2, "at_most"),
        ("circulant", circulant(12, seed=seed + 2), "cyclic_diagonal", 0, "equal"),
        (
            "dense_random",
            np.random.default_rng(seed + 3).integers(-7, 8, size=(12, 12), dtype=np.int64),
            "best_registered",
            9,
            "at_least",
        ),
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    for name, matrix, expected_operator, threshold, relation in controls:
        certificates = certify_registered_displacements(matrix, primes=primes)
        selected = select_favorable_displacement(certificates)
        by_name = {item.operator: item for item in certificates}
        observed = (
            selected.rank_lower_bound
            if expected_operator == "best_registered"
            else by_name[expected_operator].rank_lower_bound
        )
        passed = (
            observed <= threshold
            if relation == "at_most"
            else observed >= threshold
            if relation == "at_least"
            else observed == threshold
        )
        failures += int(not passed)
        rows.append(
            {
                "control": name,
                "shape": list(matrix.shape),
                "expected_operator": expected_operator,
                "relation": relation,
                "threshold": threshold,
                "observed_rank": observed,
                "selected_operator": selected.operator,
                "selected_rank": selected.rank_lower_bound,
                "passed": passed,
                "operator_ranks": {
                    item.operator: item.rank_lower_bound for item in certificates
                },
            }
        )
    return rows, failures


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_059/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_059_candidate"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/exp_059_huggingface"
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
    expected_checksums = exp057_q4_checksums()
    matrix_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    checksum_mismatches = 0
    missing_checksums = 0
    unregistered_dense = 0
    certificate_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id, revision=revision, cache_dir=arguments.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        state = model.state_dict()
        model_2d = 0
        model_dense = 0

        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            model_2d += 1
            role = matrix_role(tensor_name)
            model_dense += int(role == "dense_projection")
            try:
                floating = tensor.detach().cpu().contiguous().numpy().astype(
                    np.float32, copy=False
                )
                q4 = symmetric_row_quantize(floating, bits=4)
                q4_values = np.ascontiguousarray(q4.values)
                q4_sha = sha256_bytes(q4_values.tobytes())
                expected_sha = expected_checksums.get((model_id, tensor_name))
                missing_checksums += int(expected_sha is None)
                checksum_mismatches += int(
                    expected_sha is not None and expected_sha != q4_sha
                )

                cert_started = time.perf_counter_ns()
                certificates = certify_registered_displacements(
                    q4_values, primes=primes
                )
                cert_elapsed = time.perf_counter_ns() - cert_started
                selected = select_favorable_displacement(certificates)
                for certificate in certificates:
                    try:
                        certificate.validate(q4_values)
                    except Exception:
                        certificate_mismatches += 1
                        raise
                    operator_rows.append(
                        {
                            "model_id": model_id,
                            "revision": revision,
                            "tensor_name": tensor_name,
                            "matrix_role": role,
                            "shape": list(q4_values.shape),
                            "q4_integer_sha256": q4_sha,
                            "operator": certificate.operator,
                            "displacement_sha256": sha256_bytes(
                                np.ascontiguousarray(certificate.displacement).tobytes()
                            ),
                            "displacement_nonzero_scalar_count": int(
                                np.count_nonzero(certificate.displacement)
                            ),
                            "certificate": certificate_dict(
                                certificate.rank_certificate
                            ),
                            "lower_bounds": certificate.lower_bounds,
                            "selected": certificate.operator == selected.operator,
                        }
                    )
                matrix_rows.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "tensor_name": tensor_name,
                        "matrix_role": role,
                        "shape": list(q4_values.shape),
                        "q4_integer_sha256": q4_sha,
                        "exp057_q4_integer_sha256": expected_sha,
                        "q4_checksum_matches_exp057": expected_sha == q4_sha,
                        "operator_count": len(certificates),
                        "all_operator_certificate_elapsed_ns": cert_elapsed,
                        "all_prime_certificate_count": sum(
                            len(item.rank_certificate.prime_certificates)
                            for item in certificates
                        ),
                        "selected_operator": selected.operator,
                        "selected_displacement_rank_lower_bound": selected.rank_lower_bound,
                        "selected_displacement_rank_fraction": (
                            selected.rank_lower_bound / min(q4_values.shape)
                        ),
                        "selected_query_fraction_lower_bound": float(
                            selected.lower_bounds["query_fraction_lower_bound"]
                        ),
                        "selected_storage_fraction_lower_bound": float(
                            selected.lower_bounds["storage_fraction_lower_bound"]
                        ),
                    }
                )
            except Exception:
                if role == "dense_projection":
                    unregistered_dense += 1
                raise

        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "two_dimensional_tensor_count": model_2d,
                "dense_projection_tensor_count": model_dense,
            }
        )
        del state
        del model

    controls, control_failures = control_rows(
        primes=primes, seed=int(config["seed"])
    )
    certificate_mismatches += control_failures
    primary = [
        row
        for row in matrix_rows
        if row["matrix_role"] == config["primary_matrix_role"]
    ]
    if not primary:
        raise RuntimeError("no primary dense projections were registered")
    query_fractions = [
        float(row["selected_query_fraction_lower_bound"]) for row in primary
    ]
    storage_fractions = [
        float(row["selected_storage_fraction_lower_bound"]) for row in primary
    ]
    rank_fractions = [
        float(row["selected_displacement_rank_fraction"]) for row in primary
    ]
    model_p50_query: dict[str, float] = {}
    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        values = [
            float(row["selected_query_fraction_lower_bound"])
            for row in primary
            if row["model_id"] == model_id
        ]
        model_p50_query[model_id] = percentile(values, 0.5)
    ordered_model_values = [
        model_p50_query[str(entry["model_id"])] for entry in config["models"]
    ]
    degradation = max(
        0.0,
        ordered_model_values[-1] / ordered_model_values[0] - 1.0,
    )
    p50_query = percentile(query_fractions, 0.5)
    p90_query = percentile(query_fractions, 0.9)
    p50_storage = percentile(storage_fractions, 0.5)
    p90_storage = percentile(storage_fractions, 0.9)
    gate = config["early_gate"]
    registration_pass = (
        unregistered_dense <= int(gate["maximum_unregistered_dense_projections"])
        and len(primary) == 144
    )
    checksum_pass = (
        checksum_mismatches
        <= int(gate["maximum_q4_checksum_mismatches_against_exp_057"])
        and missing_checksums == 0
    )
    certificate_pass = certificate_mismatches <= int(
        gate["maximum_certificate_or_control_mismatches"]
    )
    query_pass = (
        p50_query <= float(gate["maximum_p50_query_lower_bound_fraction"])
        and p90_query <= float(gate["maximum_p90_query_lower_bound_fraction"])
    )
    storage_pass = (
        p50_storage <= float(gate["maximum_p50_storage_lower_bound_fraction"])
        and p90_storage <= float(gate["maximum_p90_storage_lower_bound_fraction"])
    )
    trend_pass = degradation <= float(gate["maximum_model_size_degradation"])
    survives = all(
        (registration_pass, checksum_pass, certificate_pass, query_pass, storage_pass, trend_pass)
    )
    decision = (
        "PROMOTE_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_TO_CONSTRUCTIVE_GATE"
        if survives
        else str(config["failure_decision"])
    )
    selected_distribution = Counter(
        str(row["selected_operator"]) for row in primary
    )
    best = min(
        primary,
        key=lambda row: (
            float(row["selected_query_fraction_lower_bound"]),
            float(row["selected_storage_fraction_lower_bound"]),
            str(row["model_id"]),
            str(row["tensor_name"]),
        ),
    )
    summary = {
        "experiment": "EXP-059",
        "name": "pinned_real_q4_exact_shift_displacement_rank_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "registration_gate_pass": registration_pass,
            "q4_checksum_gate_pass": checksum_pass,
            "certificate_control_gate_pass": certificate_pass,
            "query_lower_bound_gate_pass": query_pass,
            "storage_lower_bound_gate_pass": storage_pass,
            "model_size_trend_gate_pass": trend_pass,
            "real_q4_exact_shift_displacement_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "favorable r*max(m,n) query products and r*(m+n) generator "
                "scalars; transforms, boundary terms, metadata, bitwidth growth "
                "and operator-search runtime excluded"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "two_dimensional_tensor_count": len(matrix_rows),
            "dense_projection_matrix_count": len(primary),
            "registered_operator_count": len(REGISTERED_OPERATORS),
            "operator_certificate_row_count": len(operator_rows),
            "q4_checksum_mismatches_against_exp_057": checksum_mismatches,
            "missing_exp_057_q4_checksum_count": missing_checksums,
            "unregistered_dense_projection_count": unregistered_dense,
            "certificate_or_control_mismatches": certificate_mismatches,
            "control_row_count": len(controls),
            "p50_selected_displacement_rank_fraction": percentile(rank_fractions, 0.5),
            "p90_selected_displacement_rank_fraction": percentile(rank_fractions, 0.9),
            "p50_query_lower_bound_fraction": p50_query,
            "p90_query_lower_bound_fraction": p90_query,
            "p50_storage_lower_bound_fraction": p50_storage,
            "p90_storage_lower_bound_fraction": p90_storage,
            "best_real_matrix_query_lower_bound_fraction": float(
                best["selected_query_fraction_lower_bound"]
            ),
            "best_real_matrix_storage_lower_bound_fraction": float(
                best["selected_storage_fraction_lower_bound"]
            ),
            "best_real_matrix_model": best["model_id"],
            "best_real_matrix_tensor": best["tensor_name"],
            "best_real_matrix_operator": best["selected_operator"],
            "selected_operator_distribution": dict(sorted(selected_distribution.items())),
            "model_p50_query_lower_bound_fraction": model_p50_query,
            "model_size_degradation_fraction": degradation,
            "total_all_operator_certificate_elapsed_ns": sum(
                int(row["all_operator_certificate_elapsed_ns"]) for row in matrix_rows
            ),
            "total_prime_certificate_count": sum(
                int(row["all_prime_certificate_count"]) for row in matrix_rows
            ),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "UNVERIFIED": [
            "Q4 model output preservation",
            "constructive exact displacement generators",
            "exact FFT or NTT transform kernels",
            "actual Transformer operation replacement",
            "70B or 405B displacement rank",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "q4_output_preservation": "NOT TESTED",
            "constructive_generator_execution": "NOT TESTED",
            "real_transformer_operation_replacement": False,
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp057_summary_sha256": sha256_file(ROOT / "results/exp_057/summary.json"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/matrix_rows.jsonl", matrix_rows)
    dump_rows(output / "raw/operator_rows.jsonl", operator_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(
        output / "artifacts/environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )
    (output / "artifacts/contract.txt").write_text(
        "Phase C observation only. 405B, 8 GiB, target hardware and actual "
        "Transformer operation replacement are NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
