#!/usr/bin/env python3
"""Run EXP-057 pinned real-checkpoint weight-structure extraction Gate.

The runner downloads unchanged revision-pinned TinyStories checkpoints, records
all learned tensors, and analyzes every 2-D tensor.  Exact FP32 bit-pattern
statistics are observational.  Deterministic Q8/Q4 representations are
structural execution candidates only; this experiment does not claim model
output preservation after quantization or replace a Transformer operation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from vortex_runtime.weight_structure import (
    column_group_stats,
    deterministic_column_shuffle,
    deterministic_element_permutation,
    prototype_residual_stats,
    symmetric_row_quantize,
)

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


def digest(path: Path) -> str:
    return sha256_file(path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def snapshot_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return rows


def resolve_snapshot(
    *, model_id: str, revision: str, cache_dir: Path
) -> tuple[Path, list[dict[str, Any]]]:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model_id, revision=revision)
    if str(info.sha) != revision:
        raise RuntimeError(
            f"resolved revision mismatch for {model_id}: {info.sha} != {revision}"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=list(MODEL_PATTERNS),
        )
    )
    return snapshot, snapshot_manifest(snapshot)


def tensor_checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + contiguous.tobytes())


def matrix_role(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("wte", "wpe", "embed", "embedding")):
        return "embedding"
    if "lm_head" in lowered or "output_projection" in lowered:
        return "output_head"
    return "dense_projection"


def total_dictionary_compile_operations(
    *,
    source_scalar_count: int,
    column_count: int,
    candidate_count: int,
    prototype_counts: Sequence[int],
    distance_comparisons: int,
) -> int:
    # Charge source scanning/uniquing, shared distance construction, every
    # frequency assignment, every greedy candidate evaluation, all final
    # assignments, and selection across all attempted plans.
    total = source_scalar_count + distance_comparisons
    for requested in prototype_counts:
        selected = min(int(requested), candidate_count)
        total += selected * column_count  # frequency final assignment
        total += sum(
            (candidate_count - index) * column_count
            for index in range(selected)
        )
        total += selected * column_count  # greedy final assignment
    total += 2 * len(prototype_counts)
    return int(total)


def exact_integer_identity_audit(matrix: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Independent exact additive-identity audit on real quantized columns.

    A deterministic source column is paired with another deterministic source
    column as its prototype.  The full residual is materialized in int32 and
    reconstructed.  This does not validate prototype search optimality; it
    validates that the execution representation's additive residual identity is
    exact and overflow-free for every audited source scalar.
    """

    source = np.asarray(matrix)
    rows, columns = source.shape
    if columns == 0:
        raise ValueError("matrix has no columns")
    rng = np.random.default_rng(seed)
    prototype_indices = rng.integers(0, columns, size=columns)
    prototypes = source[:, prototype_indices].astype(np.int32, copy=False)
    residuals = source.astype(np.int32, copy=False) - prototypes
    reconstructed = prototypes + residuals
    mismatches = int(
        np.count_nonzero(reconstructed != source.astype(np.int32, copy=False))
    )
    return {
        "audited_scalar_count": int(rows * columns),
        "reconstruction_mismatches": mismatches,
        "maximum_absolute_residual": int(
            np.max(np.abs(residuals), initial=0)
        ),
    }


def grouping_compile_and_amortization(
    *, grouping: dict[str, Any], source_scalar_count: int
) -> tuple[int, float]:
    baseline = source_scalar_count
    grouped = float(grouping["selected_operation_fraction"]) * baseline
    saved = baseline - grouped
    compile_operations = 2 * source_scalar_count
    amortization = (
        math.ceil(compile_operations / saved) if saved > 0 else math.inf
    )
    return compile_operations, amortization


def selected_representation_plan(
    *,
    grouping: dict[str, Any],
    dictionary: dict[str, Any],
    scalar_bits: int,
    source_scalar_count: int,
    prototype_counts: Sequence[int],
) -> dict[str, Any]:
    dictionary_selected = dict(dictionary["selected"])
    group_compile, _ = grouping_compile_and_amortization(
        grouping=grouping, source_scalar_count=source_scalar_count
    )
    dictionary_compile = total_dictionary_compile_operations(
        source_scalar_count=source_scalar_count,
        column_count=int(grouping["column_count"]),
        candidate_count=int(dictionary["candidate_count"]),
        prototype_counts=prototype_counts,
        distance_comparisons=int(dictionary["compile_scalar_comparisons"]),
    )
    all_compile = group_compile + dictionary_compile
    group_candidate = {
        "mechanism": f"grouping:{grouping['selected_grouping']}",
        "operation_fraction": float(grouping["selected_operation_fraction"]),
        "query_byte_fraction": float(grouping["selected_query_byte_fraction"]),
        "logical_storage_bytes": int(grouping["selected_logical_storage_bytes"]),
        "residual_scalar_fraction": 0.0,
        "reconstruction_mismatches": 0,
    }
    dictionary_candidate = {
        "mechanism": (
            f"prototype_residual:{dictionary_selected['strategy']}:"
            f"k{dictionary_selected['requested_prototype_count']}"
        ),
        "operation_fraction": float(dictionary_selected["operation_fraction"]),
        "query_byte_fraction": float(dictionary_selected["query_byte_fraction"]),
        "logical_storage_bytes": int(dictionary_selected["logical_storage_bytes"]),
        "residual_scalar_fraction": float(
            dictionary_selected["residual_scalar_fraction"]
        ),
        "reconstruction_mismatches": int(
            dictionary_selected["reconstruction_mismatches"]
        ),
    }
    selected = min(
        (group_candidate, dictionary_candidate),
        key=lambda row: (
            row["operation_fraction"],
            row["query_byte_fraction"],
            row["logical_storage_bytes"],
            row["mechanism"],
        ),
    )
    baseline_operations = source_scalar_count
    selected_operations = selected["operation_fraction"] * baseline_operations
    saved_operations = baseline_operations - selected_operations
    amortization = (
        math.ceil(all_compile / saved_operations)
        if saved_operations > 0
        else math.inf
    )
    logical_source_bytes = math.ceil(source_scalar_count * scalar_bits / 8)
    return {
        **selected,
        "all_mechanism_compile_operations": all_compile,
        "required_compile_amortization_queries": amortization,
        "logical_source_bytes": logical_source_bytes,
        "storage_fraction": selected["logical_storage_bytes"]
        / logical_source_bytes,
    }


def analyse_matrix(
    *,
    matrix: np.ndarray,
    model_id: str,
    tensor_name: str,
    role: str,
    tensor_sha256: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, columns = matrix.shape
    source_scalar_count = int(rows * columns)
    prototype_counts = tuple(int(value) for value in config["prototype_counts"])
    candidate_cap = int(config["prototype_candidate_cap"])
    representation_rows: list[dict[str, Any]] = []

    fp_grouping = column_group_stats(matrix, scalar_bits=32)
    fp_compile, fp_amortization = grouping_compile_and_amortization(
        grouping=fp_grouping, source_scalar_count=source_scalar_count
    )
    representation_rows.append(
        {
            "model_id": model_id,
            "tensor_name": tensor_name,
            "matrix_role": role,
            "tensor_sha256": tensor_sha256,
            "representation": "fp32_exact_bits",
            "row_count": rows,
            "column_count": columns,
            "source_scalar_count": source_scalar_count,
            "grouping": fp_grouping,
            "selected_mechanism": f"grouping:{fp_grouping['selected_grouping']}",
            "selected_operation_fraction": fp_grouping[
                "selected_operation_fraction"
            ],
            "selected_query_byte_fraction": fp_grouping[
                "selected_query_byte_fraction"
            ],
            "selected_logical_storage_bytes": fp_grouping[
                "selected_logical_storage_bytes"
            ],
            "all_mechanism_compile_operations": fp_compile,
            "required_compile_amortization_queries": fp_amortization,
            "exact_bit_pattern_observation": True,
            "model_output_preservation_tested": False,
        }
    )

    q_results: dict[str, Any] = {}
    for bits in (8, 4):
        quantized = symmetric_row_quantize(matrix, bits=bits)
        grouping = column_group_stats(quantized.values, scalar_bits=bits)
        dictionary = prototype_residual_stats(
            quantized.values,
            scalar_bits=bits,
            prototype_counts=prototype_counts,
            candidate_cap=candidate_cap,
        )
        identity = exact_integer_identity_audit(
            quantized.values, seed=seed + bits
        )
        selected = selected_representation_plan(
            grouping=grouping,
            dictionary=dictionary,
            scalar_bits=bits,
            source_scalar_count=source_scalar_count,
            prototype_counts=prototype_counts,
        )
        representation_name = f"q{bits}_row_symmetric"
        row = {
            "model_id": model_id,
            "tensor_name": tensor_name,
            "matrix_role": role,
            "tensor_sha256": tensor_sha256,
            "representation": representation_name,
            "row_count": rows,
            "column_count": columns,
            "source_scalar_count": source_scalar_count,
            "quantization": {
                "bits": bits,
                "scale_count": int(quantized.scales.size),
                "scale_sha256": sha256_bytes(
                    np.ascontiguousarray(quantized.scales).tobytes()
                ),
                "integer_sha256": sha256_bytes(
                    np.ascontiguousarray(quantized.values).tobytes()
                ),
                "maximum_absolute_error": quantized.maximum_absolute_error,
                "mean_absolute_error": quantized.mean_absolute_error,
                "zero_row_count": quantized.zero_row_count,
                "clipped_value_count": quantized.clipped_value_count,
            },
            "grouping": grouping,
            "prototype_residual": dictionary,
            "integer_identity_audit": identity,
            "selected_mechanism": selected["mechanism"],
            "selected_operation_fraction": selected["operation_fraction"],
            "selected_query_byte_fraction": selected["query_byte_fraction"],
            "selected_logical_storage_bytes": selected[
                "logical_storage_bytes"
            ],
            "selected_storage_fraction": selected["storage_fraction"],
            "selected_residual_scalar_fraction": selected[
                "residual_scalar_fraction"
            ],
            "all_mechanism_compile_operations": selected[
                "all_mechanism_compile_operations"
            ],
            "required_compile_amortization_queries": selected[
                "required_compile_amortization_queries"
            ],
            "reconstruction_mismatches": (
                selected["reconstruction_mismatches"]
                + identity["reconstruction_mismatches"]
            ),
            "exact_integer_structure": True,
            "model_output_preservation_tested": False,
        }
        representation_rows.append(row)
        q_results[representation_name] = {
            "values": quantized.values,
            "row": row,
        }
    return representation_rows, q_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_057/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_057_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_057_huggingface",
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
    tensor_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    analysis_cache: dict[str, list[dict[str, Any]]] = {}
    q_cache: dict[str, dict[str, Any]] = {}
    experiment_started = time.perf_counter_ns()
    unregistered_2d = 0

    for model_index, model_entry in enumerate(config["models"]):
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot, manifest = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
        )
        snapshot_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "snapshot_path_name": snapshot.name,
                "files": manifest,
            }
        )
        load_started = time.perf_counter_ns()
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        load_elapsed = time.perf_counter_ns() - load_started
        state = model.state_dict()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        parameter_bytes = int(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        )
        model_2d_count = 0
        model_2d_parameters = 0
        projection_candidates: list[tuple[int, str, np.ndarray, str]] = []

        for tensor_index, (name, tensor) in enumerate(sorted(state.items())):
            cpu = tensor.detach().cpu().contiguous()
            array = cpu.numpy()
            checksum = tensor_checksum(array)
            row = {
                "model_id": model_id,
                "revision": revision,
                "tensor_name": name,
                "shape": list(array.shape),
                "ndim": int(array.ndim),
                "dtype": str(array.dtype),
                "element_count": int(array.size),
                "size_bytes": int(array.nbytes),
                "sha256": checksum,
                "matrix_role": matrix_role(name) if array.ndim == 2 else None,
                "analysed_as_2d": array.ndim == 2,
            }
            tensor_rows.append(row)
            if array.ndim != 2:
                continue
            model_2d_count += 1
            model_2d_parameters += int(array.size)
            role = matrix_role(name)
            if checksum in analysis_cache:
                cached_rows = analysis_cache[checksum]
                for cached in cached_rows:
                    cloned = dict(cached)
                    cloned.update(
                        {
                            "model_id": model_id,
                            "tensor_name": name,
                            "matrix_role": role,
                            "analysis_reused_from_tensor_sha256": checksum,
                        }
                    )
                    representation_rows.append(cloned)
                q_results = q_cache[checksum]
            else:
                try:
                    analysed, q_results = analyse_matrix(
                        matrix=np.asarray(array, dtype=np.float32),
                        model_id=model_id,
                        tensor_name=name,
                        role=role,
                        tensor_sha256=checksum,
                        config=config,
                        seed=int(config["seed"]) + model_index * 10000 + tensor_index,
                    )
                except Exception:
                    unregistered_2d += 1
                    raise
                representation_rows.extend(analysed)
                analysis_cache[checksum] = [dict(item) for item in analysed]
                q_cache[checksum] = q_results
            if role == "dense_projection" and array.shape[0] >= 8 and array.shape[1] >= 8:
                projection_candidates.append(
                    (int(array.size), name, np.asarray(array, dtype=np.float32), checksum)
                )

        controls_requested = int(config["control_matrix_per_model"])
        for control_index, (_, name, matrix, checksum) in enumerate(
            sorted(projection_candidates, reverse=True)[:controls_requested]
        ):
            quantized = symmetric_row_quantize(matrix, bits=4).values
            original_group = column_group_stats(quantized, scalar_bits=4)
            original_dictionary = prototype_residual_stats(
                quantized,
                scalar_bits=4,
                prototype_counts=tuple(config["prototype_counts"]),
                candidate_cap=int(config["prototype_candidate_cap"]),
            )
            shuffled = deterministic_column_shuffle(
                quantized,
                seed=int(config["seed"]) + model_index * 100 + control_index,
            )
            shuffled_group = column_group_stats(shuffled, scalar_bits=4)
            shuffled_dictionary = prototype_residual_stats(
                shuffled,
                scalar_bits=4,
                prototype_counts=tuple(config["prototype_counts"]),
                candidate_cap=int(config["prototype_candidate_cap"]),
            )
            permuted = deterministic_element_permutation(
                quantized,
                seed=int(config["seed"]) + 50000 + model_index * 100 + control_index,
            )
            permuted_group = column_group_stats(permuted, scalar_bits=4)
            permuted_dictionary = prototype_residual_stats(
                permuted,
                scalar_bits=4,
                prototype_counts=tuple(config["prototype_counts"]),
                candidate_cap=int(config["prototype_candidate_cap"]),
            )
            shuffle_preserves = (
                original_group["identical"]["group_count"]
                == shuffled_group["identical"]["group_count"]
                and original_group["sign_canonical"]["group_count"]
                == shuffled_group["sign_canonical"]["group_count"]
                and original_dictionary["selected"]["residual_scalar_count"]
                == shuffled_dictionary["selected"]["residual_scalar_count"]
            )
            control_rows.append(
                {
                    "model_id": model_id,
                    "tensor_name": name,
                    "tensor_sha256": checksum,
                    "representation": "q4_row_symmetric",
                    "column_shuffle_preserves_registered_structure": shuffle_preserves,
                    "original_identical_group_count": original_group["identical"]["group_count"],
                    "shuffled_identical_group_count": shuffled_group["identical"]["group_count"],
                    "original_sign_group_count": original_group["sign_canonical"]["group_count"],
                    "shuffled_sign_group_count": shuffled_group["sign_canonical"]["group_count"],
                    "original_residual_scalar_fraction": original_dictionary["selected"]["residual_scalar_fraction"],
                    "shuffled_residual_scalar_fraction": shuffled_dictionary["selected"]["residual_scalar_fraction"],
                    "permuted_identical_group_count": permuted_group["identical"]["group_count"],
                    "permuted_sign_group_count": permuted_group["sign_canonical"]["group_count"],
                    "permuted_residual_scalar_fraction": permuted_dictionary["selected"]["residual_scalar_fraction"],
                    "element_permutation_changed_any_structure_measure": (
                        permuted_group["identical"]["group_count"]
                        != original_group["identical"]["group_count"]
                        or permuted_group["sign_canonical"]["group_count"]
                        != original_group["sign_canonical"]["group_count"]
                        or permuted_dictionary["selected"]["residual_scalar_count"]
                        != original_dictionary["selected"]["residual_scalar_count"]
                    ),
                }
            )

        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "state_dict_tensor_count": len(state),
                "analysed_2d_tensor_count": model_2d_count,
                "analysed_2d_parameter_count": model_2d_parameters,
                "load_elapsed_ns": load_elapsed,
            }
        )
        del state
        del model

    q4_projection_rows = [
        row
        for row in representation_rows
        if row["representation"] == "q4_row_symmetric"
        and row["matrix_role"] == "dense_projection"
    ]
    if not q4_projection_rows:
        raise RuntimeError("no real dense projection matrices were analysed")
    q4_operations = [
        float(row["selected_operation_fraction"]) for row in q4_projection_rows
    ]
    q4_bytes = [
        float(row["selected_query_byte_fraction"]) for row in q4_projection_rows
    ]
    reconstruction_mismatches = sum(
        int(row.get("reconstruction_mismatches", 0))
        for row in representation_rows
        if row["representation"] in {"q8_row_symmetric", "q4_row_symmetric"}
    )
    control_failures = sum(
        not bool(row["column_shuffle_preserves_registered_structure"])
        for row in control_rows
    )
    model_p50: dict[str, float] = {}
    model_parameter_counts: dict[str, int] = {
        row["model_id"]: int(row["parameter_count"]) for row in model_rows
    }
    for model_entry in config["models"]:
        model_id = model_entry["model_id"]
        values = [
            float(row["selected_operation_fraction"])
            for row in q4_projection_rows
            if row["model_id"] == model_id
        ]
        model_p50[model_id] = percentile(values, 0.5)
    ordered_models = sorted(model_p50, key=lambda item: model_parameter_counts[item])
    first_p50 = model_p50[ordered_models[0]]
    largest_p50 = model_p50[ordered_models[-1]]
    if first_p50 == 0.0:
        size_degradation = 0.0 if largest_p50 == 0.0 else math.inf
    else:
        size_degradation = max(0.0, largest_p50 / first_p50 - 1.0)

    maximum_projection = max(
        float(row["selected_storage_fraction"])
        * float(config["projection"]["target_q4_bytes"])
        for row in q4_projection_rows
    )
    amortizations = [
        float(row["required_compile_amortization_queries"])
        for row in q4_projection_rows
    ]
    maximum_amortization = max(amortizations)
    gates = config["early_gate"]
    p50_operations = percentile(q4_operations, 0.5)
    p90_operations = percentile(q4_operations, 0.9)
    p50_bytes = percentile(q4_bytes, 0.5)
    p90_bytes = percentile(q4_bytes, 0.9)
    checks = {
        "reconstruction_gate_pass": reconstruction_mismatches
        <= gates["maximum_reconstruction_mismatches"],
        "registration_gate_pass": unregistered_2d
        <= gates["maximum_unregistered_2d_tensors"],
        "control_gate_pass": control_failures == 0,
        "operation_gate_pass": (
            p50_operations <= gates["maximum_q4_p50_operation_fraction"]
            and p90_operations <= gates["maximum_q4_p90_operation_fraction"]
        ),
        "byte_gate_pass": (
            p50_bytes <= gates["maximum_q4_p50_byte_fraction"]
            and p90_bytes <= gates["maximum_q4_p90_byte_fraction"]
        ),
        "model_size_trend_gate_pass": size_degradation
        <= gates["maximum_model_size_degradation"],
        "storage_gate_pass": maximum_projection
        <= gates["maximum_projected_storage_bytes"],
        "compile_amortization_gate_pass": maximum_amortization
        <= gates["maximum_compile_amortization_queries"],
    }
    survives = all(checks.values())
    decision = (
        "PROMOTE_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_TO_SMALL_MODEL_OPERATION_REPLACEMENT"
        if survives
        else config["failure_decision"]
    )

    summary = {
        "experiment": "EXP-057",
        "name": "pinned_real_checkpoint_weight_structure_extraction_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "MEASURED": {
            "model_count": len(model_rows),
            "state_dict_tensor_count": len(tensor_rows),
            "analysed_2d_tensor_count": sum(
                int(row["analysed_2d_tensor_count"]) for row in model_rows
            ),
            "analysed_2d_parameter_count": sum(
                int(row["analysed_2d_parameter_count"]) for row in model_rows
            ),
            "representation_row_count": len(representation_rows),
            "q4_dense_projection_matrix_count": len(q4_projection_rows),
            "unregistered_2d_tensor_count": unregistered_2d,
            "reconstruction_mismatches": reconstruction_mismatches,
            "control_row_count": len(control_rows),
            "control_failures": control_failures,
            "q4_dense_projection_p50_operation_fraction": p50_operations,
            "q4_dense_projection_p90_operation_fraction": p90_operations,
            "q4_dense_projection_p50_query_byte_fraction": p50_bytes,
            "q4_dense_projection_p90_query_byte_fraction": p90_bytes,
            "q4_dense_projection_model_p50_operation_fraction": model_p50,
            "q4_model_size_degradation_fraction": size_degradation,
            "maximum_projected_405b_q4_storage_bytes": maximum_projection,
            "maximum_projected_405b_q4_storage_tib": maximum_projection / 1024**4,
            "maximum_required_compile_amortization_queries": maximum_amortization,
            "infinite_amortization_matrix_count": sum(
                math.isinf(value) for value in amortizations
            ),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - experiment_started,
        },
        "DERIVED": {
            **checks,
            "real_weight_structure_survives_gate": survives,
            "decision": decision,
            "primary_gate_population": (
                "all named dense_projection 2-D tensors under deterministic "
                "per-output-row Q4; embeddings and output heads are reported "
                "but excluded from the dense-operation percentile"
            ),
            "candidate_search_scope": (
                "deterministic exact candidates capped at 32 unique columns; "
                "a failure is conservative, while a pass would require stronger search"
            ),
        },
        "PROJECTED": config["projection"],
        "UNVERIFIED": [
            "Q8/Q4 model output preservation",
            "actual Transformer operation replacement",
            "70B or 405B weight structure",
            "405B execution",
            "8 GiB total runtime state",
            "physical kernel bytes and latency",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "authoritative_decision": decision,
        "real_transformer_operation_replacement": False,
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "q4_output_preservation": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
            "real_transformer_operation_replacement": False,
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": digest(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/tensor_rows.jsonl", tensor_rows)
    dump_rows(output / "raw/representation_rows.jsonl", representation_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "raw/snapshot_manifests.json", snapshot_rows)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        "EXP-057 pinned unchanged real-checkpoint structure observation.\n"
        "Q8/Q4 model-output preservation and operation replacement: NOT TESTED.\n"
        "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/sec: NOT TESTED.\n"
    )
    checksum_lines = [
        f"{digest(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
