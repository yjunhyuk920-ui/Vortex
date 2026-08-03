#!/usr/bin/env python3
"""Run EXP-063 pinned causal exact cached-KV equivalence Gate."""

from __future__ import annotations

import argparse
from collections import Counter
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

from experiments.exp_061.run_experiment import (
    MODEL_PATTERNS,
    TOKENIZER_PATTERNS,
    dump,
    dump_rows,
    resolve_snapshot,
    sha256_file,
    sha256_tokens,
    unregistered_projection_parameters,
    write_checksums,
)
from experiments.exp_062.run_experiment import attention_layout
from vortex_runtime.activation_sparsity import ActivationSparsityRecorder
from vortex_runtime.kv_equivalence import (
    KVReuseAccounting,
    account_kv_reuse,
    combine_whole_model_accounting,
    group_exact_kv_pairs,
    group_exact_vectors,
)

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def weighted_percentile(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    weight_key: str,
    probability: float,
) -> float:
    ordered = sorted(
        (float(row[value_key]), int(row[weight_key])) for row in rows
    )
    if not ordered:
        raise RuntimeError("weighted percentile requires rows")
    total = sum(weight for _, weight in ordered)
    target = max(1, math.ceil(probability * total))
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    raise AssertionError("unreachable weighted percentile")


def legacy_cache(past_key_values: Any) -> tuple[Any, ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        value = past_key_values.to_legacy_cache()
    else:
        value = past_key_values
    result = tuple(value)
    if not result:
        raise RuntimeError("empty KV cache")
    return result


def reference_cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
) -> tuple[int, ...]:
    generated: list[int] = []
    past = None
    token = None
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if step == 0:
                output = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=token.reshape(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
            generated.append(int(token.item()))
    return tuple(generated)


def phase_for_step(step: int) -> str:
    if step == 0:
        return "prefill"
    if step == 1:
        return "first_decode"
    return "warm_decode"


def observed_cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    recorder: ActivationSparsityRecorder,
    model_id: str,
    prompt_family: str,
    attention_layers: tuple[str, ...],
    local_window_size: int,
) -> tuple[tuple[int, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    generated: list[int] = []
    forward_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    past = None
    token = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            phase = phase_for_step(step)
            recorder.set_context(
                model_id=model_id,
                prompt_family=prompt_family,
                phase=phase,
                decode_step=step,
            )
            call_start = len(recorder.calls)
            if step == 0:
                output = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=token.reshape(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            linear_calls = recorder.calls[call_start:]
            cache = legacy_cache(output.past_key_values)
            if len(cache) != len(attention_layers):
                raise RuntimeError(
                    f"cache layer mismatch: {len(cache)} != {len(attention_layers)}"
                )
            step_rows: list[KVReuseAccounting] = []
            for layer_index, (kind, layer_cache) in enumerate(
                zip(attention_layers, cache)
            ):
                if len(layer_cache) < 2:
                    raise RuntimeError("KV cache layer has fewer than two tensors")
                keys, values = layer_cache[0], layer_cache[1]
                if keys.shape != values.shape or keys.ndim != 4:
                    raise RuntimeError(
                        f"invalid KV cache shape at layer {layer_index}: "
                        f"{tuple(keys.shape)} vs {tuple(values.shape)}"
                    )
                batch_size, head_count, cache_length, head_dimension = (
                    int(value) for value in keys.shape
                )
                if batch_size != 1:
                    raise RuntimeError("EXP-063 requires batch size one")
                eligible_length = (
                    min(cache_length, local_window_size)
                    if kind == "local"
                    else cache_length
                )
                start = cache_length - eligible_length
                for head_index in range(head_count):
                    key_rows = (
                        keys[0, head_index, start:, :]
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                    )
                    value_rows = (
                        values[0, head_index, start:, :]
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                    )
                    accounting, key_groups, kv_groups = account_kv_reuse(
                        model_id=model_id,
                        prompt_family=prompt_family,
                        phase=phase,
                        decode_step=step,
                        layer_index=layer_index,
                        head_index=head_index,
                        attention_kind=kind,
                        keys=key_rows,
                        values=value_rows,
                    )
                    row = accounting.as_dict()
                    row.update(
                        {
                            "cache_length": cache_length,
                            "eligible_start": start,
                            "key_group_sizes": list(key_groups.group_sizes),
                            "kv_group_sizes": list(kv_groups.group_sizes),
                            "key_group_representatives": list(
                                key_groups.representative_positions
                            ),
                            "kv_group_representatives": list(
                                kv_groups.representative_positions
                            ),
                        }
                    )
                    group_rows.append(row)
                    step_rows.append(accounting)
            linear_operations = sum(
                call.dense_operation_terms for call in linear_calls
            )
            linear_bytes = sum(call.dense_q4_weight_bytes for call in linear_calls)
            combined = combine_whole_model_accounting(
                linear_dense_operations=linear_operations,
                linear_dense_q4_bytes=linear_bytes,
                rows=step_rows,
            )
            eligible_positions = sum(row.eligible_length for row in step_rows)
            duplicate_keys = sum(row.duplicate_key_count for row in step_rows)
            duplicate_kv = sum(row.duplicate_kv_count for row in step_rows)
            forward_rows.append(
                {
                    "model_id": model_id,
                    "prompt_family": prompt_family,
                    "phase": phase,
                    "decode_step": step,
                    "linear_call_count": len(linear_calls),
                    "kv_head_row_count": len(step_rows),
                    "eligible_position_count": eligible_positions,
                    "duplicate_key_count": duplicate_keys,
                    "duplicate_kv_count": duplicate_kv,
                    "key_duplicate_fraction": duplicate_keys / eligible_positions,
                    "kv_duplicate_fraction": duplicate_kv / eligible_positions,
                    "attention_operation_fraction": (
                        sum(row.candidate_attention_operation_terms for row in step_rows)
                        / sum(row.dense_attention_operation_terms for row in step_rows)
                    ),
                    "attention_query_byte_fraction": (
                        sum(
                            row.candidate_cache_bytes + row.metadata_bytes
                            for row in step_rows
                        )
                        / sum(row.dense_cache_bytes for row in step_rows)
                    ),
                    **combined,
                }
            )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
            generated.append(int(token.item()))

    recorder.set_context(
        model_id=model_id,
        prompt_family=prompt_family,
        phase="inactive",
        decode_step=-1,
    )
    return tuple(generated), forward_rows, group_rows


def control_rows() -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0

    keys = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values = np.asarray([[5.0, 6.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    accounting, key_groups, kv_groups = account_kv_reuse(
        model_id="control",
        prompt_family="duplicates",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_index=0,
        attention_kind="global",
        keys=keys,
        values=values,
    )
    passed = key_groups.duplicate_count == 1 and kv_groups.duplicate_count == 1
    failures += int(not passed)
    rows.append(
        {
            "control": "duplicate_key_and_kv",
            "passed": passed,
            "key_duplicates": key_groups.duplicate_count,
            "kv_duplicates": kv_groups.duplicate_count,
            "operation_fraction": accounting.attention_operation_fraction,
        }
    )

    changed_values = values.copy()
    changed_values[1, 0] = np.nextafter(
        changed_values[1, 0], np.float32(np.inf), dtype=np.float32
    )
    passed = (
        group_exact_vectors(keys).unique_count == 2
        and group_exact_kv_pairs(keys, changed_values).unique_count == 3
    )
    failures += int(not passed)
    rows.append({"control": "one_bit_value_breaks_kv_only", "passed": passed})

    signed_zero = np.asarray([[0.0], [-0.0]], dtype=np.float32)
    passed = group_exact_vectors(signed_zero).unique_count == 2
    failures += int(not passed)
    rows.append({"control": "signed_zero_distinct", "passed": passed})
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_063/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_063_candidate"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/exp_063_huggingface"
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    tokenizer_entry = config["tokenizer"]
    tokenizer_snapshot = resolve_snapshot(
        model_id=str(tokenizer_entry["model_id"]),
        revision=str(tokenizer_entry["revision"]),
        cache_dir=arguments.cache_dir,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)

    forward_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    registration_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    mismatch_count = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        attention_layers, window_size, head_dimension = attention_layout(model)
        recorder = ActivationSparsityRecorder.from_model(model)
        unregistered = unregistered_projection_parameters(model, recorder)
        mismatch_count += len(unregistered)
        for registration in recorder.registrations:
            registration_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "kind": "linear",
                    "canonical_name": registration.canonical_name,
                    "aliases": list(registration.aliases),
                    "input_width": registration.input_width,
                    "output_width": registration.output_width,
                    "weight_sha256": registration.weight_sha256,
                }
            )
        for layer_index, kind in enumerate(attention_layers):
            registration_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "kind": "kv_cache",
                    "layer_index": layer_index,
                    "attention_kind": kind,
                    "local_window_size": window_size if kind == "local" else None,
                    "head_dimension": head_dimension,
                }
            )
        recorder.attach()
        model_forward_start = len(forward_rows)
        model_group_start = len(group_rows)

        for prompt in config["held_out_prompts"]:
            family = str(prompt["family"])
            input_ids = tokenizer(
                str(prompt["text"]), return_tensors="pt", add_special_tokens=False
            ).input_ids[:, -int(config["max_input_tokens"]):]
            reference = reference_cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                max_new_tokens=int(config["max_new_tokens"]),
            )
            observed, case_forwards, case_groups = observed_cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                max_new_tokens=int(config["max_new_tokens"]),
                recorder=recorder,
                model_id=model_id,
                prompt_family=family,
                attention_layers=attention_layers,
                local_window_size=window_size,
            )
            token_mismatches = sum(
                int(left != right) for left, right in zip(reference, observed)
            ) + abs(len(reference) - len(observed))
            mismatch_count += token_mismatches
            forward_rows.extend(case_forwards)
            group_rows.extend(case_groups)
            eligible = sum(int(row["eligible_position_count"]) for row in case_forwards)
            key_duplicates = sum(int(row["duplicate_key_count"]) for row in case_forwards)
            kv_duplicates = sum(int(row["duplicate_kv_count"]) for row in case_forwards)
            case_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "prompt_family": family,
                    "prompt_token_count": int(input_ids.numel()),
                    "generated_token_count": len(observed),
                    "reference_token_sha256": sha256_tokens(reference),
                    "observed_token_sha256": sha256_tokens(observed),
                    "token_mismatches": token_mismatches,
                    "forward_count": len(case_forwards),
                    "group_row_count": len(case_groups),
                    "eligible_position_count": eligible,
                    "duplicate_key_count": key_duplicates,
                    "duplicate_kv_count": kv_duplicates,
                    "key_duplicate_fraction": key_duplicates / eligible,
                    "kv_duplicate_fraction": kv_duplicates / eligible,
                }
            )

        missing_called = recorder.missing_called_modules()
        mismatch_count += len(missing_called)
        recorder.detach()
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "linear_registration_count": len(recorder.registrations),
                "kv_layer_count": len(attention_layers),
                "attention_layer_distribution": dict(
                    sorted(Counter(attention_layers).items())
                ),
                "unregistered_projection_parameters": list(unregistered),
                "uncalled_registered_projections": list(missing_called),
                "forward_row_count": len(forward_rows) - model_forward_start,
                "group_row_count": len(group_rows) - model_group_start,
            }
        )
        del model

    controls, control_failures = control_rows()
    mismatch_count += control_failures
    warm = [row for row in forward_rows if row["phase"] == "warm_decode"]
    if not warm:
        raise RuntimeError("no warm-decode forward rows")
    p50_operations = weighted_percentile(
        warm,
        value_key="whole_model_operation_fraction",
        weight_key="dense_whole_model_operations",
        probability=0.50,
    )
    p90_operations = weighted_percentile(
        warm,
        value_key="whole_model_operation_fraction",
        weight_key="dense_whole_model_operations",
        probability=0.90,
    )
    p50_bytes = weighted_percentile(
        warm,
        value_key="whole_model_query_byte_fraction",
        weight_key="dense_whole_model_bytes",
        probability=0.50,
    )
    p90_bytes = weighted_percentile(
        warm,
        value_key="whole_model_query_byte_fraction",
        weight_key="dense_whole_model_bytes",
        probability=0.90,
    )
    p50_key_duplicates = weighted_percentile(
        warm,
        value_key="key_duplicate_fraction",
        weight_key="eligible_position_count",
        probability=0.50,
    )
    p90_key_duplicates = weighted_percentile(
        warm,
        value_key="key_duplicate_fraction",
        weight_key="eligible_position_count",
        probability=0.90,
    )
    p50_kv_duplicates = weighted_percentile(
        warm,
        value_key="kv_duplicate_fraction",
        weight_key="eligible_position_count",
        probability=0.50,
    )
    p90_kv_duplicates = weighted_percentile(
        warm,
        value_key="kv_duplicate_fraction",
        weight_key="eligible_position_count",
        probability=0.90,
    )
    model_p50_operations: dict[str, float] = {}
    for entry in config["models"]:
        model_id = str(entry["model_id"])
        model_p50_operations[model_id] = weighted_percentile(
            [row for row in warm if row["model_id"] == model_id],
            value_key="whole_model_operation_fraction",
            weight_key="dense_whole_model_operations",
            probability=0.50,
        )
    ordered = [model_p50_operations[str(entry["model_id"])] for entry in config["models"]]
    degradation = max(0.0, ordered[-1] / ordered[0] - 1.0)
    family_count = len({row["prompt_family"] for row in case_rows})
    gate = config["early_gate"]
    correctness_pass = mismatch_count <= int(
        gate["maximum_token_registration_control_mismatches"]
    )
    family_pass = family_count >= int(gate["required_prompt_family_count"])
    operation_pass = (
        p50_operations
        <= float(gate["maximum_p50_whole_model_warm_operation_fraction"])
        and p90_operations
        <= float(gate["maximum_p90_whole_model_warm_operation_fraction"])
    )
    byte_pass = (
        p50_bytes
        <= float(gate["maximum_p50_whole_model_warm_query_byte_fraction"])
        and p90_bytes
        <= float(gate["maximum_p90_whole_model_warm_query_byte_fraction"])
    )
    trend_pass = degradation <= float(gate["maximum_largest_model_degradation"])
    survives = all((correctness_pass, family_pass, operation_pass, byte_pass, trend_pass))
    decision = (
        "PROMOTE_CAUSAL_EXACT_KV_EQUIVALENCE_TO_GROUPED_KERNEL_GATE"
        if survives
        else str(config["failure_decision"])
    )
    phase_totals: dict[str, dict[str, int | float]] = {}
    for phase in ("prefill", "first_decode", "warm_decode"):
        rows = [row for row in forward_rows if row["phase"] == phase]
        eligible = sum(int(row["eligible_position_count"]) for row in rows)
        key_duplicates = sum(int(row["duplicate_key_count"]) for row in rows)
        kv_duplicates = sum(int(row["duplicate_kv_count"]) for row in rows)
        phase_totals[phase] = {
            "forward_count": len(rows),
            "eligible_position_count": eligible,
            "duplicate_key_count": key_duplicates,
            "duplicate_kv_count": kv_duplicates,
            "aggregate_key_duplicate_fraction": key_duplicates / eligible,
            "aggregate_kv_duplicate_fraction": kv_duplicates / eligible,
            "aggregate_whole_model_operation_fraction": (
                sum(int(row["candidate_whole_model_operations"]) for row in rows)
                / sum(int(row["dense_whole_model_operations"]) for row in rows)
            ),
            "aggregate_whole_model_query_byte_fraction": (
                sum(int(row["candidate_whole_model_bytes"]) for row in rows)
                / sum(int(row["dense_whole_model_bytes"]) for row in rows)
            ),
        }

    summary = {
        "experiment": "EXP-063",
        "name": "pinned_causal_exact_cached_kv_equivalence_reuse_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "token_registration_control_gate_pass": correctness_pass,
            "prompt_family_gate_pass": family_pass,
            "whole_model_operation_gate_pass": operation_pass,
            "whole_model_query_byte_gate_pass": byte_pass,
            "model_size_trend_gate_pass": trend_pass,
            "cached_kv_equivalence_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "exact dtype/shape/bit-pattern K and KV groups only; incremental "
                "new-vector hash scan, score copies, unchanged softmax and Value "
                "additions, group metadata, representative K/V reads, and all "
                "unchanged Linear work/bytes charged"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "case_count": len(case_rows),
            "prompt_family_count": family_count,
            "generated_tokens_per_case": int(config["max_new_tokens"]),
            "forward_row_count": len(forward_rows),
            "group_row_count": len(group_rows),
            "registration_row_count": len(registration_rows),
            "token_registration_or_control_mismatches": mismatch_count,
            "warm_forward_count": len(warm),
            "p50_warm_key_duplicate_fraction": p50_key_duplicates,
            "p90_warm_key_duplicate_fraction": p90_key_duplicates,
            "p50_warm_kv_duplicate_fraction": p50_kv_duplicates,
            "p90_warm_kv_duplicate_fraction": p90_kv_duplicates,
            "maximum_group_row_key_duplicate_fraction": max(
                float(row["key_duplicate_fraction"]) for row in group_rows
            ),
            "maximum_group_row_kv_duplicate_fraction": max(
                float(row["kv_duplicate_fraction"]) for row in group_rows
            ),
            "p50_whole_model_warm_operation_fraction": p50_operations,
            "p90_whole_model_warm_operation_fraction": p90_operations,
            "p50_whole_model_warm_query_byte_fraction": p50_bytes,
            "p90_whole_model_warm_query_byte_fraction": p90_bytes,
            "model_p50_whole_model_warm_operation_fraction": model_p50_operations,
            "largest_model_degradation_fraction": degradation,
            "phase_totals": phase_totals,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "UNVERIFIED": [
            "physical grouped-attention kernels",
            "405B exact KV equivalence statistics",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "physical_grouped_attention_kernel": "NOT TESTED",
            "405b_kv_equivalence_statistics": "NOT TESTED",
            "real_transformer_operation_replacement": False,
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "tokenizer_id": str(tokenizer_entry["model_id"]),
            "tokenizer_revision": str(tokenizer_entry["revision"]),
        },
    }
    dump_rows(output / "raw/forward_rows.jsonl", forward_rows)
    dump_rows(output / "raw/group_rows.jsonl", group_rows)
    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump_rows(output / "raw/registration_rows.jsonl", registration_rows)
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
        "Phase C observation only. Physical grouped-attention kernels, 405B "
        "KV statistics, 405B execution, 8 GiB and target hardware are NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
