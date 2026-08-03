#!/usr/bin/env python3
"""Run EXP-062 pinned causal exact attention-probability sparsity Gate."""

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
from vortex_runtime.activation_sparsity import ActivationSparsityRecorder
from vortex_runtime.attention_probability_sparsity import (
    AttentionProbabilityAccounting,
    account_attention_probabilities,
    combine_whole_model_accounting,
    zero_skipped_value_accumulation,
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


def attention_layout(model: Any) -> tuple[tuple[str, ...], int, int]:
    layers = tuple(str(value) for value in getattr(model.config, "attention_layers", ()))
    layer_count = int(
        getattr(model.config, "num_layers", getattr(model.config, "num_hidden_layers", 0))
    )
    if not layers:
        layers = tuple("global" for _ in range(layer_count))
    if len(layers) != layer_count or not layers:
        raise RuntimeError(
            f"attention layer registration mismatch: {len(layers)} != {layer_count}"
        )
    if any(value not in {"global", "local"} for value in layers):
        raise RuntimeError(f"unsupported attention layer types: {layers}")
    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    head_count = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads", 0))
    )
    if hidden_size <= 0 or head_count <= 0 or hidden_size % head_count:
        raise RuntimeError("invalid hidden/head dimensions")
    return layers, int(getattr(model.config, "window_size", 0)), hidden_size // head_count


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
    head_dimension: int,
    row_sum_tolerance: float,
) -> tuple[tuple[int, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    generated: list[int] = []
    forward_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
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
            linear_start = len(recorder.calls)
            if step == 0:
                output = model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=token.reshape(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
            linear_calls = recorder.calls[linear_start:]
            attentions = output.attentions
            if attentions is None or len(attentions) != len(attention_layers):
                raise RuntimeError(
                    f"returned attention population mismatch: "
                    f"{0 if attentions is None else len(attentions)} != {len(attention_layers)}"
                )
            step_attention: list[AttentionProbabilityAccounting] = []
            for layer_index, (kind, probabilities) in enumerate(
                zip(attention_layers, attentions)
            ):
                query_length = int(probabilities.shape[-2])
                key_length = int(probabilities.shape[-1])
                past_length = key_length - query_length
                row = account_attention_probabilities(
                    probabilities,
                    model_id=model_id,
                    prompt_family=prompt_family,
                    phase=phase,
                    decode_step=step,
                    layer_index=layer_index,
                    head_dimension=head_dimension,
                    past_length=past_length,
                    attention_kind=kind,
                    local_window_size=(local_window_size if kind == "local" else None),
                    row_sum_tolerance=row_sum_tolerance,
                )
                step_attention.append(row)
                attention_rows.append(row.as_dict())

            linear_operations = sum(
                call.dense_operation_terms for call in linear_calls
            )
            linear_bytes = sum(call.dense_q4_weight_bytes for call in linear_calls)
            combined = combine_whole_model_accounting(
                linear_dense_operations=linear_operations,
                linear_dense_q4_bytes=linear_bytes,
                attention_rows=step_attention,
            )
            eligible = sum(row.eligible_probability_count for row in step_attention)
            exact_zeros = sum(row.exact_nonmask_zero_count for row in step_attention)
            forward_rows.append(
                {
                    "model_id": model_id,
                    "prompt_family": prompt_family,
                    "phase": phase,
                    "decode_step": step,
                    "linear_call_count": len(linear_calls),
                    "attention_layer_count": len(step_attention),
                    "eligible_probability_count": eligible,
                    "exact_nonmask_zero_count": exact_zeros,
                    "exact_nonmask_zero_fraction": exact_zeros / eligible,
                    "attention_operation_fraction": (
                        sum(row.sparse_attention_operation_terms for row in step_attention)
                        / sum(row.dense_attention_operation_terms for row in step_attention)
                    ),
                    "attention_query_byte_fraction": (
                        sum(
                            row.sparse_attention_bytes + row.metadata_bytes
                            for row in step_attention
                        )
                        / sum(row.dense_attention_bytes for row in step_attention)
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
    return tuple(generated), forward_rows, attention_rows


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
                    output_attentions=False,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=token.reshape(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
            generated.append(int(token.item()))
    return tuple(generated)


def control_rows(torch: Any, *, tolerance: float) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0

    probabilities = torch.softmax(
        torch.tensor([[[[0.0, -200.0, -300.0]]]], dtype=torch.float32),
        dim=-1,
    )
    underflow = account_attention_probabilities(
        probabilities,
        model_id="control",
        prompt_family="underflow",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_dimension=8,
        past_length=2,
        row_sum_tolerance=tolerance,
    )
    passed = underflow.exact_nonmask_zero_count >= 1
    failures += int(not passed)
    rows.append(
        {
            "control": "unmasked_underflow",
            "passed": passed,
            "exact_zero_count": underflow.exact_nonmask_zero_count,
        }
    )

    moderate = account_attention_probabilities(
        torch.softmax(
            torch.tensor([[[[0.0, -2.0, 1.0]]]], dtype=torch.float32), dim=-1
        ),
        model_id="control",
        prompt_family="moderate",
        phase="warm_decode",
        decode_step=2,
        layer_index=0,
        head_dimension=8,
        past_length=2,
        row_sum_tolerance=tolerance,
    )
    passed = moderate.exact_nonmask_zero_count == 0
    failures += int(not passed)
    rows.append({"control": "moderate_no_false_zero", "passed": passed})

    local = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    for query in range(4):
        start = max(0, query - 1)
        local[0, 0, query, start : query + 1] = 1.0 / (query - start + 1)
    local_row = account_attention_probabilities(
        local,
        model_id="control",
        prompt_family="local_mask",
        phase="prefill",
        decode_step=0,
        layer_index=0,
        head_dimension=8,
        past_length=0,
        attention_kind="local",
        local_window_size=2,
        row_sum_tolerance=tolerance,
    )
    passed = local_row.exact_nonmask_zero_count == 0 and local_row.structural_masked_probability_count == 9
    failures += int(not passed)
    rows.append({"control": "local_mask_exclusion", "passed": passed})

    dense, sparse = zero_skipped_value_accumulation(
        probabilities=[0.5, 0.0, -0.0, 0.5],
        values=[[1.0, 2.0], [8.0, 9.0], [5.0, 6.0], [3.0, 4.0]],
    )
    passed = dense == sparse == (2.0, 3.0)
    failures += int(not passed)
    rows.append({"control": "value_accumulation_equality", "passed": passed})
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_062/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_062_candidate"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/exp_062_huggingface"
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
    attention_rows: list[dict[str, Any]] = []
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
            snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        layers, window_size, head_dimension = attention_layout(model)
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
        for index, kind in enumerate(layers):
            registration_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "kind": "attention",
                    "layer_index": index,
                    "attention_kind": kind,
                    "local_window_size": window_size if kind == "local" else None,
                    "head_dimension": head_dimension,
                }
            )

        recorder.attach()
        model_forward_start = len(forward_rows)
        model_attention_start = len(attention_rows)
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
            observed, case_forwards, case_attentions = observed_cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                max_new_tokens=int(config["max_new_tokens"]),
                recorder=recorder,
                model_id=model_id,
                prompt_family=family,
                attention_layers=layers,
                local_window_size=window_size,
                head_dimension=head_dimension,
                row_sum_tolerance=float(config["row_sum_tolerance"]),
            )
            token_mismatches = sum(
                int(left != right) for left, right in zip(reference, observed)
            ) + abs(len(reference) - len(observed))
            mismatch_count += token_mismatches
            forward_rows.extend(case_forwards)
            attention_rows.extend(case_attentions)
            eligible = sum(
                int(row["eligible_probability_count"]) for row in case_attentions
            )
            exact_zeros = sum(
                int(row["exact_nonmask_zero_count"]) for row in case_attentions
            )
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
                    "attention_row_count": len(case_attentions),
                    "eligible_probability_count": eligible,
                    "exact_nonmask_zero_count": exact_zeros,
                    "exact_nonmask_zero_fraction": exact_zeros / eligible,
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
                "attention_layer_count": len(layers),
                "attention_layer_distribution": dict(sorted(Counter(layers).items())),
                "unregistered_projection_parameters": list(unregistered),
                "uncalled_registered_projections": list(missing_called),
                "forward_row_count": len(forward_rows) - model_forward_start,
                "attention_row_count": len(attention_rows) - model_attention_start,
            }
        )
        del model

    controls, control_failures = control_rows(
        torch, tolerance=float(config["row_sum_tolerance"])
    )
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
    p50_zero = weighted_percentile(
        warm,
        value_key="exact_nonmask_zero_fraction",
        weight_key="eligible_probability_count",
        probability=0.50,
    )
    p90_zero = weighted_percentile(
        warm,
        value_key="exact_nonmask_zero_fraction",
        weight_key="eligible_probability_count",
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
        "PROMOTE_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_TO_KERNEL_GATE"
        if survives
        else str(config["failure_decision"])
    )
    phase_totals: dict[str, dict[str, int | float]] = {}
    for phase in ("prefill", "first_decode", "warm_decode"):
        phase_rows = [row for row in forward_rows if row["phase"] == phase]
        eligible = sum(int(row["eligible_probability_count"]) for row in phase_rows)
        zeros = sum(int(row["exact_nonmask_zero_count"]) for row in phase_rows)
        phase_totals[phase] = {
            "forward_count": len(phase_rows),
            "eligible_probability_count": eligible,
            "exact_nonmask_zero_count": zeros,
            "aggregate_exact_nonmask_zero_fraction": zeros / eligible,
            "aggregate_whole_model_operation_fraction": (
                sum(int(row["sparse_whole_model_operations"]) for row in phase_rows)
                / sum(int(row["dense_whole_model_operations"]) for row in phase_rows)
            ),
            "aggregate_whole_model_query_byte_fraction": (
                sum(int(row["sparse_whole_model_bytes"]) for row in phase_rows)
                / sum(int(row["dense_whole_model_bytes"]) for row in phase_rows)
            ),
        }

    summary = {
        "experiment": "EXP-062",
        "name": "pinned_causal_exact_nonmask_attention_probability_sparsity_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "token_registration_control_gate_pass": correctness_pass,
            "prompt_family_gate_pass": family_pass,
            "whole_model_operation_gate_pass": operation_pass,
            "whole_model_query_byte_gate_pass": byte_pass,
            "model_size_trend_gate_pass": trend_pass,
            "attention_probability_sparsity_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "exact non-mask post-softmax zeros only; causal/local masks excluded; "
                "QK, softmax, Value zero scan, indexes, unchanged Linear work, and "
                "logical Q4 Linear bytes charged"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "case_count": len(case_rows),
            "prompt_family_count": family_count,
            "generated_tokens_per_case": int(config["max_new_tokens"]),
            "forward_row_count": len(forward_rows),
            "attention_row_count": len(attention_rows),
            "registration_row_count": len(registration_rows),
            "token_registration_or_control_mismatches": mismatch_count,
            "warm_forward_count": len(warm),
            "p50_warm_exact_nonmask_zero_fraction": p50_zero,
            "p90_warm_exact_nonmask_zero_fraction": p90_zero,
            "maximum_attention_row_exact_nonmask_zero_fraction": max(
                float(row["exact_nonmask_zero_fraction"]) for row in attention_rows
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
            "physical attention-sparse kernels",
            "405B attention probability sparsity",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "physical_attention_sparse_kernel": "NOT TESTED",
            "405b_attention_statistics": "NOT TESTED",
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
    dump_rows(output / "raw/attention_rows.jsonl", attention_rows)
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
        },
    )
    (output / "artifacts/contract.txt").write_text(
        "Phase C observation only. Physical attention-sparse kernels, 405B "
        "statistics, 405B execution, 8 GiB, and target hardware are NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
