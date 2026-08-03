#!/usr/bin/env python3
"""Run EXP-061 pinned causal exact activation-sparsity Gate."""

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

from vortex_runtime.activation_sparsity import (
    ActivationCallAccounting,
    ActivationSparsityRecorder,
    account_activation_call,
    exact_zero_skipped_dot,
    weighted_percentile,
)

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tokens(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(token)) for token in tokens).encode("utf-8")
    ).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_snapshot(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    allow_patterns: Sequence[str],
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
            allow_patterns=list(allow_patterns),
        )
    )


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


def expected_projection_parameter_names(recorder: ActivationSparsityRecorder) -> set[str]:
    return {
        f"{alias}.weight" if alias else "weight"
        for registration in recorder.registrations
        for alias in registration.aliases
    }


def unregistered_projection_parameters(
    model: Any, recorder: ActivationSparsityRecorder
) -> tuple[str, ...]:
    expected = expected_projection_parameter_names(recorder)
    missing: list[str] = []
    for name, tensor in model.state_dict().items():
        if tensor.ndim != 2:
            continue
        if matrix_role(name) not in {"dense_projection", "output_head"}:
            continue
        if name not in expected:
            missing.append(name)
    return tuple(sorted(missing))


def cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    recorder: ActivationSparsityRecorder | None,
    model_id: str,
    prompt_family: str,
) -> tuple[int, ...]:
    generated: list[int] = []
    with torch.inference_mode():
        if recorder is not None:
            recorder.set_context(
                model_id=model_id,
                prompt_family=prompt_family,
                phase="prefill",
                decode_step=0,
            )
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1)
        generated.append(int(token.item()))
        for generated_index in range(1, max_new_tokens):
            if recorder is not None:
                recorder.set_context(
                    model_id=model_id,
                    prompt_family=prompt_family,
                    phase="first_decode" if generated_index == 1 else "warm_decode",
                    decode_step=generated_index,
                )
            output = model(
                input_ids=token.reshape(1, 1),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
            generated.append(int(token.item()))
    if recorder is not None:
        recorder.set_context(
            model_id=model_id,
            prompt_family=prompt_family,
            phase="inactive",
            decode_step=-1,
        )
    return tuple(generated)


def aggregate_fraction(
    calls: Sequence[ActivationCallAccounting], *, numerator: str
) -> float:
    dense = sum(row.dense_operation_terms for row in calls)
    if dense <= 0:
        raise RuntimeError("activation calls have no dense operation terms")
    return sum(int(getattr(row, numerator)) for row in calls) / dense


def aggregate_query_fraction(calls: Sequence[ActivationCallAccounting]) -> float:
    dense = sum(row.dense_q4_weight_bytes for row in calls)
    return sum(
        row.sparse_q4_weight_bytes + row.activation_metadata_bytes for row in calls
    ) / dense


def control_rows(torch: Any) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0

    all_zero = account_activation_call(
        model_id="control",
        prompt_family="all_zero",
        phase="warm_decode",
        decode_step=2,
        module_name="linear",
        module_aliases=("linear",),
        input_width=32,
        output_width=64,
        vector_count=1,
        exact_zero_count=32,
    )
    passed = all_zero.sparse_operation_terms == 0
    failures += int(not passed)
    rows.append(
        {
            "control": "all_zero_vector",
            "passed": passed,
            "sparse_operation_fraction": all_zero.sparse_operation_fraction,
            "fully_accounted_operation_fraction": all_zero.fully_accounted_operation_fraction,
        }
    )

    relu = torch.relu(torch.tensor([-2.0, -0.5, 0.0, 3.0]))
    passed = int((relu == 0).sum().item()) == 3
    failures += int(not passed)
    rows.append({"control": "relu_exact_zeros", "passed": passed})

    gelu = torch.nn.functional.gelu(torch.tensor([-2.0, -0.5, 0.5, 3.0]))
    passed = int((gelu == 0).sum().item()) == 0
    failures += int(not passed)
    rows.append({"control": "gelu_no_false_zeros", "passed": passed})

    signed_zeros = torch.tensor([0.0, -0.0, 1.0, -1.0])
    passed = int((signed_zeros == 0).sum().item()) == 2
    failures += int(not passed)
    rows.append({"control": "positive_negative_zero", "passed": passed})

    dense, sparse = exact_zero_skipped_dot(
        weights=[1.5, -2.0, 4.0, 0.25],
        values=[3.0, 0.0, -0.0, 8.0],
    )
    passed = dense == sparse == 6.5
    failures += int(not passed)
    rows.append({"control": "zero_skipped_dot", "passed": passed})
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
        "--config", type=Path, default=ROOT / "experiments/exp_061/config.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_061_candidate"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/exp_061_huggingface"
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

    call_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    registration_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    output_token_mismatches = 0
    registration_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        model_snapshot = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        recorder = ActivationSparsityRecorder.from_model(model)
        unregistered = unregistered_projection_parameters(model, recorder)
        registration_mismatches += len(unregistered)
        for item in recorder.registrations:
            registration_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "canonical_name": item.canonical_name,
                    "aliases": list(item.aliases),
                    "input_width": item.input_width,
                    "output_width": item.output_width,
                    "weight_shape": list(item.weight_shape),
                    "weight_sha256": item.weight_sha256,
                }
            )
        recorder.attach()
        model_case_start = len(case_rows)
        model_call_start = len(recorder.calls)

        for prompt in config["held_out_prompts"]:
            family = str(prompt["family"])
            encoded = tokenizer(
                str(prompt["text"]), return_tensors="pt", add_special_tokens=False
            ).input_ids[:, -int(config["max_input_tokens"]):]
            reference = cached_greedy(
                torch=torch,
                model=model,
                input_ids=encoded,
                max_new_tokens=int(config["max_new_tokens"]),
                recorder=None,
                model_id=model_id,
                prompt_family=family,
            )
            call_start = len(recorder.calls)
            observed = cached_greedy(
                torch=torch,
                model=model,
                input_ids=encoded,
                max_new_tokens=int(config["max_new_tokens"]),
                recorder=recorder,
                model_id=model_id,
                prompt_family=family,
            )
            calls = recorder.calls[call_start:]
            mismatch_count = sum(
                int(left != right) for left, right in zip(reference, observed)
            ) + abs(len(reference) - len(observed))
            output_token_mismatches += mismatch_count
            case_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "prompt_family": family,
                    "prompt_token_count": int(encoded.numel()),
                    "generated_token_count": len(observed),
                    "reference_token_sha256": sha256_tokens(reference),
                    "observed_token_sha256": sha256_tokens(observed),
                    "output_token_mismatches": mismatch_count,
                    "call_count": len(calls),
                    "phase_call_distribution": dict(
                        sorted(Counter(row.phase for row in calls).items())
                    ),
                    "aggregate_exact_zero_fraction": (
                        sum(row.exact_zero_count for row in calls)
                        / sum(row.input_scalar_count for row in calls)
                    ),
                    "aggregate_fully_accounted_operation_fraction": aggregate_fraction(
                        calls, numerator="fully_accounted_operation_terms"
                    ),
                    "aggregate_query_byte_fraction": aggregate_query_fraction(calls),
                }
            )

        missing_called = recorder.missing_called_modules()
        registration_mismatches += len(missing_called)
        recorder.detach()
        model_calls = recorder.calls[model_call_start:]
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "registered_projection_count": len(recorder.registrations),
                "unregistered_projection_parameters": list(unregistered),
                "uncalled_registered_projections": list(missing_called),
                "case_count": len(case_rows) - model_case_start,
                "call_count": len(model_calls),
            }
        )
        call_rows.extend(row.as_dict() for row in recorder.calls)
        del model

    controls, control_failures = control_rows(torch)
    registration_mismatches += control_failures
    warm_calls = [
        ActivationCallAccounting(
            model_id=row["model_id"],
            prompt_family=row["prompt_family"],
            phase=row["phase"],
            decode_step=int(row["decode_step"]),
            module_name=row["module_name"],
            module_aliases=tuple(row["module_aliases"]),
            input_width=int(row["input_width"]),
            output_width=int(row["output_width"]),
            vector_count=int(row["vector_count"]),
            input_scalar_count=int(row["input_scalar_count"]),
            exact_zero_count=int(row["exact_zero_count"]),
            nonzero_count=int(row["nonzero_count"]),
            dense_operation_terms=int(row["dense_operation_terms"]),
            sparse_operation_terms=int(row["sparse_operation_terms"]),
            zero_scan_terms=int(row["zero_scan_terms"]),
            fully_accounted_operation_terms=int(
                row["fully_accounted_operation_terms"]
            ),
            dense_q4_weight_bytes=int(row["dense_q4_weight_bytes"]),
            sparse_q4_weight_bytes=int(row["sparse_q4_weight_bytes"]),
            activation_metadata_bytes=int(row["activation_metadata_bytes"]),
        )
        for row in call_rows
        if row["phase"] == "warm_decode"
    ]
    if not warm_calls:
        raise RuntimeError("no warm-decode activation calls were recorded")
    p50_operations = weighted_percentile(
        warm_calls,
        field_name="fully_accounted_operation_fraction",
        probability=0.50,
    )
    p90_operations = weighted_percentile(
        warm_calls,
        field_name="fully_accounted_operation_fraction",
        probability=0.90,
    )
    p50_bytes = weighted_percentile(
        warm_calls, field_name="query_byte_fraction", probability=0.50
    )
    p90_bytes = weighted_percentile(
        warm_calls, field_name="query_byte_fraction", probability=0.90
    )
    p50_zero = weighted_percentile(
        warm_calls, field_name="exact_zero_fraction", probability=0.50
    )
    p90_zero = weighted_percentile(
        warm_calls, field_name="exact_zero_fraction", probability=0.90
    )
    model_p50_operations: dict[str, float] = {}
    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        model_p50_operations[model_id] = weighted_percentile(
            [row for row in warm_calls if row.model_id == model_id],
            field_name="fully_accounted_operation_fraction",
            probability=0.50,
        )
    ordered = [model_p50_operations[str(entry["model_id"])] for entry in config["models"]]
    largest_model_degradation = max(0.0, ordered[-1] / ordered[0] - 1.0)
    family_count = len({str(row["prompt_family"]) for row in case_rows})
    gate = config["early_gate"]
    output_pass = output_token_mismatches <= int(
        gate["maximum_output_token_mismatches"]
    )
    registration_pass = registration_mismatches <= int(
        gate["maximum_hook_registration_control_mismatches"]
    )
    operation_pass = (
        p50_operations
        <= float(gate["maximum_p50_warm_decode_operation_fraction"])
        and p90_operations
        <= float(gate["maximum_p90_warm_decode_operation_fraction"])
    )
    byte_pass = (
        p50_bytes <= float(gate["maximum_p50_warm_decode_query_byte_fraction"])
        and p90_bytes <= float(gate["maximum_p90_warm_decode_query_byte_fraction"])
    )
    family_pass = family_count >= int(gate["required_prompt_family_count"])
    trend_pass = largest_model_degradation <= float(
        gate["maximum_largest_model_degradation"]
    )
    survives = all(
        (output_pass, registration_pass, operation_pass, byte_pass, family_pass, trend_pass)
    )
    decision = (
        "PROMOTE_CAUSAL_EXACT_ACTIVATION_SPARSITY_TO_PHYSICAL_KERNEL_GATE"
        if survives
        else str(config["failure_decision"])
    )
    phase_totals: dict[str, dict[str, float | int]] = {}
    for phase in ("prefill", "first_decode", "warm_decode"):
        rows = [row for row in call_rows if row["phase"] == phase]
        phase_totals[phase] = {
            "call_count": len(rows),
            "input_scalar_count": sum(int(row["input_scalar_count"]) for row in rows),
            "exact_zero_count": sum(int(row["exact_zero_count"]) for row in rows),
            "aggregate_exact_zero_fraction": (
                sum(int(row["exact_zero_count"]) for row in rows)
                / sum(int(row["input_scalar_count"]) for row in rows)
            ),
            "aggregate_fully_accounted_operation_fraction": (
                sum(int(row["fully_accounted_operation_terms"]) for row in rows)
                / sum(int(row["dense_operation_terms"]) for row in rows)
            ),
            "aggregate_query_byte_fraction": (
                sum(
                    int(row["sparse_q4_weight_bytes"])
                    + int(row["activation_metadata_bytes"])
                    for row in rows
                )
                / sum(int(row["dense_q4_weight_bytes"]) for row in rows)
            ),
        }
    summary = {
        "experiment": "EXP-061",
        "name": "pinned_causal_exact_activation_sparsity_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "output_token_gate_pass": output_pass,
            "hook_registration_control_gate_pass": registration_pass,
            "warm_decode_operation_gate_pass": operation_pass,
            "warm_decode_query_byte_gate_pass": byte_pass,
            "prompt_family_gate_pass": family_pass,
            "model_size_trend_gate_pass": trend_pass,
            "causal_exact_activation_sparsity_survives_gate": survives,
            "decision": decision,
            "accounting_scope": (
                "exact +/- zero projection-input coordinates only; Q4 skipped "
                "weight columns, nonzero-coordinate indexes, vector pointers, "
                "and a full activation zero scan charged"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "case_count": len(case_rows),
            "prompt_family_count": family_count,
            "generated_tokens_per_case": int(config["max_new_tokens"]),
            "registration_row_count": len(registration_rows),
            "activation_call_count": len(call_rows),
            "warm_decode_call_count": len(warm_calls),
            "output_token_mismatches": output_token_mismatches,
            "hook_registration_or_control_mismatches": registration_mismatches,
            "p50_warm_decode_exact_zero_fraction": p50_zero,
            "p90_warm_decode_exact_zero_fraction": p90_zero,
            "maximum_warm_decode_exact_zero_fraction": max(
                row.exact_zero_fraction for row in warm_calls
            ),
            "p50_warm_decode_operation_fraction": p50_operations,
            "p90_warm_decode_operation_fraction": p90_operations,
            "p50_warm_decode_query_byte_fraction": p50_bytes,
            "p90_warm_decode_query_byte_fraction": p90_bytes,
            "model_p50_warm_decode_operation_fraction": model_p50_operations,
            "largest_model_degradation_fraction": largest_model_degradation,
            "phase_totals": phase_totals,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "UNVERIFIED": [
            "physical activation-sparse projection kernels",
            "405B activation sparsity",
            "actual Transformer operation replacement",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "physical_sparse_kernel": "NOT TESTED",
            "405b_activation_statistics": "NOT TESTED",
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
    dump_rows(output / "raw/call_rows.jsonl", call_rows)
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
        "Phase C observation only. Physical sparse kernels, 405B activation "
        "statistics, 405B execution, 8 GiB, and target hardware are NOT TESTED.\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
