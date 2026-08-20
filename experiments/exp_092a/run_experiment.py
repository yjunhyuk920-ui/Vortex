from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.block_span_correction import (  # noqa: E402
    DEFAULT_PRIMES,
    LLAMA405_HIDDEN_SIZE,
    LLAMA405_INTERMEDIATE_SIZE,
    LLAMA405_LAYER_COUNT,
    analyze_block_span,
    canonical_sha256,
    correction_operation_multiplier,
    deterministic_columns,
    optimistic_sidecar_bytes,
    tensor_sha256,
)
from vortex_runtime.exact_jacobi_fixed_point import (  # noqa: E402
    build_seed,
    stable_argmax_rows,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    config_dict,
    model_tensor_sha256,
)
from vortex_runtime.prefix_state_bisimulation import (  # noqa: E402
    cache_nbytes,
    cache_sha256,
)


@dataclass
class PrefillState:
    case_id: str
    split: str
    family: str
    prompt_sha256: str
    prompt_ids: tuple[int, ...]
    boundary_token: int
    prefix_cache: tuple[Any, ...]
    prefix_cache_sha256: str
    prefix_cache_bytes: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_checksums(output_dir: Path, relative_paths: Iterable[str]) -> None:
    lines = [
        f"{sha256_file(output_dir / relative)}  {relative}"
        for relative in sorted(set(relative_paths))
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def verify_runtime() -> dict[str, Any]:
    import safetensors

    actual = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "cpu_count": os.cpu_count(),
    }
    expected = {
        "torch": PINNED_TORCH,
        "transformers": PINNED_TRANSFORMERS,
        "safetensors": PINNED_SAFETENSORS,
    }
    mismatch = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatch:
        raise RuntimeError(f"runtime pin mismatch: {mismatch}")
    return actual


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-092a-block-span-correction-closure-v1":
        raise RuntimeError("unexpected schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("configuration was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-092A freezes K=128")
    if config["seed_mode"] != "prompt_suffix_cycle_16":
        raise RuntimeError("EXP-092A freezes the EXP-091A selected seed")
    if list(map(int, config["layer_indices"])) != [0, 15, 29]:
        raise RuntimeError("representative complete-layer population changed")
    if list(config["operator_roles"]) != [
        "attention_qkv_input",
        "attention_o_input",
        "mlp_gate_up_input",
        "mlp_down_input",
    ]:
        raise RuntimeError("operator role population changed")
    if tuple(map(int, config["rank_primes"])) != DEFAULT_PRIMES:
        raise RuntimeError("rank-prime population changed")
    if int(config["rank_sketch_columns"]) <= (
        int(config["block_length"])
        + int(config["maximum_extra_directions_per_operator"])
    ):
        raise RuntimeError("rank sketch cannot decide the frozen threshold")
    if float(config["maximum_extra_fraction"]) != (
        int(config["maximum_extra_directions_per_operator"])
        / int(config["block_length"])
    ):
        raise RuntimeError("extra-direction fraction mismatch")
    target = config["target_projection"]
    expected_target = {
        "llama405_layer_count": LLAMA405_LAYER_COUNT,
        "llama405_hidden_size": LLAMA405_HIDDEN_SIZE,
        "llama405_intermediate_size": LLAMA405_INTERMEDIATE_SIZE,
    }
    for key, value in expected_target.items():
        if int(target[key]) != value:
            raise RuntimeError(f"target projection mismatch: {key}")
    cases = list(config["cases"])
    build = [case for case in cases if case["split"] == "build"]
    holdout = [case for case in cases if case["split"] == "holdout"]
    if len(build) < int(config["success"]["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(config["success"]["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case IDs")
    build_hashes = {sha256_bytes(case["prompt"].encode()) for case in build}
    holdout_hashes = {sha256_bytes(case["prompt"].encode()) for case in holdout}
    if build_hashes.intersection(holdout_hashes):
        raise RuntimeError("build and holdout prompts overlap")


def load_model(config: Mapping[str, Any]) -> tuple[Any, Any, int]:
    start = time.perf_counter_ns()
    model = LlamaForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        low_cpu_mem_usage=False,
    ).eval()
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != config["revision"]:
        raise RuntimeError(f"resolved checkpoint mismatch: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], use_fast=True
    )
    return model, tokenizer, time.perf_counter_ns() - start


def legacy_cache(past_key_values: Any) -> tuple[Any, ...]:
    if isinstance(past_key_values, DynamicCache):
        return tuple(past_key_values.to_legacy_cache())
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    raise RuntimeError(f"unsupported cache type: {type(past_key_values).__name__}")


def fresh_cache(prefix_cache: tuple[Any, ...]) -> DynamicCache:
    return DynamicCache.from_legacy_cache(prefix_cache)


def prefill_case(
    model: Any, tokenizer: Any, case: Mapping[str, Any]
) -> PrefillState:
    input_ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"]
    if int(input_ids.shape[1]) < 1:
        raise RuntimeError(f"empty prompt: {case['id']}")
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    boundary = int(
        stable_argmax_rows(output.logits[:, -1, :].detach().to(torch.float32).cpu())[
            0
        ].item()
    )
    prefix = legacy_cache(output.past_key_values)
    return PrefillState(
        case_id=str(case["id"]),
        split=str(case["split"]),
        family=str(case["family"]),
        prompt_sha256=sha256_bytes(case["prompt"].encode("utf-8")),
        prompt_ids=tuple(int(value) for value in input_ids[0].tolist()),
        boundary_token=boundary,
        prefix_cache=prefix,
        prefix_cache_sha256=cache_sha256(prefix),
        prefix_cache_bytes=cache_nbytes(prefix),
    )


def generate_ar_target(
    model: Any, prefill: PrefillState, *, block_length: int
) -> tuple[tuple[int, ...], dict[str, Any]]:
    started_ns = time.perf_counter_ns()
    cache: Any = fresh_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    for _ in range(int(block_length)):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        token = int(
            stable_argmax_rows(
                output.logits[:, -1, :].detach().to(torch.float32).cpu()
            )[0].item()
        )
        tokens.append(token)
        current = token
        cache = output.past_key_values
    return tuple(tokens), {
        "target_started_ns": started_ns,
        "target_calls": int(block_length),
        "target_sha256": canonical_sha256(tokens),
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
    }


def role_modules(layer: Any) -> dict[str, Any]:
    return {
        "attention_qkv_input": layer.self_attn.q_proj,
        "attention_o_input": layer.self_attn.o_proj,
        "mlp_gate_up_input": layer.mlp.gate_proj,
        "mlp_down_input": layer.mlp.down_proj,
    }


def capture_block(
    model: Any,
    prefill: PrefillState,
    block_tokens: Sequence[int],
    *,
    layer_indices: Sequence[int],
    roles: Sequence[str],
) -> tuple[dict[str, torch.Tensor], tuple[int, ...], dict[str, Any]]:
    if len(block_tokens) < 1:
        raise RuntimeError("block token sequence is empty")
    input_tokens = [int(prefill.boundary_token), *map(int, block_tokens[:-1])]
    captures: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(key: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if key in captures:
                raise RuntimeError(f"operator input captured twice: {key}")
            value = args[0]
            if value.ndim != 3 or int(value.shape[0]) != 1:
                raise RuntimeError(f"unexpected operator input shape: {tuple(value.shape)}")
            captures[key] = value[0].detach().contiguous().cpu()

        return hook

    for layer_index in map(int, layer_indices):
        modules = role_modules(model.model.layers[layer_index])
        for role in roles:
            key = f"layer_{layer_index}:{role}"
            handles.append(modules[str(role)].register_forward_pre_hook(make_hook(key)))

    cache = fresh_cache(prefill.prefix_cache)
    start_ns = time.perf_counter_ns()
    try:
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([input_tokens], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    wall_ns = time.perf_counter_ns() - start_ns
    proposals = tuple(
        int(value)
        for value in stable_argmax_rows(output.logits.detach().to(torch.float32).cpu())[
            0
        ].tolist()
    )
    expected = {
        f"layer_{int(layer_index)}:{role}"
        for layer_index in layer_indices
        for role in roles
    }
    if set(captures) != expected:
        raise RuntimeError(
            f"capture key mismatch missing={sorted(expected-set(captures))} "
            f"extra={sorted(set(captures)-expected)}"
        )
    if any(int(value.shape[0]) != len(block_tokens) for value in captures.values()):
        raise RuntimeError("captured state count differs from block length")
    return captures, proposals, {
        "started_ns": start_ns,
        "completed_ns": start_ns + wall_ns,
        "wall_ns": wall_ns,
        "input_sha256": canonical_sha256(input_tokens),
        "proposal_sha256": canonical_sha256(proposals),
        "logits_sha256": tensor_sha256(output.logits.detach().to(torch.float32).cpu()),
        "cache_sha256": cache_sha256(output.past_key_values),
        "cache_bytes": cache_nbytes(output.past_key_values),
    }


def model_free_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    guess = torch.zeros(16, 48, dtype=torch.bfloat16)
    true = torch.zeros(16, 48, dtype=torch.bfloat16)
    for row in range(16):
        guess[row, row] = 1
        true[row, row] = 1
    for row in range(12):
        true[row, 16 + row] = 1
    columns = tuple(range(48))
    injected = analyze_block_span(
        guess, true, columns=columns, primes=tuple(map(int, config["rank_primes"]))
    )
    identity = analyze_block_span(
        guess, guess.clone(), columns=columns, primes=tuple(map(int, config["rank_primes"]))
    )
    threshold = int(config["maximum_extra_directions_per_operator"])
    controls = {
        "identity_extra_rank": identity.certified_extra_rank_lower_bound,
        "injected_extra_rank": injected.certified_extra_rank_lower_bound,
        "threshold_directions": threshold,
        "rank_sketch_columns": int(config["rank_sketch_columns"]),
        "correction_multiplier_at_threshold": correction_operation_multiplier(
            int(config["block_length"]), threshold
        ),
        "sidecar_bytes_at_threshold": optimistic_sidecar_bytes(threshold),
    }
    controls["passed"] = all(
        [
            controls["identity_extra_rank"] == 0,
            controls["injected_extra_rank"] == 12,
            controls["correction_multiplier_at_threshold"] == 1.0625,
            controls["sidecar_bytes_at_threshold"] == 206_438_400,
        ]
    )
    return controls


def checkpoint_identity(model: Any, tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer_payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    return {
        "model_id": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "model_class": type(model).__name__,
        "config_sha256": canonical_sha256(config_dict(model.config)),
        "checkpoint_tensor_sha256": model_tensor_sha256(model),
        "tokenizer": {**tokenizer_payload, "sha256": canonical_sha256(tokenizer_payload)},
    }


def aggregate(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"empty split aggregate: {split}")
    directions = [int(row["certified_extra_rank_lower_bound"]) for row in selected]
    fractions = [float(row["certified_extra_fraction_lower_bound"]) for row in selected]
    return {
        "split": split,
        "report_count": len(selected),
        "extra_rank_min": min(directions),
        "extra_rank_p50": float(np.percentile(directions, 50, method="linear")),
        "extra_rank_p95": float(np.percentile(directions, 95, method="linear")),
        "extra_rank_max": max(directions),
        "extra_fraction_min": min(fractions),
        "extra_fraction_p50": float(np.percentile(fractions, 50, method="linear")),
        "extra_fraction_p95": float(np.percentile(fractions, 95, method="linear")),
        "extra_fraction_max": max(fractions),
        "zero_increment_count": sum(value == 0 for value in directions),
        "guess_full_row_rank_certified_count": sum(
            bool(row["guess_full_row_rank_certified"]) for row in selected
        ),
        "rank_lower_bound_undecidable_count": sum(
            not bool(row["guess_full_row_rank_certified"]) for row in selected
        ),
        "over_threshold_count": sum(value > 8 for value in directions),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)

    model, tokenizer, model_load_ns = load_model(config)
    identity = checkpoint_identity(model, tokenizer, config)
    controls = model_free_controls(config)
    layer_indices = tuple(map(int, config["layer_indices"]))
    roles = tuple(map(str, config["operator_roles"]))
    block_length = int(config["block_length"])
    rank_columns = int(config["rank_sketch_columns"])
    primes = tuple(map(int, config["rank_primes"]))
    threshold = int(config["maximum_extra_directions_per_operator"])

    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    run_start_ns = time.perf_counter_ns()

    for case in config["cases"]:
        prefill = prefill_case(model, tokenizer, case)
        seed, seed_manifest = build_seed(
            str(config["seed_mode"]),
            prompt_ids=prefill.prompt_ids,
            boundary_token=prefill.boundary_token,
            block_length=block_length,
        )
        guess_captures, guess_proposal, guess_meta = capture_block(
            model,
            prefill,
            seed,
            layer_indices=layer_indices,
            roles=roles,
        )
        target_tokens, target_meta = generate_ar_target(
            model, prefill, block_length=block_length
        )
        true_captures, true_proposal, true_meta = capture_block(
            model,
            prefill,
            target_tokens,
            layer_indices=layer_indices,
            roles=roles,
        )
        true_teacher_forcing_exact = tuple(true_proposal) == tuple(target_tokens)
        guess_precedes_target = int(guess_meta["completed_ns"]) <= int(
            target_meta["target_started_ns"]
        )

        case_report_rows: list[dict[str, Any]] = []
        for key in sorted(guess_captures):
            guess_tensor = guess_captures[key]
            true_tensor = true_captures[key]
            width = int(guess_tensor.shape[1])
            columns = deterministic_columns(
                width,
                min(rank_columns, width),
                seed=f"{config['schema']}:{key}",
            )
            report = analyze_block_span(
                guess_tensor,
                true_tensor,
                columns=columns,
                primes=primes,
            )
            layer_text, role = key.split(":", 1)
            layer_index = int(layer_text.split("_", 1)[1])
            row = {
                "case_id": prefill.case_id,
                "split": prefill.split,
                "family": prefill.family,
                "prompt_sha256": prefill.prompt_sha256,
                "layer_index": layer_index,
                "operator_role": role,
                "operator_key": key,
                "threshold_directions": threshold,
                "threshold_passed_by_lower_bound": report.certified_extra_rank_lower_bound
                <= threshold,
                **report.to_dict(),
            }
            rows.append(row)
            case_report_rows.append(row)

        case_rows.append(
            {
                "case_id": prefill.case_id,
                "split": prefill.split,
                "family": prefill.family,
                "prompt_sha256": prefill.prompt_sha256,
                "prompt_token_count": len(prefill.prompt_ids),
                "boundary_token": prefill.boundary_token,
                "prefix_cache_sha256": prefill.prefix_cache_sha256,
                "prefix_cache_bytes": prefill.prefix_cache_bytes,
                "seed_mode": config["seed_mode"],
                "seed_manifest": seed_manifest.to_dict(),
                "seed_sha256": canonical_sha256(seed),
                "guess_proposal_sha256": canonical_sha256(guess_proposal),
                "target_sha256": target_meta["target_sha256"],
                "true_proposal_sha256": canonical_sha256(true_proposal),
                "true_teacher_forcing_exact": true_teacher_forcing_exact,
                "guess_capture_completed_before_ar_target": guess_precedes_target,
                "guess_capture": guess_meta,
                "target_meta": target_meta,
                "true_capture": true_meta,
                "maximum_certified_extra_rank_lower_bound": max(
                    int(row["certified_extra_rank_lower_bound"])
                    for row in case_report_rows
                ),
                "minimum_certified_extra_rank_lower_bound": min(
                    int(row["certified_extra_rank_lower_bound"])
                    for row in case_report_rows
                ),
            }
        )
        print(
            f"{prefill.split} {prefill.case_id}: "
            f"extra-rank-lb min/max="
            f"{case_rows[-1]['minimum_certified_extra_rank_lower_bound']}/"
            f"{case_rows[-1]['maximum_certified_extra_rank_lower_bound']}",
            flush=True,
        )

    experiment_ns = time.perf_counter_ns() - run_start_ns
    build_aggregate = aggregate(rows, "build")
    holdout_aggregate = aggregate(rows, "holdout")

    integrity_failures: list[str] = []
    if not bool(controls["passed"]):
        integrity_failures.append("model_free_controls")
    if not all(bool(row["true_teacher_forcing_exact"]) for row in case_rows):
        integrity_failures.append("teacher_forcing_ar_equality")
    if not all(
        bool(row["guess_capture_completed_before_ar_target"]) for row in case_rows
    ):
        integrity_failures.append("guess_target_order")
    if len({row["prompt_sha256"] for row in case_rows}) != len(case_rows):
        integrity_failures.append("duplicate_prompt_state")
    if identity["resolved_revision"] not in {"", config["revision"]}:
        integrity_failures.append("checkpoint_revision")
    expected_report_count = len(config["cases"]) * len(layer_indices) * len(roles)
    if len(rows) != expected_report_count:
        integrity_failures.append("report_population")

    integrity_passed = len(integrity_failures) <= int(
        config["success"]["maximum_integrity_failures"]
    )
    decisive_rejection = any(
        row["split"] == "holdout"
        and int(row["certified_extra_rank_lower_bound"]) > threshold
        for row in rows
    )
    if not integrity_passed:
        decision = "INVALID_BLOCK_SPAN_CORRECTION_CONTROL_FAILURE"
    elif decisive_rejection:
        decision = "REJECT_ONE_SWEEP_BLOCK_SPAN_CORRECTION_AS_405B_CORE"
    else:
        decision = "SURVIVES_BLOCK_SPAN_LOWER_BOUND_REQUIRES_FULL_EXACT_COEFFICIENT_GATE"

    observed_required_directions = int(holdout_aggregate["extra_rank_max"])
    projection = {
        "threshold_directions": threshold,
        "threshold_extra_fraction": threshold / block_length,
        "threshold_operation_multiplier": correction_operation_multiplier(
            block_length, threshold
        ),
        "threshold_sidecar_bytes": optimistic_sidecar_bytes(threshold),
        "threshold_sidecar_within_8gib": optimistic_sidecar_bytes(threshold)
        <= int(config["target_projection"]["target_hot_limit_bytes"]),
        "observed_certified_required_directions": observed_required_directions,
        "observed_certified_extra_fraction_lower_bound": observed_required_directions
        / block_length,
        "observed_operation_multiplier_lower_bound": correction_operation_multiplier(
            block_length, observed_required_directions
        ),
        "observed_sidecar_bytes_lower_bound": optimistic_sidecar_bytes(
            observed_required_directions
        ),
        "claim": "dimension-derived lower bound under one static BF16 basis direction per operator input; coefficients, nonlinear recomputation, routing, native-order repair, and basis discovery are granted free",
    }

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": "one-guessed-target-block/complete-linear-operator-input-cache/exact-dyadic-row-span/192-coordinate-restriction/three-prime-rank-lower-bound/true-teacher-forced-oracle/first-middle-last-complete-layers",
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "block_length": block_length,
        "seed_mode": config["seed_mode"],
        "layer_indices": list(layer_indices),
        "operator_roles": list(roles),
        "rank_sketch_columns": rank_columns,
        "rank_primes": list(primes),
        "controls": controls,
        "case_rows": case_rows,
        "span_rows": rows,
        "build_aggregate": build_aggregate,
        "holdout_aggregate": holdout_aggregate,
        "target_projection": projection,
        "gates": {
            "integrity_passed": integrity_passed,
            "integrity_failures": integrity_failures,
            "decisive_holdout_lower_bound_exceeds_threshold": decisive_rejection,
            "maximum_extra_directions_per_operator": threshold,
            "maximum_extra_fraction": float(config["maximum_extra_fraction"]),
        },
        "authoritative_decision": decision,
    }
    deterministic_core_sha256 = canonical_sha256(deterministic_core)
    result = {
        **deterministic_core,
        "deterministic_core_sha256": deterministic_core_sha256,
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "experiment_ns": experiment_ns,
            "official_prefill_calls": len(case_rows),
            "official_ar_calls": len(case_rows) * block_length,
            "official_block_calls": len(case_rows) * 2,
        },
        "provenance": {
            "measured": [
                "DEV-W guessed and true teacher-forced complete operator-input blocks, official AR equality, modular ranks on frozen coordinate restrictions"
            ],
            "derived": [
                "certified rational-rank increment lower bounds, correction-direction fractions, optimistic 405B sidecar and operation multipliers"
            ],
            "granted": [
                "true block oracle, exact coefficients, basis discovery, routing, nonlinear recomputation, coefficient application, and native-rounding repair"
            ],
            "unverified": [
                "full-coordinate exact ranks when the lower-bound Gate survives, causal coefficient constructor, all 405B layers, target GPU/SSD/PCIe/HBM, exact sparse correction kernel, 4B-class p50/p95"
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/span_rows.json", rows)
    write_json(output_dir / "raw/case_rows.json", case_rows)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/deterministic_core.json",
            "raw/span_rows.json",
            "raw/case_rows.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "holdout_extra_rank_p50": holdout_aggregate["extra_rank_p50"],
                "holdout_extra_rank_p95": holdout_aggregate["extra_rank_p95"],
                "holdout_extra_rank_max": holdout_aggregate["extra_rank_max"],
                "threshold_directions": threshold,
                "observed_operation_multiplier_lower_bound": projection[
                    "observed_operation_multiplier_lower_bound"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
