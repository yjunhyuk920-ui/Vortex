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

from vortex_runtime.exact_jacobi_fixed_point import (  # noqa: E402
    build_seed,
    jacobi_input,
    stable_argmax_rows,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    config_dict,
    model_tensor_sha256,
)
from vortex_runtime.one_sweep_true_token_rank import (  # noqa: E402
    candidate_id_bytes,
    canonical_sha256,
    flatten_rank_rows,
    longest_common_prefix,
    model_free_controls,
    percentile,
    rank_histogram,
    rank_metrics,
    stable_true_token_rank,
    static_topk_gate,
    tensor_sha256,
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
    if config.get("schema") != "exp-093a-one-sweep-true-token-rank-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("configuration was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-093A freezes K=128")
    if str(config["seed_mode"]) != "prompt_suffix_cycle_16":
        raise RuntimeError("EXP-093A freezes the EXP-091A selected seed")
    if int(config["p95_true_token_rank_limit"]) != 4:
        raise RuntimeError("p95 rank threshold changed")
    if int(config["maximum_true_token_rank_limit"]) != 16:
        raise RuntimeError("maximum rank threshold changed")
    if int(config["candidate_token_bytes"]) != 2:
        raise RuntimeError("candidate-token representation changed")

    target = config["target_projection"]
    if target["model_id"] != TARGET_MODEL_ID or target["revision"] != TARGET_REVISION:
        raise RuntimeError("TARGET-W identity mismatch")
    block_length = int(config["block_length"])
    ratio = float(target["checkpoint_compression_ratio_favorable"])
    expected_no_compression = 1.0 / block_length
    expected_compressed = 1.0 / (block_length * ratio)
    if abs(float(target["one_sweep_no_compression_fraction"]) - expected_no_compression) > 1e-15:
        raise RuntimeError("no-compression one-sweep fraction mismatch")
    if abs(float(target["one_sweep_favorable_compression_fraction"]) - expected_compressed) > 1e-15:
        raise RuntimeError("compressed one-sweep fraction mismatch")

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
    prompt_hashes = [sha256_bytes(case["prompt"].encode("utf-8")) for case in cases]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("duplicate prompts")


def load_model(config: Mapping[str, Any]) -> tuple[Any, Any, int]:
    started_ns = time.perf_counter_ns()
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
    return model, tokenizer, time.perf_counter_ns() - started_ns


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


def prefill_case(model: Any, tokenizer: Any, case: Mapping[str, Any]) -> PrefillState:
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


def run_guessed_sweep(
    model: Any,
    prefill: PrefillState,
    seed: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, ...], dict[str, Any]]:
    input_tokens = jacobi_input(prefill.boundary_token, seed)
    cache = fresh_cache(prefill.prefix_cache)
    started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([input_tokens], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    completed_ns = time.perf_counter_ns()
    logits = output.logits.detach().to(torch.float32).contiguous().cpu()
    if logits.ndim != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != len(seed):
        raise RuntimeError(f"unexpected guessed-sweep logits shape: {tuple(logits.shape)}")
    proposals = tuple(
        int(value) for value in stable_argmax_rows(logits)[0].tolist()
    )
    return logits[0], proposals, {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "input_sha256": canonical_sha256(input_tokens),
        "seed_sha256": canonical_sha256(seed),
        "proposal_sha256": canonical_sha256(proposals),
        "logits_sha256": tensor_sha256(logits),
        "post_sweep_cache_sha256": cache_sha256(output.past_key_values),
        "post_sweep_cache_bytes": cache_nbytes(output.past_key_values),
        "target_information_used": False,
    }


def run_incremental_target(
    model: Any,
    prefill: PrefillState,
    guessed_logits: torch.Tensor,
    *,
    block_length: int,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    if guessed_logits.ndim != 2 or int(guessed_logits.shape[0]) != int(block_length):
        raise RuntimeError("guessed logits do not cover the frozen block")
    cache: Any = fresh_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    ranks: list[int] = []
    step_logits_sha256: list[str] = []
    started_ns = time.perf_counter_ns()
    for position in range(int(block_length)):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        logits = output.logits[0, -1].detach().to(torch.float32).contiguous().cpu()
        token = int(stable_argmax_rows(logits.unsqueeze(0))[0].item())
        tokens.append(token)
        ranks.append(stable_true_token_rank(guessed_logits[position], token))
        step_logits_sha256.append(tensor_sha256(logits))
        current = token
        cache = output.past_key_values
    completed_ns = time.perf_counter_ns()
    return tuple(tokens), tuple(ranks), {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "target_calls": int(block_length),
        "target_sha256": canonical_sha256(tokens),
        "step_logits_sha256": canonical_sha256(step_logits_sha256),
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
    }


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
        "checkpoint_tensor_sha256": model_tensor_sha256(model),
        "config_sha256": canonical_sha256(config_dict(model.config)),
        "tokenizer": {
            **tokenizer_payload,
            "sha256": canonical_sha256(tokenizer_payload),
        },
    }


def case_row(
    model: Any,
    tokenizer: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prefill = prefill_case(model, tokenizer, case)
    seed, seed_manifest = build_seed(
        str(config["seed_mode"]),
        prompt_ids=prefill.prompt_ids,
        boundary_token=prefill.boundary_token,
        block_length=int(config["block_length"]),
    )
    guessed_logits, proposals, sweep_meta = run_guessed_sweep(model, prefill, seed)
    target_tokens, ranks, target_meta = run_incremental_target(
        model,
        prefill,
        guessed_logits,
        block_length=int(config["block_length"]),
    )
    metrics = rank_metrics(ranks)
    gate = static_topk_gate(
        ranks,
        p95_limit=int(config["p95_true_token_rank_limit"]),
        maximum_limit=int(config["maximum_true_token_rank_limit"]),
    )
    exact_positions = sum(
        int(proposal == target)
        for proposal, target in zip(proposals, target_tokens)
    )
    ordering_passed = int(sweep_meta["completed_ns"]) <= int(target_meta["started_ns"])
    first_position_control = (
        ranks[0] == 1
        and proposals[0] == target_tokens[0]
    )
    return {
        "case_id": prefill.case_id,
        "split": prefill.split,
        "family": prefill.family,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "boundary_token": prefill.boundary_token,
        "prefix_cache_sha256": prefill.prefix_cache_sha256,
        "prefix_cache_bytes": prefill.prefix_cache_bytes,
        "seed_manifest": seed_manifest.to_dict(),
        "sweep": sweep_meta,
        "target": target_meta,
        "sweep_precedes_target": ordering_passed,
        "first_position_exact_control": first_position_control,
        "proposal_target_exact_positions": exact_positions,
        "proposal_target_exact_fraction": exact_positions / len(target_tokens),
        "proposal_target_prefix": longest_common_prefix(proposals, target_tokens),
        "proposal_sha256": canonical_sha256(proposals),
        "target_sha256": canonical_sha256(target_tokens),
        "true_token_ranks": list(ranks),
        "rank_histogram": rank_histogram(ranks),
        "rank_metrics": metrics.to_dict(),
        "static_topk_gate": gate,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], split: str, config: Mapping[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == str(split)]
    if not selected:
        raise RuntimeError(f"empty aggregate: {split}")
    ranks = flatten_rank_rows(row["true_token_ranks"] for row in selected)
    metrics = rank_metrics(ranks)
    gate = static_topk_gate(
        ranks,
        p95_limit=int(config["p95_true_token_rank_limit"]),
        maximum_limit=int(config["maximum_true_token_rank_limit"]),
    )
    path_bits = [
        float(row["rank_metrics"]["static_path_information_bits"])
        for row in selected
    ]
    return {
        "split": split,
        "case_count": len(selected),
        "position_count": len(ranks),
        "metrics": metrics.to_dict(),
        "rank_histogram": rank_histogram(ranks),
        "per_case_p95": [float(row["rank_metrics"]["p95"]) for row in selected],
        "per_case_maximum": [int(row["rank_metrics"]["maximum"]) for row in selected],
        "per_case_top16_fraction": [
            float(row["rank_metrics"]["top16_fraction"]) for row in selected
        ],
        "per_case_static_path_information_bits": path_bits,
        "static_path_information_bits_p50": percentile(
            [int(round(value * 1_000_000)) for value in path_bits], 50.0
        )
        / 1_000_000.0,
        "static_path_information_bits_max": max(path_bits),
        "all_case_gates_passed": all(
            bool(row["static_topk_gate"]["passed"]) for row in selected
        ),
        "gate": gate,
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    controls = model_free_controls()
    if not controls["passed"]:
        raise RuntimeError("model-free true-token rank controls failed")

    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_load_ns = load_model(config)
    identity = checkpoint_identity(model, tokenizer, config)
    if int(identity["tokenizer"]["vocab_size"]) > 1 << (
        8 * int(config["candidate_token_bytes"])
    ):
        raise RuntimeError("candidate token width cannot encode the vocabulary")
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)

    rows: list[dict[str, Any]] = []
    experiment_started_ns = time.perf_counter_ns()
    for case in config["cases"]:
        row = case_row(model, tokenizer, case, config)
        rows.append(row)
        metrics = row["rank_metrics"]
        print(
            f"{row['split']} {row['case_id']}: "
            f"rank p50/p95/max={metrics['p50']}/{metrics['p95']}/{metrics['maximum']} "
            f"top16={metrics['top16_fraction']:.6%}",
            flush=True,
        )
    experiment_ns = time.perf_counter_ns() - experiment_started_ns

    build_aggregate = aggregate(rows, "build", config)
    holdout_aggregate = aggregate(rows, "holdout", config)
    prompt_hashes = [str(row["prompt_sha256"]) for row in rows]
    integrity_failures: list[str] = []
    if len(prompt_hashes) != len(set(prompt_hashes)):
        integrity_failures.append("duplicate_prompt_hash")
    if not all(bool(row["sweep_precedes_target"]) for row in rows):
        integrity_failures.append("target_started_before_sweep_completed")
    if not all(bool(row["first_position_exact_control"]) for row in rows):
        integrity_failures.append("first_position_reference_control")
    if not all(
        int(row["target"]["target_calls"]) == int(config["block_length"])
        and len(row["true_token_ranks"]) == int(config["block_length"])
        for row in rows
    ):
        integrity_failures.append("incomplete_target_or_rank_trace")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")
    integrity_passed = (
        controls["passed"]
        and len(integrity_failures)
        <= int(config["success"]["maximum_integrity_failures"])
    )

    population_gate_passed = (
        bool(build_aggregate["gate"]["passed"])
        and bool(build_aggregate["all_case_gates_passed"])
        and bool(holdout_aggregate["gate"]["passed"])
        and bool(holdout_aggregate["all_case_gates_passed"])
    )
    if not integrity_passed:
        authoritative_decision = "INVALID_ONE_SWEEP_TRUE_TOKEN_RANK_CONTROL_FAILURE"
    elif population_gate_passed:
        authoritative_decision = (
            "PROMOTE_ONE_SWEEP_STATIC_TOPK_TO_EXACT_BRANCH_CLOSURE_GATE"
        )
    else:
        authoritative_decision = (
            "REJECT_ONE_SWEEP_STATIC_TOPK_BRANCH_SOURCE_AS_405B_CORE"
        )

    block_length = int(config["block_length"])
    max_candidates = int(config["maximum_true_token_rank_limit"])
    target_projection = {
        **dict(config["target_projection"]),
        "candidate_id_bytes_per_block": candidate_id_bytes(
            block_length=block_length,
            maximum_candidates=max_candidates,
            token_bytes=int(config["candidate_token_bytes"]),
        ),
        "one_sweep_dense_arithmetic_fraction_per_token": 1.0,
        "branch_transition_construction_cost": "NOT_MEASURED",
        "branch_verification_cost": "NOT_MEASURED",
        "claim": (
            "one lossless checkpoint sweep may amortize checkpoint traffic over 128 positions; "
            "it does not reduce the 128 dense token computations and does not construct branch successors"
        ),
    }

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "one-official-guessed-context-sweep/frozen-prompt-suffix-cycle-16/"
            "complete-vocabulary-fp32-logits/stable-true-token-rank/"
            "static-top4-p95-top16-worst-case/target-after-sweep/"
            "build-and-untouched-holdout"
        ),
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "block_length": block_length,
        "seed_mode": config["seed_mode"],
        "controls": {
            "model_free": controls,
            "integrity_failures": integrity_failures,
            "integrity_passed": integrity_passed,
        },
        "case_rows": rows,
        "build_aggregate": build_aggregate,
        "holdout_aggregate": holdout_aggregate,
        "gates": {
            "p95_true_token_rank_limit": int(
                config["p95_true_token_rank_limit"]
            ),
            "maximum_true_token_rank_limit": max_candidates,
            "build_population_passed": bool(build_aggregate["gate"]["passed"]),
            "holdout_population_passed": bool(
                holdout_aggregate["gate"]["passed"]
            ),
            "all_case_gates_passed": population_gate_passed,
            "integrity_passed": integrity_passed,
            "promotion_gate_passed": integrity_passed and population_gate_passed,
        },
        "target_projection": target_projection,
        "authoritative_decision": authoritative_decision,
    }
    deterministic_core_sha256 = canonical_sha256(deterministic_core)
    result = {
        **deterministic_core,
        "deterministic_core_sha256": deterministic_core_sha256,
        "provenance": {
            "measured": [
                "official DEV-W guessed-context full-vocabulary logits, delayed official incremental greedy target tokens and caches, stable true-token ranks, ordering controls"
            ],
            "derived": [
                "top-k coverage, oracle static-path information volume, one-sweep logical checkpoint traffic, candidate-ID bytes"
            ],
            "granted": [
                "candidate-set extraction and token-ID storage have zero runtime cost; future branch transitions are not credited"
            ],
            "unverified": [
                "exact branch successor construction, branch verification cost, arithmetic reduction, TARGET-W behavior, 8-GiB VRAM, target latency"
            ],
        },
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "experiment_ns": experiment_ns,
            "prefill_calls": len(rows),
            "guessed_block_calls": len(rows),
            "incremental_target_calls": len(rows) * block_length,
        },
    }

    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/case_rows.json", rows)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/deterministic_core.json",
            "raw/case_rows.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": authoritative_decision,
                "holdout_rank_p50": holdout_aggregate["metrics"]["p50"],
                "holdout_rank_p95": holdout_aggregate["metrics"]["p95"],
                "holdout_rank_max": holdout_aggregate["metrics"]["maximum"],
                "holdout_top16_fraction": holdout_aggregate["metrics"][
                    "top16_fraction"
                ],
                "promotion_gate_passed": integrity_passed
                and population_gate_passed,
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
