from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.reality_first_speculative import (
    CaseAccounting,
    evaluate_holdout_gate,
    minimum_committed_tokens,
    select_build_block_length,
    verify_greedy_block,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-102a-reality-first-causal-block-v1":
        raise RuntimeError("unexpected schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("config was not frozen before results")
    if config.get("authoritative_arm") != "REAL_EXECUTOR_ONLY":
        raise RuntimeError("authoritative arm must be REAL_EXECUTOR_ONLY")
    forbidden = set(config.get("forbidden_grants", []))
    required_forbidden = {
        "future_target_tokens",
        "perfect_selector",
        "free_transforms",
        "free_metadata",
        "free_workspace",
        "free_repair",
        "free_fallback",
        "free_N_over_A",
        "unmeasured_compression",
        "peak_throughput_as_measurement",
    }
    if forbidden != required_forbidden:
        raise RuntimeError("reality-first forbidden-grant set changed")
    if config["target_contract"]["raw_p50_fraction"] != 0.011851851851851851:
        raise RuntimeError("registered p50 traffic fraction changed")
    if config["target_contract"]["raw_p95_fraction"] != 0.014814814814814815:
        raise RuntimeError("registered p95 traffic fraction changed")
    if config["target_contract"]["p50_latency_ratio"] != 1.2:
        raise RuntimeError("p50 latency target changed")
    if config["target_contract"]["p95_latency_ratio"] != 1.5:
        raise RuntimeError("p95 latency target changed")
    block_lengths = [int(value) for value in config["block_lengths"]]
    if block_lengths != sorted(set(block_lengths)) or min(block_lengths) <= 0:
        raise RuntimeError("invalid block lengths")
    prompts = config["prompts"]
    build = [row for row in prompts if row["split"] == "build"]
    holdout = [row for row in prompts if row["split"] == "holdout"]
    if len(build) < 3 or len(holdout) < 3:
        raise RuntimeError("need at least three build and holdout prompts")
    if len({row["id"] for row in prompts}) != len(prompts):
        raise RuntimeError("duplicate prompt IDs")
    if len(config["draft_arms"]) < 2:
        raise RuntimeError("same-family and cross-family arms are required")
    names = {row["name"] for row in config["draft_arms"]}
    if names != {"same_family", "cross_family_text_bridge"}:
        raise RuntimeError("draft arm population changed")


def model_parameter_bytes(model: Any) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def cache_bytes(cache: Any) -> int:
    total = 0
    for key, value in cache:
        total += int(key.numel() * key.element_size())
        total += int(value.numel() * value.element_size())
    return total


def cache_sha256(cache: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for layer_index, (key, value) in enumerate(cache):
        for kind, tensor in (("k", key), ("v", value)):
            contiguous = tensor.detach().contiguous().cpu()
            digest.update(f"{layer_index}:{kind}:{tuple(contiguous.shape)}:{contiguous.dtype}".encode("utf-8"))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tokenizer_sha256(tokenizer: Any) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    return sha256_json(payload)


def greedy_token(logits: Any) -> int:
    return int(logits[0, -1].argmax(dim=-1).item())


def crop_cache(cache: Any, length: int) -> None:
    if not hasattr(cache, "crop"):
        raise RuntimeError(f"cache does not support exact crop: {type(cache)!r}")
    cache.crop(int(length))


def bridge_completion_to_target(
    *,
    prompt_text: str,
    completion_text: str,
    target_tokenizer: Any,
    target_prompt_ids: Sequence[int],
    limit: int,
) -> tuple[list[int], dict[str, Any]]:
    combined_ids = target_tokenizer(
        prompt_text + completion_text,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    prefix = [int(value) for value in target_prompt_ids]
    stable = combined_ids[: len(prefix)] == prefix
    suffix = [int(value) for value in combined_ids[len(prefix) :]] if stable else []
    suffix = suffix[: int(limit)]
    return suffix, {
        "bridge_prefix_stable": bool(stable),
        "draft_text_utf8_bytes": len(completion_text.encode("utf-8")),
        "target_suffix_tokens": len(suffix),
    }


def prefill(model: Any, input_ids: Any, DynamicCache: Any) -> tuple[Any, int]:
    cache = DynamicCache()
    output = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return cache, greedy_token(output.logits)


def generate_draft_tokens(
    *,
    model: Any,
    prompt_ids: Any,
    requested_tokens: int,
    DynamicCache: Any,
    torch: Any,
) -> tuple[list[int], int, int, Any]:
    # The draft prefill is additional work absent from the target-only warm
    # baseline in this one-block Gate, so it is measured rather than granted.
    prefill_started = time.perf_counter_ns()
    cache, next_token = prefill(model, prompt_ids, DynamicCache)
    prefill_ns = time.perf_counter_ns() - prefill_started

    generated: list[int] = []
    decode_started = time.perf_counter_ns()
    for _ in range(int(requested_tokens)):
        token = int(next_token)
        generated.append(token)
        token_tensor = torch.tensor([[token]], dtype=torch.long)
        output = model(
            input_ids=token_tensor,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token = greedy_token(output.logits)
        eos = getattr(model.config, "eos_token_id", None)
        if eos is not None and token == int(eos):
            break
    decode_ns = time.perf_counter_ns() - decode_started
    return generated, int(prefill_ns), int(decode_ns), cache


def synchronize_direct_draft_cache(
    *,
    model: Any,
    cache: Any,
    prompt_length: int,
    accepted_draft_tokens: int,
    mismatch_token: int | None,
    torch: Any,
) -> tuple[int, int]:
    """Crop/replay the real same-tokenizer draft state after verification."""

    started = time.perf_counter_ns()
    crop_cache(cache, int(prompt_length) + int(accepted_draft_tokens))
    if mismatch_token is not None:
        model(
            input_ids=torch.tensor([[int(mismatch_token)]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    elapsed = time.perf_counter_ns() - started
    return int(elapsed), int(cache_bytes(cache))


def rebuild_cross_family_draft_state(
    *,
    model: Any,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    target_prompt_ids: Sequence[int],
    committed_tokens: Sequence[int],
    DynamicCache: Any,
) -> tuple[int, int, str]:
    """Charge target decode, draft retokenization, and full causal prefill."""

    started = time.perf_counter_ns()
    committed_text = target_tokenizer.decode(
        [int(value) for value in target_prompt_ids]
        + [int(value) for value in committed_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    ids = draft_tokenizer(
        committed_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    cache, _ = prefill(model, ids, DynamicCache)
    elapsed = time.perf_counter_ns() - started
    return int(elapsed), int(cache_bytes(cache)), committed_text


def target_reference_decode(
    *,
    model: Any,
    prompt_ids: Any,
    token_count: int,
    DynamicCache: Any,
    torch: Any,
) -> tuple[list[int], Any, int, int]:
    prefill_started = time.perf_counter_ns()
    cache, next_token = prefill(model, prompt_ids, DynamicCache)
    prefill_ns = time.perf_counter_ns() - prefill_started
    tokens: list[int] = []
    started = time.perf_counter_ns()
    for _ in range(int(token_count)):
        token = int(next_token)
        tokens.append(token)
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token = greedy_token(output.logits)
    elapsed = time.perf_counter_ns() - started
    return tokens, cache, int(prefill_ns), int(elapsed)


def run_case(
    *,
    case: Mapping[str, Any],
    requested_block_length: int,
    arm: Mapping[str, Any],
    target_model: Any,
    target_tokenizer: Any,
    draft_model: Any,
    draft_tokenizer: Any,
    DynamicCache: Any,
    torch: Any,
) -> tuple[CaseAccounting, dict[str, Any]]:
    prompt_text = str(case["prompt"])
    target_prompt_ids = target_tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    draft_prompt_ids = draft_tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    if target_prompt_ids.shape[1] < 1 or draft_prompt_ids.shape[1] < 1:
        raise RuntimeError("empty prompt tokenization")

    generated_draft_ids, draft_prefill_ns, draft_decode_ns, draft_cache = generate_draft_tokens(
        model=draft_model,
        prompt_ids=draft_prompt_ids,
        requested_tokens=int(arm["draft_generation_multiplier"] * requested_block_length),
        DynamicCache=DynamicCache,
        torch=torch,
    )
    draft_cache_peak = cache_bytes(draft_cache)

    bridge_started = time.perf_counter_ns()
    if bool(arm["direct_token_ids"]):
        candidate_ids = generated_draft_ids[:requested_block_length]
        bridge = {
            "bridge_prefix_stable": True,
            "draft_text_utf8_bytes": 0,
            "target_suffix_tokens": len(candidate_ids),
            "direct_token_ids": True,
        }
    else:
        # Decoding is part of the cross-tokenizer bridge and is charged here.
        draft_text = draft_tokenizer.decode(
            generated_draft_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        candidate_ids, bridge = bridge_completion_to_target(
            prompt_text=prompt_text,
            completion_text=draft_text,
            target_tokenizer=target_tokenizer,
            target_prompt_ids=target_prompt_ids[0].tolist(),
            limit=requested_block_length,
        )
        bridge["direct_token_ids"] = False
    bridge_ns = time.perf_counter_ns() - bridge_started
    if not candidate_ids:
        # Fail closed with a one-position explicit target call. This keeps every
        # case executable and makes the missing candidate visible in N/A.
        target_prefill_started = time.perf_counter_ns()
        target_cache, first_target = prefill(target_model, target_prompt_ids, DynamicCache)
        target_prefill_ns = time.perf_counter_ns() - target_prefill_started
        target_started = time.perf_counter_ns()
        target_model(
            input_ids=torch.tensor([[first_target]], dtype=torch.long),
            past_key_values=target_cache,
            use_cache=True,
            return_dict=True,
        )
        target_verify_ns = time.perf_counter_ns() - target_started
        exact_tokens = [first_target]
        speculative_cache = target_cache
        target_positions = 1
        target_repair_ns = 0
        accepted = 0
        mismatch_index = 0
    else:
        target_prefill_started = time.perf_counter_ns()
        target_cache, first_target = prefill(target_model, target_prompt_ids, DynamicCache)
        target_prefill_ns = time.perf_counter_ns() - target_prefill_started
        verify_started = time.perf_counter_ns()
        candidate_tensor = torch.tensor([candidate_ids], dtype=torch.long)
        verify_output = target_model(
            input_ids=candidate_tensor,
            past_key_values=target_cache,
            use_cache=True,
            return_dict=True,
        )
        subsequent = (
            verify_output.logits[:, :-1, :].argmax(dim=-1)[0].tolist()
            if len(candidate_ids) > 1
            else []
        )
        target_tokens = [first_target] + [int(value) for value in subsequent]
        outcome = verify_greedy_block(candidate_ids, target_tokens)
        target_verify_ns = time.perf_counter_ns() - verify_started
        accepted = outcome.accepted_draft_tokens
        mismatch_index = outcome.mismatch_index
        exact_tokens = list(outcome.exact_committed_tokens)
        target_positions = len(candidate_ids)
        crop_cache(target_cache, int(target_prompt_ids.shape[1]) + accepted)
        target_repair_ns = 0
        if mismatch_index is not None:
            repair_token = exact_tokens[-1]
            repair_started = time.perf_counter_ns()
            target_model(
                input_ids=torch.tensor([[repair_token]], dtype=torch.long),
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
            target_repair_ns = time.perf_counter_ns() - repair_started
            target_positions += 1
        speculative_cache = target_cache

    if bool(arm["direct_token_ids"]):
        mismatch_token = exact_tokens[-1] if mismatch_index is not None else None
        rollback_ns, rebuilt_draft_cache_bytes = synchronize_direct_draft_cache(
            model=draft_model,
            cache=draft_cache,
            prompt_length=int(draft_prompt_ids.shape[1]),
            accepted_draft_tokens=int(accepted),
            mismatch_token=mismatch_token,
            torch=torch,
        )
        committed_text = ""
        draft_state_strategy = "crop_and_replay_mismatch"
    else:
        rollback_ns, rebuilt_draft_cache_bytes, committed_text = rebuild_cross_family_draft_state(
            model=draft_model,
            draft_tokenizer=draft_tokenizer,
            target_tokenizer=target_tokenizer,
            target_prompt_ids=target_prompt_ids[0].tolist(),
            committed_tokens=exact_tokens,
            DynamicCache=DynamicCache,
        )
        draft_state_strategy = "full_text_retokenize_and_prefill"

    reference_tokens, reference_cache, baseline_prefill_ns, baseline_ns = target_reference_decode(
        model=target_model,
        prompt_ids=target_prompt_ids,
        token_count=len(exact_tokens),
        DynamicCache=DynamicCache,
        torch=torch,
    )
    token_match = reference_tokens == exact_tokens
    state_match = cache_sha256(reference_cache) == cache_sha256(speculative_cache)

    accounting = CaseAccounting(
        case_id=str(case["id"]),
        block_length=int(requested_block_length),
        accepted_draft_tokens=int(accepted),
        committed_tokens=len(exact_tokens),
        target_positions_evaluated=int(target_positions),
        draft_prefill_ns=int(draft_prefill_ns),
        draft_decode_ns=int(draft_decode_ns),
        bridge_ns=int(bridge_ns),
        target_verify_ns=int(target_verify_ns),
        target_repair_ns=int(target_repair_ns),
        draft_rollback_ns=int(rollback_ns),
        baseline_decode_ns=int(baseline_ns),
        exact_token_match=bool(token_match),
        exact_terminal_state_match=bool(state_match),
    )
    detail = {
        **asdict(accounting),
        "arm": arm["name"],
        "split": case["split"],
        "family": case["family"],
        "requested_block_length": int(requested_block_length),
        "proposed_target_tokens": len(candidate_ids),
        "draft_generated_tokens": len(generated_draft_ids),
        "mismatch_index": mismatch_index,
        "candidate_ratio": accounting.candidate_ratio,
        "latency_ratio": accounting.latency_ratio,
        "speculative_ns": accounting.speculative_ns,
        "target_prompt_tokens": int(target_prompt_ids.shape[1]),
        "target_prefill_ns_excluded_common_warm_boundary": int(target_prefill_ns),
        "baseline_prefill_ns_excluded_common_warm_boundary": int(baseline_prefill_ns),
        "draft_prompt_tokens": int(draft_prompt_ids.shape[1]),
        "draft_cache_peak_bytes": int(draft_cache_peak),
        "draft_rebuilt_cache_bytes": int(rebuilt_draft_cache_bytes),
        "draft_state_strategy": draft_state_strategy,
        "committed_text_utf8_bytes": len(committed_text.encode("utf-8")) if committed_text else 0,
        "resident_parameter_bytes": int(model_parameter_bytes(target_model) + model_parameter_bytes(draft_model)),
        "process_peak_rss_bytes_at_case_end": int(rss_bytes()),
        "target_terminal_cache_bytes": int(cache_bytes(speculative_cache)),
        "target_terminal_cache_sha256": cache_sha256(speculative_cache),
        "reference_terminal_cache_sha256": cache_sha256(reference_cache),
        "bridge": bridge,
    }
    return accounting, detail


def load_model_and_tokenizer(spec: Mapping[str, Any], *, torch: Any, AutoModelForCausalLM: Any, AutoTokenizer: Any) -> tuple[Any, Any, dict[str, Any]]:
    load_started = time.perf_counter_ns()
    dtype_name = str(spec["dtype"])
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    tokenizer_id = spec.get("tokenizer_id", spec["model_id"])
    tokenizer_revision = spec.get("tokenizer_revision", spec["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        use_fast=True,
    )
    model_kwargs = {
        "revision": spec["revision"],
        "torch_dtype": dtype,
        "low_cpu_mem_usage": False,
    }
    if spec.get("attention_implementation") is not None:
        model_kwargs["attn_implementation"] = spec["attention_implementation"]
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        **model_kwargs,
    )
    model.eval()
    identity = {
        "model_id": spec["model_id"],
        "requested_revision": spec["revision"],
        "resolved_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "model_class": model.__class__.__name__,
        "dtype": dtype_name,
        "parameter_bytes": model_parameter_bytes(model),
        "vocab_size": int(model.config.vocab_size),
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
        "load_wall_ns": int(time.perf_counter_ns() - load_started),
    }
    if identity["resolved_revision"] and identity["resolved_revision"] != spec["revision"]:
        raise RuntimeError(f"resolved revision mismatch: {identity}")
    return model, tokenizer, identity


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(config["runtime"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()

    raw_p50 = minimum_committed_tokens(
        target_fraction=float(config["target_contract"]["raw_p50_fraction"])
    )
    raw_p95 = minimum_committed_tokens(
        target_fraction=float(config["target_contract"]["raw_p95_fraction"])
    )

    target_model, target_tokenizer, target_identity = load_model_and_tokenizer(
        config["target"],
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )

    prompt_rows = list(config["prompts"])
    build_specs = [row for row in prompt_rows if row["split"] == "build"]
    holdout_specs = [row for row in prompt_rows if row["split"] == "holdout"]
    all_details: list[dict[str, Any]] = []
    arm_results: dict[str, Any] = {}
    integrity_failures: list[str] = []

    for arm in config["draft_arms"]:
        draft_model, draft_tokenizer, draft_identity = load_model_and_tokenizer(
            arm,
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
        )
        if bool(arm["direct_token_ids"]):
            if target_identity["tokenizer_sha256"] != draft_identity["tokenizer_sha256"]:
                integrity_failures.append(f"{arm['name']}:direct_tokenizer_mismatch")
            if target_identity["vocab_size"] != draft_identity["vocab_size"]:
                integrity_failures.append(f"{arm['name']}:direct_vocab_mismatch")

        build_rows: list[CaseAccounting] = []
        for block_length in config["block_lengths"]:
            for case in build_specs:
                accounting, detail = run_case(
                    case=case,
                    requested_block_length=int(block_length),
                    arm=arm,
                    target_model=target_model,
                    target_tokenizer=target_tokenizer,
                    draft_model=draft_model,
                    draft_tokenizer=draft_tokenizer,
                    DynamicCache=DynamicCache,
                    torch=torch,
                )
                build_rows.append(accounting)
                all_details.append(detail)
                print(json.dumps({
                    "arm": arm["name"],
                    "split": "build",
                    "case": case["id"],
                    "K": block_length,
                    "committed": accounting.committed_tokens,
                    "accepted": accounting.accepted_draft_tokens,
                    "N_over_A": accounting.candidate_ratio,
                    "latency_ratio": accounting.latency_ratio,
                    "exact_token": accounting.exact_token_match,
                    "exact_state": accounting.exact_terminal_state_match,
                }), flush=True)

        selected_k = select_build_block_length(
            build_rows,
            raw_required_tokens=raw_p50.minimum_committed_tokens,
        )
        holdout_rows: list[CaseAccounting] = []
        for case in holdout_specs:
            accounting, detail = run_case(
                case=case,
                requested_block_length=selected_k,
                arm=arm,
                target_model=target_model,
                target_tokenizer=target_tokenizer,
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,
                DynamicCache=DynamicCache,
                torch=torch,
            )
            holdout_rows.append(accounting)
            all_details.append(detail)
            print(json.dumps({
                "arm": arm["name"],
                "split": "holdout",
                "case": case["id"],
                "K": selected_k,
                "committed": accounting.committed_tokens,
                "accepted": accounting.accepted_draft_tokens,
                "N_over_A": accounting.candidate_ratio,
                "latency_ratio": accounting.latency_ratio,
                "exact_token": accounting.exact_token_match,
                "exact_state": accounting.exact_terminal_state_match,
            }), flush=True)

        gate = evaluate_holdout_gate(
            holdout_rows,
            selected_block_length=selected_k,
            raw_required_tokens=raw_p50.minimum_committed_tokens,
            p50_latency_limit=float(config["target_contract"]["p50_latency_ratio"]),
            p95_latency_limit=float(config["target_contract"]["p95_latency_ratio"]),
            p95_candidate_ratio_limit=float(config["target_contract"]["p95_candidate_ratio"]),
        )
        arm_results[arm["name"]] = {
            "draft_identity": draft_identity,
            "selected_block_length_from_build_only": selected_k,
            "build_rows": [asdict(row) | {
                "candidate_ratio": row.candidate_ratio,
                "latency_ratio": row.latency_ratio,
                "speculative_ns": row.speculative_ns,
            } for row in build_rows],
            "holdout_gate": gate,
        }
        del draft_model
        del draft_tokenizer
        gc.collect()

    same_pass = bool(arm_results["same_family"]["holdout_gate"]["gate_passed"])
    cross_pass = bool(arm_results["cross_family_text_bridge"]["holdout_gate"]["gate_passed"])
    if integrity_failures:
        decision = config["decisions"]["control_failure"]
    elif same_pass and cross_pass:
        decision = config["decisions"]["promotion"]
    elif same_pass:
        decision = config["decisions"]["restricted_auxiliary"]
    else:
        decision = config["decisions"]["scientific_rejection"]

    core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "real-causal-greedy-draft/real-target-block-verification/"
            "charged-draft-prefill-and-decode/charged-cross-tokenizer-bridge/"
            "charged-target-N-over-A/charged-mismatch-repair/"
            "charged-real-draft-cache-crop-or-rebuild/raw-no-compression-traffic-gate/"
            "exact-token-plus-terminal-KV/same-and-cross-family-arms"
        ),
        "authoritative_arm": config["authoritative_arm"],
        "forbidden_grants": config["forbidden_grants"],
        "target_identity": target_identity,
        "requirements": {
            "raw_p50": asdict(raw_p50),
            "raw_p95": asdict(raw_p95),
            "latency_p50_ratio": config["target_contract"]["p50_latency_ratio"],
            "latency_p95_ratio": config["target_contract"]["p95_latency_ratio"],
            "candidate_ratio_p95": config["target_contract"]["p95_candidate_ratio"],
        },
        "arms": arm_results,
        "integrity_failures": integrity_failures,
        "authoritative_decision": decision,
        "claim_boundary": config["claim_boundary"],
    }
    result = {
        **core,
        "deterministic_core_sha256": sha256_json(core),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "peak_rss_bytes": rss_bytes(),
            "wall_ns": time.perf_counter_ns() - started,
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip(),
        },
    }
    write_json(output_dir / "artifacts/deterministic_core.json", core)
    write_json(output_dir / "raw/case_rows.json", all_details)
    write_json(output_dir / "result.json", result)
    checks = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checks.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output_dir).as_posix()}"
            )
    (output_dir / "checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "same_family": arm_results["same_family"]["holdout_gate"],
        "cross_family_text_bridge": arm_results["cross_family_text_bridge"]["holdout_gate"],
        "integrity_failures": integrity_failures,
    }, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
