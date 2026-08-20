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
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.causal_suffix_action_cache import (  # noqa: E402
    RowwiseSymmetricQ4Head,
    SuffixActionInvariantError,
    bf16_roundtrip,
    canonical_sha256,
    longest_common_prefix,
    predict_suffix_action,
    project_target_hot_state,
    quantize_rowwise_symmetric_int4,
    select_predictor_mode,
    stable_argmax,
    stable_token_rank,
    summarize_ranks,
    synthetic_carry_control,
    synthetic_nearest_control,
    tensor_bytes,
    tensor_sha256,
    verification_weight_fraction,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    config_dict,
    model_tensor_sha256,
    module_parameter_bytes,
)
from vortex_runtime.prefix_state_bisimulation import cache_nbytes, cache_sha256  # noqa: E402


@dataclass
class PrefillState:
    case_id: str
    family: str
    prompt_sha256: str
    prompt_ids: tuple[int, ...]
    exact_first_token: int
    target_past_key_values: Any
    history_hidden_bf16: torch.Tensor
    history_residual_bf16: torch.Tensor
    history_prompt_positions: tuple[int, ...]
    prefill_logits_sha256: str
    shallow_logits_sha256: str
    history_hidden_sha256: str
    history_residual_sha256: str
    first_layer_cache_sha256: str
    prefill_target_calls: int


@dataclass
class CandidateTrace:
    mode: str
    tokens: list[int]
    history_indices: list[int | None]
    token_scores_sha256: list[str]
    first_hidden_sha256: str
    first_q4_logits_sha256: str
    candidate_sha256: str
    candidate_calls: int
    completed_ns: int


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
    lines: list[str] = []
    for relative in sorted(set(relative_paths)):
        lines.append(f"{sha256_file(output_dir / relative)}  {relative}")
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
        if expected[key] != actual[key]
    }
    if mismatch:
        raise RuntimeError(f"runtime pin mismatch: {mismatch}")
    return actual


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "exp-090a-causal-suffix-action-cache-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("experiment was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["shallow_depth"]) != 1:
        raise RuntimeError("EXP-090A freezes exactly one shallow decoder layer")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-090A freezes a 128-token draft block")
    if int(config["history_limit"]) != 64:
        raise RuntimeError("EXP-090A freezes a 64-entry verified history")
    expected_modes = [
        "raw_q4",
        "carry_bf16",
        "secant_bf16",
        "nearest_history_bf16",
    ]
    if list(config["predictor_modes"]) != expected_modes:
        raise RuntimeError("predictor language/order changed")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("case population is empty")
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case IDs")
    build = [case for case in cases if case["split"] == "build"]
    holdout = [case for case in cases if case["split"] == "holdout"]
    if len(build) < int(config["success"]["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(config["success"]["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    build_prompts = {sha256_bytes(case["prompt"].encode()) for case in build}
    holdout_prompts = {sha256_bytes(case["prompt"].encode()) for case in holdout}
    if build_prompts.intersection(holdout_prompts):
        raise RuntimeError("build and holdout prompts overlap")


def load_model(config: dict[str, Any]) -> tuple[Any, Any, int]:
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
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    raise RuntimeError(f"unsupported cache type: {type(past_key_values).__name__}")


def first_layer_cache_sha256(past_key_values: Any) -> str:
    layer = legacy_cache(past_key_values)[0]
    return canonical_sha256(
        {
            "key": tensor_sha256(layer[0]),
            "value": tensor_sha256(layer[1]),
            "key_shape": list(layer[0].shape),
            "value_shape": list(layer[1].shape),
            "dtype": str(layer[0].dtype),
        }
    )


def clone_first_layer_dynamic_cache(past_key_values: Any) -> DynamicCache:
    first = legacy_cache(past_key_values)[0]
    cache = DynamicCache()
    cache.key_cache = [first[0].detach().clone()]
    cache.value_cache = [first[1].detach().clone()]
    cache._seen_tokens = int(first[0].shape[-2])
    return cache


def checkpoint_identity(model: Any, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
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


def prefill_case(
    model: Any,
    tokenizer: Any,
    q4_head: RowwiseSymmetricQ4Head,
    case: Mapping[str, Any],
    *,
    history_limit: int,
    shallow_depth: int,
) -> PrefillState:
    input_ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"]
    if int(input_ids.shape[1]) < 2:
        raise RuntimeError(f"prompt {case['id']} has fewer than two tokens")
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        shallow_hidden = output.hidden_states[int(shallow_depth)]
        shallow_normed = model.model.norm(shallow_hidden)
        shallow_logits = q4_head.logits(shallow_normed)
    final_logits = output.logits.detach().to(torch.float32).cpu()
    shallow_logits = shallow_logits.detach().to(torch.float32).cpu()
    shallow_hidden = shallow_hidden.detach().to(torch.bfloat16).cpu()
    if final_logits.shape != shallow_logits.shape:
        raise RuntimeError("prefill final/shallow logit shapes differ")

    residual = (final_logits - shallow_logits).to(torch.bfloat16).contiguous()
    count = min(int(history_limit), int(input_ids.shape[1]))
    start = int(input_ids.shape[1]) - count
    history_hidden = shallow_hidden[0, start:, :].contiguous()
    history_residual = residual[0, start:, :].contiguous()
    if history_hidden.shape[0] != count or history_residual.shape[0] != count:
        raise RuntimeError("history extraction failed")

    return PrefillState(
        case_id=str(case["id"]),
        family=str(case["family"]),
        prompt_sha256=sha256_bytes(case["prompt"].encode("utf-8")),
        prompt_ids=tuple(int(value) for value in input_ids[0].tolist()),
        exact_first_token=stable_argmax(final_logits[0, -1]),
        target_past_key_values=output.past_key_values,
        history_hidden_bf16=history_hidden,
        history_residual_bf16=history_residual,
        history_prompt_positions=tuple(range(start, int(input_ids.shape[1]))),
        prefill_logits_sha256=tensor_sha256(final_logits),
        shallow_logits_sha256=tensor_sha256(shallow_logits),
        history_hidden_sha256=tensor_sha256(history_hidden),
        history_residual_sha256=tensor_sha256(history_residual),
        first_layer_cache_sha256=first_layer_cache_sha256(output.past_key_values),
        prefill_target_calls=1,
    )


def shallow_step(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    *,
    token: int,
    cache: DynamicCache,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_tensor = torch.tensor([[int(token)]], dtype=torch.long)
    cache_position = torch.tensor([int(position)], dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    with torch.inference_mode():
        hidden = model.model.embed_tokens(token_tensor)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        layer_output = model.model.layers[0](
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        shallow_hidden = layer_output[0]
        shallow_logits = q4_head.logits(model.model.norm(shallow_hidden))
    return (
        shallow_hidden[0, -1, :].detach().to(torch.bfloat16).cpu(),
        shallow_logits[0, -1, :].detach().to(torch.float32).cpu(),
    )


def generate_candidate(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    *,
    mode: str,
    block_length: int,
) -> CandidateTrace:
    cache = clone_first_layer_dynamic_cache(prefill.target_past_key_values)
    current_token = int(prefill.exact_first_token)
    position = len(prefill.prompt_ids)
    tokens: list[int] = []
    indices: list[int | None] = []
    score_hashes: list[str] = []
    first_hidden_hash = ""
    first_q4_hash = ""

    for step in range(int(block_length)):
        query_hidden, q4_logits = shallow_step(
            model,
            q4_head,
            token=current_token,
            cache=cache,
            position=position + step,
        )
        if step == 0:
            first_hidden_hash = tensor_sha256(query_hidden)
            first_q4_hash = tensor_sha256(q4_logits)
        correction, history_index = predict_suffix_action(
            mode,
            step_offset=step,
            query_hidden=query_hidden,
            history_hidden=prefill.history_hidden_bf16,
            history_residual=prefill.history_residual_bf16,
        )
        scores = q4_logits + correction
        token = stable_argmax(scores)
        tokens.append(token)
        indices.append(history_index)
        score_hashes.append(tensor_sha256(scores))
        current_token = token

    completed = time.perf_counter_ns()
    return CandidateTrace(
        mode=mode,
        tokens=tokens,
        history_indices=indices,
        token_scores_sha256=score_hashes,
        first_hidden_sha256=first_hidden_hash,
        first_q4_logits_sha256=first_q4_hash,
        candidate_sha256=canonical_sha256(tokens),
        candidate_calls=int(block_length),
        completed_ns=completed,
    )


def generate_target_and_diagnostics(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    *,
    modes: Sequence[str],
    block_length: int,
) -> tuple[list[int], dict[str, dict[str, Any]], dict[str, Any]]:
    target_start_ns = time.perf_counter_ns()
    past = prefill.target_past_key_values
    current_token = int(prefill.exact_first_token)
    target_tokens: list[int] = []
    diagnostics: dict[str, dict[str, Any]] = {
        str(mode): {
            "ranks": [],
            "top1_tokens": [],
            "history_indices": [],
            "score_sha256": [],
        }
        for mode in modes
    }
    first_hidden_sha256 = ""
    first_q4_logits_sha256 = ""

    for step in range(int(block_length)):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current_token]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            target_logits = output.logits[0, -1, :].detach().to(torch.float32).cpu()
            target_hidden = output.hidden_states[1][0, -1, :].detach().to(torch.bfloat16).cpu()
            q4_logits = q4_head.logits(
                model.model.norm(output.hidden_states[1][:, -1:, :])
            )[0, -1, :].detach().to(torch.float32).cpu()
        if step == 0:
            first_hidden_sha256 = tensor_sha256(target_hidden)
            first_q4_logits_sha256 = tensor_sha256(q4_logits)
        target_token = stable_argmax(target_logits)
        target_tokens.append(target_token)

        for mode in modes:
            correction, history_index = predict_suffix_action(
                str(mode),
                step_offset=step,
                query_hidden=target_hidden,
                history_hidden=prefill.history_hidden_bf16,
                history_residual=prefill.history_residual_bf16,
            )
            scores = q4_logits + correction
            diagnostics[str(mode)]["ranks"].append(
                stable_token_rank(scores, target_token)
            )
            diagnostics[str(mode)]["top1_tokens"].append(stable_argmax(scores))
            diagnostics[str(mode)]["history_indices"].append(history_index)
            diagnostics[str(mode)]["score_sha256"].append(tensor_sha256(scores))

        current_token = target_token
        past = output.past_key_values

    return target_tokens, diagnostics, {
        "target_start_ns": target_start_ns,
        "target_calls": int(block_length),
        "first_hidden_sha256": first_hidden_sha256,
        "first_q4_logits_sha256": first_q4_logits_sha256,
        "final_cache_sha256": cache_sha256(past),
        "final_cache_bytes": cache_nbytes(past),
    }


def row_for_mode(
    prefill: PrefillState,
    candidate: CandidateTrace,
    target_tokens: Sequence[int],
    diagnostics: Mapping[str, Any],
    target_meta: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = longest_common_prefix(candidate.tokens, target_tokens)
    first_mismatch = None if accepted == len(target_tokens) else accepted
    ranks = [int(value) for value in diagnostics["ranks"]]
    top1_tokens = [int(value) for value in diagnostics["top1_tokens"]]
    rank_summary = summarize_ranks(ranks)
    return {
        "case_id": prefill.case_id,
        "family": prefill.family,
        "mode": candidate.mode,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "history_count": int(prefill.history_hidden_bf16.shape[0]),
        "history_hidden_sha256": prefill.history_hidden_sha256,
        "history_residual_sha256": prefill.history_residual_sha256,
        "exact_first_token": prefill.exact_first_token,
        "candidate_tokens": candidate.tokens,
        "target_tokens": list(map(int, target_tokens)),
        "candidate_sha256": candidate.candidate_sha256,
        "target_sha256": canonical_sha256(list(map(int, target_tokens))),
        "accepted_length": accepted,
        "first_mismatch_position": first_mismatch,
        "true_path_ranks": ranks,
        "true_path_rank_summary": rank_summary,
        "true_path_top1_tokens": top1_tokens,
        "candidate_history_indices": candidate.history_indices,
        "true_path_history_indices": diagnostics["history_indices"],
        "candidate_score_sha256": candidate.token_scores_sha256,
        "true_path_score_sha256": diagnostics["score_sha256"],
        "candidate_calls": candidate.candidate_calls,
        "target_calls": int(target_meta["target_calls"]),
        "candidate_completed_before_target": candidate.completed_ns
        <= int(target_meta["target_start_ns"]),
        "manual_first_hidden_matches_official": candidate.first_hidden_sha256
        == target_meta["first_hidden_sha256"],
        "manual_first_q4_logits_matches_official": candidate.first_q4_logits_sha256
        == target_meta["first_q4_logits_sha256"],
        "logical_weight_fraction_no_compression": verification_weight_fraction(accepted),
    }


def run_case_modes(
    model: Any,
    tokenizer: Any,
    q4_head: RowwiseSymmetricQ4Head,
    case: Mapping[str, Any],
    *,
    modes: Sequence[str],
    history_limit: int,
    shallow_depth: int,
    block_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefill = prefill_case(
        model,
        tokenizer,
        q4_head,
        case,
        history_limit=history_limit,
        shallow_depth=shallow_depth,
    )
    candidates = {
        str(mode): generate_candidate(
            model,
            q4_head,
            prefill,
            mode=str(mode),
            block_length=block_length,
        )
        for mode in modes
    }
    # The target continuation is intentionally delayed until every requested
    # candidate chain is complete and hashed.
    target_tokens, diagnostics, target_meta = generate_target_and_diagnostics(
        model,
        q4_head,
        prefill,
        modes=modes,
        block_length=block_length,
    )
    rows = [
        row_for_mode(
            prefill,
            candidates[str(mode)],
            target_tokens,
            diagnostics[str(mode)],
            target_meta,
        )
        for mode in modes
    ]
    case_trace = {
        "case_id": prefill.case_id,
        "family": prefill.family,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "history_count": int(prefill.history_hidden_bf16.shape[0]),
        "history_prompt_positions": list(prefill.history_prompt_positions),
        "prefill_logits_sha256": prefill.prefill_logits_sha256,
        "shallow_logits_sha256": prefill.shallow_logits_sha256,
        "history_hidden_sha256": prefill.history_hidden_sha256,
        "history_residual_sha256": prefill.history_residual_sha256,
        "first_layer_cache_sha256": prefill.first_layer_cache_sha256,
        "exact_first_token": prefill.exact_first_token,
        "target_final_cache_sha256": target_meta["final_cache_sha256"],
        "target_final_cache_bytes": target_meta["final_cache_bytes"],
        "prefill_target_calls": prefill.prefill_target_calls,
        "continuation_target_calls": target_meta["target_calls"],
    }
    return rows, case_trace


def aggregate_rows(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["mode"]) == str(mode)]
    if not selected:
        raise RuntimeError(f"no rows for mode {mode}")
    accepted = [int(row["accepted_length"]) for row in selected]
    ranks: list[int] = []
    for row in selected:
        ranks.extend(int(value) for value in row["true_path_ranks"])
    return {
        "mode": str(mode),
        "case_count": len(selected),
        "accepted_lengths": accepted,
        "accepted_min": min(accepted),
        "accepted_p50": float(np.percentile(accepted, 50, method="linear")),
        "accepted_max": max(accepted),
        "accepted_total": sum(accepted),
        "rank": summarize_ranks(ranks),
        "all_candidates_precede_target": all(
            bool(row["candidate_completed_before_target"]) for row in selected
        ),
        "manual_first_layer_exact": all(
            bool(row["manual_first_hidden_matches_official"])
            and bool(row["manual_first_q4_logits_matches_official"])
            for row in selected
        ),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)

    model, tokenizer, model_load_ns = load_model(config)
    if len(model.model.layers) <= int(config["shallow_depth"]):
        raise RuntimeError("DEV-W does not contain the frozen shallow depth")
    q4_start = time.perf_counter_ns()
    q4_head = quantize_rowwise_symmetric_int4(
        model.lm_head.weight,
        minimum=int(config["q4_head"]["minimum"]),
        maximum=int(config["q4_head"]["maximum"]),
    )
    q4_compile_ns = time.perf_counter_ns() - q4_start
    identity = checkpoint_identity(model, tokenizer, config)
    target_hot = project_target_hot_state(config["target_projection"])

    controls = {
        "carry": synthetic_carry_control(),
        "nearest": synthetic_nearest_control(),
        "q4_rows_match_vocab": q4_head.manifest.rows == int(model.config.vocab_size),
        "q4_cols_match_hidden": q4_head.manifest.cols == int(model.config.hidden_size),
        "q4_artifact_equation": q4_head.manifest.artifact_bytes
        == q4_head.manifest.packed_weight_bytes + q4_head.manifest.scale_bytes,
        "target_projection_nonnegative": target_hot.margin_bytes >= 0,
    }
    controls["passed"] = all(
        [
            bool(controls["carry"]["passed"]),
            bool(controls["nearest"]["passed"]),
            bool(controls["q4_rows_match_vocab"]),
            bool(controls["q4_cols_match_hidden"]),
            bool(controls["q4_artifact_equation"]),
            bool(controls["target_projection_nonnegative"]),
        ]
    )

    build_cases = [case for case in config["cases"] if case["split"] == "build"]
    holdout_cases = [case for case in config["cases"] if case["split"] == "holdout"]
    modes = list(map(str, config["predictor_modes"]))
    build_rows: list[dict[str, Any]] = []
    case_traces: list[dict[str, Any]] = []
    run_start = time.perf_counter_ns()

    for case in build_cases:
        rows, trace = run_case_modes(
            model,
            tokenizer,
            q4_head,
            case,
            modes=modes,
            history_limit=int(config["history_limit"]),
            shallow_depth=int(config["shallow_depth"]),
            block_length=int(config["block_length"]),
        )
        build_rows.extend(rows)
        case_traces.append({**trace, "split": "build"})
        print(
            f"build {case['id']}: "
            + ", ".join(
                f"{row['mode']}={row['accepted_length']}" for row in rows
            ),
            flush=True,
        )

    selection = select_predictor_mode(build_rows, modes)
    selection_frozen_ns = time.perf_counter_ns()
    selected_mode = selection.mode
    holdout_rows: list[dict[str, Any]] = []
    holdout_target_began_after_selection = True

    for case in holdout_cases:
        before = time.perf_counter_ns()
        rows, trace = run_case_modes(
            model,
            tokenizer,
            q4_head,
            case,
            modes=[selected_mode],
            history_limit=int(config["history_limit"]),
            shallow_depth=int(config["shallow_depth"]),
            block_length=int(config["block_length"]),
        )
        holdout_target_began_after_selection = (
            holdout_target_began_after_selection and before >= selection_frozen_ns
        )
        holdout_rows.extend(rows)
        case_traces.append({**trace, "split": "holdout"})
        print(
            f"holdout {case['id']}: {selected_mode}={rows[0]['accepted_length']}",
            flush=True,
        )

    run_ns = time.perf_counter_ns() - run_start
    build_aggregates = {mode: aggregate_rows(build_rows, mode) for mode in modes}
    holdout_aggregate = aggregate_rows(holdout_rows, selected_mode)
    holdout_min = int(holdout_aggregate["accepted_min"])
    no_compression_fraction = verification_weight_fraction(holdout_min)
    favorable_fraction = verification_weight_fraction(
        holdout_min,
        compression_ratio=float(
            config["target_projection"]["checkpoint_compression_ratio_favorable"]
        ),
    )

    integrity_failures: list[str] = []
    if not controls["passed"]:
        integrity_failures.append("model_free_control_failure")
    if not all(row["candidate_completed_before_target"] for row in build_rows + holdout_rows):
        integrity_failures.append("candidate_target_order_failure")
    if not all(
        row["manual_first_hidden_matches_official"]
        and row["manual_first_q4_logits_matches_official"]
        for row in build_rows + holdout_rows
    ):
        integrity_failures.append("manual_shallow_reference_mismatch")
    if not holdout_target_began_after_selection:
        integrity_failures.append("holdout_observed_before_selection")
    if any(len(row["candidate_tokens"]) != int(config["block_length"]) for row in build_rows + holdout_rows):
        integrity_failures.append("candidate_length_failure")
    if any(len(row["target_tokens"]) != int(config["block_length"]) for row in build_rows + holdout_rows):
        integrity_failures.append("target_length_failure")
    if identity["resolved_revision"] not in {"", config["revision"]}:
        integrity_failures.append("checkpoint_revision_mismatch")

    success = config["success"]
    integrity_passed = len(integrity_failures) <= int(
        success["maximum_integrity_mismatches"]
    )
    chain_signal = (
        integrity_passed
        and holdout_min >= int(success["chain_required_accepted_length"])
        and target_hot.hot_limit_passed
        and no_compression_fraction
        <= float(config["target_projection"]["whole_model_fraction_limit"])
    )
    rank_summary = holdout_aggregate["rank"]
    tree_signal = (
        integrity_passed
        and not chain_signal
        and float(rank_summary["p95"]) <= float(success["tree_rank_p95_max"])
        and int(rank_summary["max"]) <= int(success["tree_rank_max_max"])
    )

    if not integrity_passed:
        decision = "INVALID_CAUSAL_SUFFIX_ACTION_CACHE_CONTROL_FAILURE"
    elif chain_signal:
        decision = "PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_EXACT_BLOCK_EXECUTOR_GATE"
    elif tree_signal:
        decision = "PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_CANDIDATE_TREE_GATE"
    else:
        decision = "REJECT_CAUSAL_SUFFIX_ACTION_CACHE_AS_128_TOKEN_DRAFT_CORE"

    total_target_calls = sum(
        int(trace["prefill_target_calls"]) + int(trace["continuation_target_calls"])
        for trace in case_traces
    )
    total_candidate_calls = sum(
        int(row["candidate_calls"]) for row in build_rows + holdout_rows
    )
    dev_layer_parameter_bytes = module_parameter_bytes(model.model.layers[0])
    dev_history_bytes_max = int(config["history_limit"]) * (
        int(model.config.hidden_size) + int(model.config.vocab_size)
    ) * 2

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": "one-official-bf16-layer/rowwise-q4-logit-lens/frozen-verified-bf16-suffix-action-history/raw-carry-secant-nearest/128-token-chain/build-selection/untouched-holdout",
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "q4_head": q4_head.manifest.to_dict(),
        "controls": controls,
        "build_rows": build_rows,
        "build_aggregates": build_aggregates,
        "selection": selection.to_dict(),
        "holdout_rows": holdout_rows,
        "holdout_aggregate": holdout_aggregate,
        "case_traces": case_traces,
        "target_projection": {
            **target_hot.to_dict(),
            "model_id": config["target_projection"]["model_id"],
            "revision": config["target_projection"]["revision"],
            "verification_fraction_no_compression": no_compression_fraction,
            "verification_fraction_favorable_compression": favorable_fraction,
            "whole_model_fraction_limit": config["target_projection"][
                "whole_model_fraction_limit"
            ],
            "arithmetic_fraction": 1.0,
        },
        "dev_resource": {
            "first_layer_parameter_bytes": dev_layer_parameter_bytes,
            "q4_head_artifact_bytes": q4_head.manifest.artifact_bytes,
            "maximum_history_bytes": dev_history_bytes_max,
            "target_calls": total_target_calls,
            "candidate_first_layer_calls": total_candidate_calls,
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "integrity_failures": integrity_failures,
            "holdout_target_began_after_selection": holdout_target_began_after_selection,
            "selected_mode": selected_mode,
            "holdout_minimum_accepted_length": holdout_min,
            "chain_signal": chain_signal,
            "tree_signal": tree_signal,
            "hot_projection_passed": target_hot.hot_limit_passed,
            "logical_weight_fraction_passed": no_compression_fraction
            <= float(config["target_projection"]["whole_model_fraction_limit"]),
            "verification_arithmetic_reduced": False,
        },
        "authoritative_decision": decision,
    }

    result = {
        **deterministic_core,
        "experiment": "EXP-090A",
        "name": "causal_suffix_action_cache_gate",
        "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E3_CAUSAL_HELDOUT_DRAFT_SIGNAL",
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "q4_compile_ns": q4_compile_ns,
            "experiment_ns": run_ns,
            "selection_frozen_ns": selection_frozen_ns,
        },
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "claim_boundary": {
            "official_public_checkpoint_loaded": True,
            "candidate_generated_before_target": integrity_passed,
            "holdout_mode_frozen_before_target": holdout_target_began_after_selection,
            "wrong_draft_changes_committed_output": False,
            "physical_exact_block_verifier": "NOT_TESTED",
            "packed_q4_kernel": "NOT_TESTED",
            "target_405b_checkpoint": "NOT_TESTED",
            "target_8gib_residency": "PROJECTED_NOT_MEASURED",
            "target_arithmetic_speed": "NOT_TESTED",
            "same_machine_4b_p50_p95": "NOT_TESTED",
        },
        "provenance": {
            "MEASURED": [
                "DEV-W causal candidate tokens, exact target tokens, accepted prefixes, true-path target ranks, target/candidate calls, Q4 simulation"
            ],
            "DERIVED": [
                "DEV-W aggregates, logical verification fractions from accepted lengths, exact target hot-state equation from frozen dimensions"
            ],
            "PROJECTED": [
                "Llama-3.1-405B one-layer/Q4-head/history/KV residency and whole-checkpoint verification fractions"
            ],
            "UNVERIFIED": [
                "actual 405B acceptance, packed draft kernel, exact block verifier, target arithmetic throughput and latency"
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(output_dir / "artifacts/q4_head_manifest.json", q4_head.manifest.to_dict())
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/build_rows.json", build_rows)
    write_json(output_dir / "raw/holdout_rows.json", holdout_rows)
    write_json(output_dir / "raw/case_traces.json", case_traces)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/q4_head_manifest.json",
            "artifacts/deterministic_core.json",
            "raw/build_rows.json",
            "raw/holdout_rows.json",
            "raw/case_traces.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "selected_mode": selected_mode,
                "build_minimum_accepted_length": selection.minimum_accepted_length,
                "holdout_accepted_lengths": holdout_aggregate["accepted_lengths"],
                "holdout_rank_p95": rank_summary["p95"],
                "holdout_rank_max": rank_summary["max"],
                "projected_hot_gib": target_hot.total_hot_gib,
                "verification_fraction_no_compression": no_compression_fraction,
                "verification_arithmetic_fraction": 1.0,
            },
            indent=2,
            sort_keys=True,
        )
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
