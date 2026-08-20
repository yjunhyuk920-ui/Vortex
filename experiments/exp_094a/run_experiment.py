from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from vortex_runtime.causal_suffix_action_cache import (  # noqa: E402
    RowwiseSymmetricQ4Head,
    project_target_hot_state,
    quantize_rowwise_symmetric_int4,
)
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
    module_parameter_bytes,
)
from vortex_runtime.one_sweep_true_token_rank import (  # noqa: E402
    canonical_sha256,
    rank_histogram,
    rank_metrics,
    stable_true_token_rank,
    tensor_sha256,
)
from vortex_runtime.parareal_residual_transport import (  # noqa: E402
    logical_source_fraction,
    model_free_controls,
    rank_true_token,
    residual_buffer_bytes,
    stable_argmax64,
    summarize_transport,
    transport_scores,
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
    if config.get("schema") != "exp-094a-parareal-residual-transport-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("configuration was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["shallow_depth"]) != 1:
        raise RuntimeError("EXP-094A freezes one coarse decoder layer")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-094A freezes K=128")
    if str(config["seed_mode"]) != "prompt_suffix_cycle_16":
        raise RuntimeError("seed changed from the frozen EXP-091A selection")
    if str(config["residual_dtype"]) != "float64":
        raise RuntimeError("residual working dtype changed")
    q4 = config["q4_head"]
    if q4["scheme"] != "rowwise_symmetric_int4":
        raise RuntimeError("Q4 head scheme changed")
    if int(q4["minimum"]) != -8 or int(q4["maximum"]) != 7:
        raise RuntimeError("Q4 head range changed")

    target = config["target_projection"]
    if target["model_id"] != TARGET_MODEL_ID or target["revision"] != TARGET_REVISION:
        raise RuntimeError("TARGET-W identity mismatch")
    if int(target["history_entries"]) != 1:
        raise RuntimeError("Parareal projection retains one boundary history entry")
    success = config["success"]
    if int(success["chain_required_accepted_length"]) != 128:
        raise RuntimeError("exact chain threshold changed")
    if int(success["tree_rank_p95_max"]) != 4 or int(success["tree_rank_max_max"]) != 16:
        raise RuntimeError("candidate-tree thresholds changed")

    cases = list(config["cases"])
    build = [case for case in cases if case["split"] == "build"]
    holdout = [case for case in cases if case["split"] == "holdout"]
    if len(build) < int(success["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(success["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case IDs")
    prompts = [sha256_bytes(case["prompt"].encode("utf-8")) for case in cases]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("duplicate prompt hashes")


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


def fresh_full_cache(prefix_cache: tuple[Any, ...]) -> DynamicCache:
    return DynamicCache.from_legacy_cache(prefix_cache)


def fresh_first_layer_cache(prefix_cache: tuple[Any, ...]) -> DynamicCache:
    key, value = prefix_cache[0]
    cache = DynamicCache()
    cache.key_cache = [key.detach().clone()]
    cache.value_cache = [value.detach().clone()]
    cache._seen_tokens = int(key.shape[-2])
    return cache


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


def prefill_case(model: Any, tokenizer: Any, case: Mapping[str, Any]) -> PrefillState:
    input_ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"]
    if int(input_ids.shape[1]) < 2:
        raise RuntimeError(f"prompt {case['id']} has fewer than two tokens")
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


def shallow_step(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    *,
    token: int,
    cache: DynamicCache,
    position: int,
) -> torch.Tensor:
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
        logits = q4_head.logits(model.model.norm(shallow_hidden))
    return logits[0, -1, :].detach().to(torch.float32).contiguous().cpu()


def run_fine_guessed_sweep(
    model: Any,
    prefill: PrefillState,
    input_tokens: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    cache = fresh_full_cache(prefill.prefix_cache)
    started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([list(map(int, input_tokens))], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    completed_ns = time.perf_counter_ns()
    logits = output.logits[0].detach().to(torch.float32).contiguous().cpu()
    if logits.ndim != 2 or int(logits.shape[0]) != len(input_tokens):
        raise RuntimeError(f"unexpected fine-sweep shape: {tuple(logits.shape)}")
    proposals = stable_argmax_rows(logits).tolist()
    return logits, {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "input_sha256": canonical_sha256(input_tokens),
        "logits_sha256": tensor_sha256(logits),
        "proposal_sha256": canonical_sha256(proposals),
        "first_proposal_token": int(proposals[0]),
        "post_sweep_cache_sha256": cache_sha256(output.past_key_values),
        "post_sweep_cache_bytes": cache_nbytes(output.past_key_values),
        "target_information_used": False,
    }


def run_coarse_old_path(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    input_tokens: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    cache = fresh_first_layer_cache(prefill.prefix_cache)
    rows: list[torch.Tensor] = []
    started_ns = time.perf_counter_ns()
    for offset, token in enumerate(input_tokens):
        rows.append(
            shallow_step(
                model,
                q4_head,
                token=int(token),
                cache=cache,
                position=len(prefill.prompt_ids) + offset,
            )
        )
    completed_ns = time.perf_counter_ns()
    logits = torch.stack(rows, dim=0).contiguous()
    return logits, {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "calls": len(rows),
        "logits_sha256": tensor_sha256(logits),
        "first_logits_sha256": tensor_sha256(logits[0]),
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
    }


def run_transport_candidate(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    fine_old: torch.Tensor,
    coarse_old: torch.Tensor,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if fine_old.shape != coarse_old.shape:
        raise RuntimeError("fine/coarse-old blocks differ")
    cache = fresh_first_layer_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    score_hashes: list[str] = []
    first_coarse_hash = ""
    first_score_hash = ""
    started_ns = time.perf_counter_ns()
    for position in range(int(fine_old.shape[0])):
        coarse_new = shallow_step(
            model,
            q4_head,
            token=current,
            cache=cache,
            position=len(prefill.prompt_ids) + position,
        )
        scores = transport_scores(fine_old[position], coarse_old[position], coarse_new)
        if position == 0:
            first_coarse_hash = tensor_sha256(coarse_new)
            first_score_hash = tensor_sha256(scores)
        token = stable_argmax64(scores)
        tokens.append(token)
        score_hashes.append(tensor_sha256(scores))
        current = token
    completed_ns = time.perf_counter_ns()
    return tuple(tokens), {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "calls": len(tokens),
        "candidate_sha256": canonical_sha256(tokens),
        "score_sha256": canonical_sha256(score_hashes),
        "first_coarse_logits_sha256": first_coarse_hash,
        "first_transport_scores_sha256": first_score_hash,
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
        "target_information_used": False,
    }


def run_target_diagnostics(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    fine_old: torch.Tensor,
    coarse_old: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    target_cache: Any = fresh_full_cache(prefill.prefix_cache)
    true_coarse_cache = fresh_first_layer_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    transported_ranks: list[int] = []
    static_fine_ranks: list[int] = []
    transported_top1: list[int] = []
    target_logit_hashes: list[str] = []
    transported_score_hashes: list[str] = []
    started_ns = time.perf_counter_ns()
    for position in range(int(fine_old.shape[0])):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
        target_logits = output.logits[0, -1].detach().to(torch.float32).contiguous().cpu()
        true_token = int(stable_argmax_rows(target_logits.unsqueeze(0))[0].item())
        coarse_true = shallow_step(
            model,
            q4_head,
            token=current,
            cache=true_coarse_cache,
            position=len(prefill.prompt_ids) + position,
        )
        transported = transport_scores(
            fine_old[position], coarse_old[position], coarse_true
        )
        tokens.append(true_token)
        transported_ranks.append(rank_true_token(transported, true_token))
        static_fine_ranks.append(
            stable_true_token_rank(fine_old[position], true_token)
        )
        transported_top1.append(stable_argmax64(transported))
        target_logit_hashes.append(tensor_sha256(target_logits))
        transported_score_hashes.append(tensor_sha256(transported))
        current = true_token
        target_cache = output.past_key_values
    completed_ns = time.perf_counter_ns()
    return (
        tuple(tokens),
        tuple(transported_ranks),
        tuple(static_fine_ranks),
        {
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "wall_ns": completed_ns - started_ns,
            "target_calls": len(tokens),
            "true_coarse_calls": len(tokens),
            "target_sha256": canonical_sha256(tokens),
            "transported_top1_sha256": canonical_sha256(transported_top1),
            "target_logits_sha256": canonical_sha256(target_logit_hashes),
            "transported_scores_sha256": canonical_sha256(
                transported_score_hashes
            ),
            "final_target_cache_sha256": cache_sha256(target_cache),
            "final_target_cache_bytes": cache_nbytes(target_cache),
            "final_true_coarse_cache_sha256": cache_sha256(true_coarse_cache),
            "final_true_coarse_cache_bytes": cache_nbytes(true_coarse_cache),
        },
    )


def run_case(
    model: Any,
    tokenizer: Any,
    q4_head: RowwiseSymmetricQ4Head,
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
    input_tokens = jacobi_input(prefill.boundary_token, seed)
    fine_old, fine_meta = run_fine_guessed_sweep(model, prefill, input_tokens)
    coarse_old, coarse_meta = run_coarse_old_path(
        model, q4_head, prefill, input_tokens
    )
    residual = (fine_old.to(torch.float64) - coarse_old.to(torch.float64)).contiguous()
    candidate, candidate_meta = run_transport_candidate(
        model, q4_head, prefill, fine_old, coarse_old
    )
    candidate_completed_ns = int(candidate_meta["completed_ns"])
    target, transported_ranks, static_ranks, target_meta = run_target_diagnostics(
        model, q4_head, prefill, fine_old, coarse_old
    )
    metrics = summarize_transport(candidate, target, transported_ranks)
    static_metrics = rank_metrics(static_ranks)
    first_controls = {
        "coarse_old_equals_candidate_coarse": coarse_meta[
            "first_logits_sha256"
        ]
        == candidate_meta["first_coarse_logits_sha256"],
        "fine_first_token_equals_target": int(fine_meta["first_proposal_token"])
        == int(target[0]),
        "candidate_first_token_equals_target": int(candidate[0]) == int(target[0]),
        "transported_true_rank_one": int(transported_ranks[0]) == 1,
        "static_true_rank_one": int(static_ranks[0]) == 1,
    }
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
        "fine_sweep": fine_meta,
        "coarse_old": coarse_meta,
        "residual_sha256": tensor_sha256(residual),
        "residual_bytes": int(residual.numel() * residual.element_size()),
        "candidate": candidate_meta,
        "target": target_meta,
        "candidate_completed_before_target": candidate_completed_ns
        <= int(target_meta["started_ns"]),
        "first_position_controls": first_controls,
        "candidate_target_prefix": metrics.accepted_prefix,
        "candidate_target_exact_positions": metrics.exact_positions,
        "candidate_target_exact_fraction": metrics.exact_fraction,
        "candidate_sha256": canonical_sha256(candidate),
        "target_sha256": canonical_sha256(target),
        "transported_true_path_ranks": list(transported_ranks),
        "transported_rank_histogram": rank_histogram(transported_ranks),
        "transport_metrics": metrics.to_dict(),
        "static_fine_true_path_ranks": list(static_ranks),
        "static_fine_rank_metrics": static_metrics.to_dict(),
        "rank_p95_improvement": static_metrics.p95 - metrics.rank_p95,
        "top16_fraction_improvement": metrics.top16_fraction
        - static_metrics.top16_fraction,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == str(split)]
    if not selected:
        raise RuntimeError(f"empty aggregate: {split}")
    accepted = [int(row["candidate_target_prefix"]) for row in selected]
    exact_positions = [
        int(row["candidate_target_exact_positions"]) for row in selected
    ]
    transported_ranks = [
        int(value)
        for row in selected
        for value in row["transported_true_path_ranks"]
    ]
    static_ranks = [
        int(value)
        for row in selected
        for value in row["static_fine_true_path_ranks"]
    ]
    transported = rank_metrics(transported_ranks)
    static = rank_metrics(static_ranks)
    return {
        "split": split,
        "case_count": len(selected),
        "position_count": len(transported_ranks),
        "accepted_lengths": accepted,
        "accepted_min": min(accepted),
        "accepted_p50": float(
            np.percentile(accepted, 50, method="linear")
        ),
        "accepted_max": max(accepted),
        "exact_positions": exact_positions,
        "exact_positions_p50": float(
            np.percentile(exact_positions, 50, method="linear")
        ),
        "transported_rank": transported.to_dict(),
        "transported_rank_histogram": rank_histogram(transported_ranks),
        "static_fine_rank": static.to_dict(),
        "rank_p95_improvement": static.p95 - transported.p95,
        "top16_fraction_improvement": transported.top16_fraction
        - static.top16_fraction,
        "all_candidates_precede_target": all(
            bool(row["candidate_completed_before_target"]) for row in selected
        ),
        "all_first_position_controls": all(
            all(bool(value) for value in row["first_position_controls"].values())
            for row in selected
        ),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    controls = model_free_controls()
    if not controls["passed"]:
        raise RuntimeError("model-free Parareal controls failed")

    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_load_ns = load_model(config)
    q4_started_ns = time.perf_counter_ns()
    q4_head = quantize_rowwise_symmetric_int4(
        model.lm_head.weight,
        minimum=int(config["q4_head"]["minimum"]),
        maximum=int(config["q4_head"]["maximum"]),
    )
    q4_compile_ns = time.perf_counter_ns() - q4_started_ns
    identity = checkpoint_identity(model, tokenizer, config)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(output_dir / "artifacts/q4_head_manifest.json", q4_head.manifest.to_dict())

    rows: list[dict[str, Any]] = []
    experiment_started_ns = time.perf_counter_ns()
    for case in config["cases"]:
        row = run_case(model, tokenizer, q4_head, case, config)
        rows.append(row)
        metrics = row["transport_metrics"]
        print(
            f"{row['split']} {row['case_id']}: "
            f"accepted={metrics['accepted_prefix']} exact={metrics['exact_positions']} "
            f"rank p50/p95/max={metrics['rank_p50']}/{metrics['rank_p95']}/{metrics['rank_maximum']} "
            f"top16={metrics['top16_fraction']:.6%}",
            flush=True,
        )
    experiment_ns = time.perf_counter_ns() - experiment_started_ns

    build = aggregate(rows, "build")
    holdout = aggregate(rows, "holdout")
    integrity_failures: list[str] = []
    if not controls["passed"]:
        integrity_failures.append("model_free_control_failure")
    if not all(bool(row["candidate_completed_before_target"]) for row in rows):
        integrity_failures.append("candidate_target_order_failure")
    if not all(
        all(bool(value) for value in row["first_position_controls"].values())
        for row in rows
    ):
        integrity_failures.append("first_position_control_failure")
    if not all(
        int(row["fine_sweep"]["target_information_used"]) == 0
        and int(row["candidate"]["target_information_used"]) == 0
        for row in rows
    ):
        integrity_failures.append("target_information_leak")
    if not all(
        int(row["target"]["target_calls"]) == int(config["block_length"])
        and int(row["candidate"]["calls"]) == int(config["block_length"])
        and int(row["coarse_old"]["calls"]) == int(config["block_length"])
        for row in rows
    ):
        integrity_failures.append("incomplete_call_trace")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")
    integrity_passed = len(integrity_failures) <= int(
        config["success"]["maximum_integrity_mismatches"]
    )

    all_rows_min_accepted = min(
        int(row["candidate_target_prefix"]) for row in rows
    )
    chain_signal = (
        integrity_passed
        and all_rows_min_accepted
        >= int(config["success"]["chain_required_accepted_length"])
    )
    tree_signal = (
        integrity_passed
        and not chain_signal
        and float(build["transported_rank"]["p95"])
        <= float(config["success"]["tree_rank_p95_max"])
        and int(build["transported_rank"]["maximum"])
        <= int(config["success"]["tree_rank_max_max"])
        and float(holdout["transported_rank"]["p95"])
        <= float(config["success"]["tree_rank_p95_max"])
        and int(holdout["transported_rank"]["maximum"])
        <= int(config["success"]["tree_rank_max_max"])
    )
    if not integrity_passed:
        decision = "INVALID_PARAREAL_RESIDUAL_TRANSPORT_CONTROL_FAILURE"
    elif chain_signal:
        decision = "PROMOTE_PARAREAL_TRANSPORT_TO_EXACT_TOKEN_CERTIFICATE_GATE"
    elif tree_signal:
        decision = "PROMOTE_PARAREAL_TRANSPORT_TO_BRANCH_SUCCESSOR_GATE"
    else:
        decision = "REJECT_PARAREAL_RESIDUAL_TRANSPORT_AS_405B_BLOCK_SOURCE"

    target_hot_base = project_target_hot_state(config["target_projection"])
    target_residual_bytes = residual_buffer_bytes(
        block_length=int(config["block_length"]),
        vocabulary_size=int(config["target_projection"]["vocab_size"]),
        value_bytes=8,
    )
    total_hot = int(target_hot_base.total_hot_bytes) + target_residual_bytes
    hot_limit = int(config["target_projection"]["hot_limit_bytes"])
    compression = float(
        config["target_projection"]["checkpoint_compression_ratio_favorable"]
    )
    holdout_min = int(holdout["accepted_min"])
    layer_parameters = int(target_hot_base.layer_parameter_count)
    target_head_parameters = int(
        config["target_projection"]["vocab_size"]
    ) * int(config["target_projection"]["hidden_size"])
    coarse_parameter_fraction = (
        layer_parameters + target_head_parameters
    ) / 405_000_000_000.0

    target_projection = {
        "base_hot_state": target_hot_base.to_dict(),
        "float64_residual_buffer_bytes": target_residual_bytes,
        "total_hot_bytes": total_hot,
        "total_hot_gib": total_hot / float(1 << 30),
        "hot_limit_bytes": hot_limit,
        "hot_margin_bytes": hot_limit - total_hot,
        "hot_limit_passed": total_hot <= hot_limit,
        "one_fine_sweep_fraction_at_full_block_raw": 1.0
        / int(config["block_length"]),
        "one_fine_sweep_fraction_at_full_block_compressed": 1.0
        / (int(config["block_length"]) * compression),
        "one_fine_sweep_fraction_at_observed_holdout_min_raw": logical_source_fraction(
            holdout_min
        ),
        "one_fine_sweep_fraction_at_observed_holdout_min_compressed": logical_source_fraction(
            holdout_min, compression_ratio=compression
        ),
        "coarse_parameter_equivalent_fraction_of_405b": coarse_parameter_fraction,
        "fine_dense_arithmetic_fraction_per_token": 1.0,
        "candidate_certificate": "NOT_CONSTRUCTED",
    }

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "one-official-k128-fine-guessed-sweep/one-official-first-layer-q4-head-coarse/"
            "position-aligned-float64-fine-minus-coarse-residual/"
            "causal-corrected-prefix-coarse-plus-frozen-residual/"
            "delayed-incremental-target/build-and-untouched-holdout"
        ),
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "q4_head": q4_head.manifest.to_dict(),
        "controls": {
            "model_free": controls,
            "integrity_failures": integrity_failures,
            "integrity_passed": integrity_passed,
        },
        "case_rows": rows,
        "build_aggregate": build,
        "holdout_aggregate": holdout,
        "target_projection": target_projection,
        "dev_resource": {
            "first_layer_parameter_bytes": module_parameter_bytes(
                model.model.layers[0]
            ),
            "q4_head_artifact_bytes": q4_head.manifest.artifact_bytes,
            "residual_buffer_bytes": residual_buffer_bytes(
                block_length=int(config["block_length"]),
                vocabulary_size=int(model.config.vocab_size),
                value_bytes=8,
            ),
            "fine_block_calls": len(rows),
            "coarse_old_calls": len(rows) * int(config["block_length"]),
            "candidate_coarse_calls": len(rows)
            * int(config["block_length"]),
            "true_path_coarse_calls": len(rows)
            * int(config["block_length"]),
            "incremental_target_calls": len(rows)
            * int(config["block_length"]),
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "all_rows_minimum_accepted_length": all_rows_min_accepted,
            "holdout_minimum_accepted_length": holdout_min,
            "chain_signal": chain_signal,
            "tree_signal": tree_signal,
            "hot_projection_passed": total_hot <= hot_limit,
            "certificate_constructed": False,
        },
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "q4_compile_ns": q4_compile_ns,
            "experiment_ns": experiment_ns,
        },
        "provenance": {
            "measured": [
                "DEV-W fine guessed-block logits, sequential coarse old/new/true logits, transported candidate tokens, delayed official target tokens/caches, true-path ranks"
            ],
            "derived": [
                "Parareal residuals, acceptance/rank aggregates, target hot-state and logical-traffic equations"
            ],
            "granted": [
                "float64 residual transport and candidate scoring are treated as exact working arithmetic"
            ],
            "unverified": [
                "sound candidate certificate, TARGET-W acceptance, physical packed kernels, arithmetic throughput, 8-GiB allocation, 4B-class latency"
            ],
        },
    }

    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/case_rows.json", rows)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/q4_head_manifest.json",
            "artifacts/deterministic_core.json",
            "raw/case_rows.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "holdout_accepted_lengths": holdout["accepted_lengths"],
                "holdout_rank_p50": holdout["transported_rank"]["p50"],
                "holdout_rank_p95": holdout["transported_rank"]["p95"],
                "holdout_rank_max": holdout["transported_rank"]["maximum"],
                "holdout_top16_fraction": holdout["transported_rank"][
                    "top16_fraction"
                ],
                "static_rank_p95": holdout["static_fine_rank"]["p95"],
                "rank_p95_improvement": holdout["rank_p95_improvement"],
                "hot_projection_passed": total_hot <= hot_limit,
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
