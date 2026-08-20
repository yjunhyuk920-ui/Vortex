from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_094a.run_experiment import (  # noqa: E402
    PrefillState,
    checkpoint_identity,
    fresh_first_layer_cache,
    fresh_full_cache,
    load_model,
    prefill_case,
    run_fine_guessed_sweep,
    sha256_bytes,
    sha256_file,
    verify_runtime,
    write_checksums,
    write_json,
)
from vortex_runtime.causal_suffix_action_cache import (  # noqa: E402
    RowwiseSymmetricQ4Head,
    project_target_hot_state,
    quantize_rowwise_symmetric_int4,
)
from vortex_runtime.exact_jacobi_fixed_point import (  # noqa: E402
    build_seed,
    jacobi_input,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    module_parameter_bytes,
)
from vortex_runtime.online_residual_affine_hull import (  # noqa: E402
    accepted_prefix,
    causal_affine_transport,
    canonical_sha256,
    model_free_controls,
    project_target_hot_bytes,
    residual_span_projection,
    stable_argmax64,
    stable_true_token_rank,
    summarize_ranks,
    tensor_sha256,
)
from vortex_runtime.parareal_residual_transport import (  # noqa: E402
    logical_source_fraction,
    residual_buffer_bytes,
)
from vortex_runtime.prefix_state_bisimulation import (  # noqa: E402
    cache_nbytes,
    cache_sha256,
)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-095a-online-residual-affine-hull-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("configuration was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["shallow_depth"]) != 1:
        raise RuntimeError("EXP-095A freezes one coarse decoder layer")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-095A freezes K=128")
    if str(config["seed_mode"]) != "prompt_suffix_cycle_16":
        raise RuntimeError("seed changed from the inherited frozen selection")
    q4 = config["q4_head"]
    if q4["scheme"] != "rowwise_symmetric_int4":
        raise RuntimeError("Q4 head scheme changed")
    if int(q4["minimum"]) != -8 or int(q4["maximum"]) != 7:
        raise RuntimeError("Q4 head range changed")

    causal = config["causal_transport"]
    if causal["mechanism"] != "affine_ridge_from_nearest_old_shallow_states":
        raise RuntimeError("causal transport mechanism changed")
    if int(causal["neighbor_count"]) != 8:
        raise RuntimeError("neighbor count changed")
    if float(causal["ridge_lambda"]) != 2.0**-10:
        raise RuntimeError("ridge coefficient changed")
    if float(causal["coefficient_clip"]) != 4.0:
        raise RuntimeError("coefficient clipping changed")
    if causal["working_dtype"] != "float64":
        raise RuntimeError("causal working dtype changed")

    oracle = config["oracle_span"]
    if oracle["mechanism"] != "full_online_residual_row_span_l2_projection":
        raise RuntimeError("oracle projection mechanism changed")
    if float(oracle["relative_eigenvalue_cutoff"]) != 2.0**-40:
        raise RuntimeError("oracle eigenvalue cutoff changed")
    if oracle["working_dtype"] != "float64":
        raise RuntimeError("oracle working dtype changed")
    if not bool(oracle["single_residual_oracle"]):
        raise RuntimeError("single-residual oracle disabled")

    target = config["target_projection"]
    if target["model_id"] != TARGET_MODEL_ID or target["revision"] != TARGET_REVISION:
        raise RuntimeError("TARGET-W identity mismatch")
    success = config["success"]
    if int(success["chain_required_accepted_length"]) != 128:
        raise RuntimeError("chain threshold changed")
    for prefix in ("causal", "oracle"):
        if int(success[f"{prefix}_rank_p95_max"]) != 4:
            raise RuntimeError(f"{prefix} p95 threshold changed")
        if int(success[f"{prefix}_rank_max_max"]) != 16:
            raise RuntimeError(f"{prefix} maximum threshold changed")

    cases = list(config["cases"])
    build = [row for row in cases if row["split"] == "build"]
    holdout = [row for row in cases if row["split"] == "holdout"]
    if len(build) < int(success["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(success["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    ids = [str(row["id"]) for row in cases]
    prompt_hashes = [sha256_bytes(row["prompt"].encode("utf-8")) for row in cases]
    if len(ids) != len(set(ids)) or len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("duplicate case ID or prompt hash")


def shallow_step_with_hidden(
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
        normalized = model.model.norm(layer_output[0])
        logits = q4_head.logits(normalized)
    return (
        logits[0, -1, :].detach().to(torch.float32).contiguous().cpu(),
        normalized[0, -1, :].detach().to(torch.float32).contiguous().cpu(),
    )


def run_old_coarse_bank(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    input_tokens: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    cache = fresh_first_layer_cache(prefill.prefix_cache)
    logits_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    started_ns = time.perf_counter_ns()
    for offset, token in enumerate(input_tokens):
        logits, hidden = shallow_step_with_hidden(
            model,
            q4_head,
            token=int(token),
            cache=cache,
            position=len(prefill.prompt_ids) + offset,
        )
        logits_rows.append(logits)
        hidden_rows.append(hidden)
    completed_ns = time.perf_counter_ns()
    logits = torch.stack(logits_rows, dim=0).contiguous()
    hidden = torch.stack(hidden_rows, dim=0).contiguous()
    return logits, hidden, {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "calls": len(logits_rows),
        "logits_sha256": tensor_sha256(logits),
        "hidden_sha256": tensor_sha256(hidden),
        "first_logits_sha256": tensor_sha256(logits[0]),
        "first_hidden_sha256": tensor_sha256(hidden[0]),
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
    }


def exact_or_affine_scores(
    *,
    coarse_logits: torch.Tensor,
    residual_bank: torch.Tensor,
    hidden_bank: torch.Tensor,
    query_hidden: torch.Tensor,
    neighbor_count: int,
    ridge_lambda: float,
    coefficient_clip: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    query = query_hidden.detach().to(torch.float64).contiguous().cpu()
    bank = hidden_bank.detach().to(torch.float64).contiguous().cpu()
    exact_rows = [
        index
        for index in range(int(bank.shape[0]))
        if torch.equal(bank[index], query)
    ]
    if exact_rows:
        index = min(exact_rows)
        scores = (
            coarse_logits.detach().to(torch.float64).contiguous().cpu()
            + residual_bank[index].detach().to(torch.float64).contiguous().cpu()
        ).contiguous()
        return scores, {
            "exact_hidden_reuse": True,
            "neighbor_indices": [index],
            "coefficients": [1.0],
            "coefficient_sum": 1.0,
            "coefficient_abs_max": 1.0,
            "hidden_reconstruction_l2": 0.0,
        }
    result = causal_affine_transport(
        coarse_logits=coarse_logits,
        residual_bank=residual_bank,
        hidden_bank=hidden_bank,
        query_hidden=query_hidden,
        neighbor_count=int(neighbor_count),
        ridge_lambda=float(ridge_lambda),
        coefficient_clip=float(coefficient_clip),
    )
    return result.scores, {
        "exact_hidden_reuse": False,
        "neighbor_indices": list(result.neighbor_indices),
        "coefficients": list(result.coefficients),
        "coefficient_sum": result.coefficient_sum,
        "coefficient_abs_max": result.coefficient_abs_max,
        "hidden_reconstruction_l2": result.reconstruction_l2,
    }


def run_causal_candidate(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    residual_bank: torch.Tensor,
    hidden_bank: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    causal = config["causal_transport"]
    cache = fresh_first_layer_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    row_manifests: list[dict[str, Any]] = []
    score_hashes: list[str] = []
    hidden_hashes: list[str] = []
    first_coarse_hash = ""
    first_hidden_hash = ""
    started_ns = time.perf_counter_ns()
    for offset in range(int(config["block_length"])):
        coarse, hidden = shallow_step_with_hidden(
            model,
            q4_head,
            token=current,
            cache=cache,
            position=len(prefill.prompt_ids) + offset,
        )
        scores, manifest = exact_or_affine_scores(
            coarse_logits=coarse,
            residual_bank=residual_bank,
            hidden_bank=hidden_bank,
            query_hidden=hidden,
            neighbor_count=int(causal["neighbor_count"]),
            ridge_lambda=float(causal["ridge_lambda"]),
            coefficient_clip=float(causal["coefficient_clip"]),
        )
        if offset == 0:
            first_coarse_hash = tensor_sha256(coarse)
            first_hidden_hash = tensor_sha256(hidden)
        token = stable_argmax64(scores)
        tokens.append(token)
        row_manifests.append(manifest)
        score_hashes.append(tensor_sha256(scores))
        hidden_hashes.append(tensor_sha256(hidden))
        current = token
    completed_ns = time.perf_counter_ns()
    return tuple(tokens), {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "calls": len(tokens),
        "candidate_sha256": canonical_sha256(tokens),
        "score_sha256": canonical_sha256(score_hashes),
        "hidden_sha256": canonical_sha256(hidden_hashes),
        "first_coarse_logits_sha256": first_coarse_hash,
        "first_hidden_sha256": first_hidden_hash,
        "exact_hidden_reuse_count": sum(
            bool(row["exact_hidden_reuse"]) for row in row_manifests
        ),
        "coefficient_abs_max": max(
            float(row["coefficient_abs_max"]) for row in row_manifests
        ),
        "coefficient_sum_max_error": max(
            abs(float(row["coefficient_sum"]) - 1.0) for row in row_manifests
        ),
        "hidden_reconstruction_l2_p50": float(
            np.percentile(
                [float(row["hidden_reconstruction_l2"]) for row in row_manifests],
                50,
                method="linear",
            )
        ),
        "rows": row_manifests,
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
        "target_information_used": False,
    }


def best_single_residual_ranks_batch(
    coarse_true: torch.Tensor,
    residual_bank: torch.Tensor,
    target_tokens: Sequence[int],
) -> tuple[list[int], list[int]]:
    coarse = coarse_true.detach().to(torch.float64).contiguous().cpu()
    bank = residual_bank.detach().to(torch.float64).contiguous().cpu()
    if coarse.shape[0] != len(target_tokens) or coarse.shape[1] != bank.shape[1]:
        raise RuntimeError("single-residual oracle shape mismatch")
    best_ranks: list[int] = []
    best_indices: list[int] = []
    for position, token_value in enumerate(target_tokens):
        token = int(token_value)
        scores = bank + coarse[position].unsqueeze(0)
        target_values = scores[:, token].unsqueeze(1)
        greater = torch.sum(scores > target_values, dim=1, dtype=torch.int64)
        earlier_equal = torch.sum(
            scores[:, :token] == target_values,
            dim=1,
            dtype=torch.int64,
        )
        ranks = 1 + greater + earlier_equal
        best_rank = int(torch.min(ranks).item())
        best_index = int(torch.nonzero(ranks == best_rank, as_tuple=False)[0].item())
        best_ranks.append(best_rank)
        best_indices.append(best_index)
    return best_ranks, best_indices


def run_delayed_target_and_oracles(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    residual_bank: torch.Tensor,
    hidden_bank: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_cache: Any = fresh_full_cache(prefill.prefix_cache)
    coarse_cache = fresh_first_layer_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    target_tokens: list[int] = []
    fine_rows: list[torch.Tensor] = []
    coarse_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    causal_ranks: list[int] = []
    aligned_ranks: list[int] = []
    causal_top1: list[int] = []
    causal_score_hashes: list[str] = []
    target_logit_hashes: list[str] = []
    started_ns = time.perf_counter_ns()
    causal = config["causal_transport"]
    for offset in range(int(config["block_length"])):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
        fine = output.logits[0, -1].detach().to(torch.float32).contiguous().cpu()
        true_token = stable_argmax64(fine)
        coarse, hidden = shallow_step_with_hidden(
            model,
            q4_head,
            token=current,
            cache=coarse_cache,
            position=len(prefill.prompt_ids) + offset,
        )
        causal_scores, _manifest = exact_or_affine_scores(
            coarse_logits=coarse,
            residual_bank=residual_bank,
            hidden_bank=hidden_bank,
            query_hidden=hidden,
            neighbor_count=int(causal["neighbor_count"]),
            ridge_lambda=float(causal["ridge_lambda"]),
            coefficient_clip=float(causal["coefficient_clip"]),
        )
        aligned_scores = (
            coarse.to(torch.float64) + residual_bank[offset].to(torch.float64)
        )
        target_tokens.append(true_token)
        fine_rows.append(fine)
        coarse_rows.append(coarse)
        hidden_rows.append(hidden)
        causal_ranks.append(stable_true_token_rank(causal_scores, true_token))
        aligned_ranks.append(stable_true_token_rank(aligned_scores, true_token))
        causal_top1.append(stable_argmax64(causal_scores))
        causal_score_hashes.append(tensor_sha256(causal_scores))
        target_logit_hashes.append(tensor_sha256(fine))
        current = true_token
        target_cache = output.past_key_values
    completed_ns = time.perf_counter_ns()

    fine_true = torch.stack(fine_rows, dim=0).contiguous()
    coarse_true = torch.stack(coarse_rows, dim=0).contiguous()
    hidden_true = torch.stack(hidden_rows, dim=0).contiguous()
    target_residual = (
        fine_true.to(torch.float64) - coarse_true.to(torch.float64)
    ).contiguous()

    single_ranks, single_indices = best_single_residual_ranks_batch(
        coarse_true, residual_bank, target_tokens
    )
    span = residual_span_projection(
        residual_bank,
        target_residual,
        relative_eigenvalue_cutoff=float(
            config["oracle_span"]["relative_eigenvalue_cutoff"]
        ),
    )
    span_scores = coarse_true.to(torch.float64) + span.projected_residual
    span_ranks = [
        stable_true_token_rank(span_scores[index], token)
        for index, token in enumerate(target_tokens)
    ]
    span_top1 = [stable_argmax64(row) for row in span_scores]

    return {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "target_calls": len(target_tokens),
        "true_coarse_calls": len(target_tokens),
        "target_tokens": target_tokens,
        "target_sha256": canonical_sha256(target_tokens),
        "target_logits_sha256": canonical_sha256(target_logit_hashes),
        "fine_true_block_sha256": tensor_sha256(fine_true),
        "coarse_true_block_sha256": tensor_sha256(coarse_true),
        "hidden_true_block_sha256": tensor_sha256(hidden_true),
        "target_residual_sha256": tensor_sha256(target_residual),
        "causal_true_path_ranks": causal_ranks,
        "causal_true_path_top1_sha256": canonical_sha256(causal_top1),
        "causal_score_sha256": canonical_sha256(causal_score_hashes),
        "position_aligned_ranks": aligned_ranks,
        "single_residual_oracle_ranks": single_ranks,
        "single_residual_oracle_indices": single_indices,
        "span_oracle_ranks": span_ranks,
        "span_oracle_top1_sha256": canonical_sha256(span_top1),
        "span_projection": {
            **span.summary(),
            "projected_residual_sha256": tensor_sha256(span.projected_residual),
            "coefficient_sha256": tensor_sha256(span.coefficients),
            "coefficient_abs_max": float(torch.max(torch.abs(span.coefficients)).item()),
        },
        "final_target_cache_sha256": cache_sha256(target_cache),
        "final_target_cache_bytes": cache_nbytes(target_cache),
        "final_true_coarse_cache_sha256": cache_sha256(coarse_cache),
        "final_true_coarse_cache_bytes": cache_nbytes(coarse_cache),
    }


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
    coarse_old, hidden_old, coarse_meta = run_old_coarse_bank(
        model, q4_head, prefill, input_tokens
    )
    residual_bank = (
        fine_old.to(torch.float64) - coarse_old.to(torch.float64)
    ).contiguous()
    candidate, candidate_meta = run_causal_candidate(
        model,
        q4_head,
        prefill,
        residual_bank,
        hidden_old,
        config,
    )
    candidate_completed_ns = int(candidate_meta["completed_ns"])
    target = run_delayed_target_and_oracles(
        model,
        q4_head,
        prefill,
        residual_bank,
        hidden_old,
        config,
    )
    target_tokens = tuple(int(value) for value in target["target_tokens"])
    causal_summary = summarize_ranks(target["causal_true_path_ranks"])
    aligned_summary = summarize_ranks(target["position_aligned_ranks"])
    single_summary = summarize_ranks(target["single_residual_oracle_ranks"])
    span_summary = summarize_ranks(target["span_oracle_ranks"])
    prefix = accepted_prefix(candidate, target_tokens)
    exact_positions = sum(
        int(left) == int(right) for left, right in zip(candidate, target_tokens)
    )
    first_controls = {
        "old_and_candidate_first_coarse_equal": coarse_meta["first_logits_sha256"]
        == candidate_meta["first_coarse_logits_sha256"],
        "old_and_candidate_first_hidden_equal": coarse_meta["first_hidden_sha256"]
        == candidate_meta["first_hidden_sha256"],
        "fine_first_token_equals_target": int(fine_meta["first_proposal_token"])
        == int(target_tokens[0]),
        "causal_first_rank_one": int(target["causal_true_path_ranks"][0]) == 1,
        "aligned_first_rank_one": int(target["position_aligned_ranks"][0]) == 1,
        "single_oracle_first_rank_one": int(
            target["single_residual_oracle_ranks"][0]
        )
        == 1,
        "span_oracle_first_rank_one": int(target["span_oracle_ranks"][0]) == 1,
    }
    row = {
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
        "residual_bank_sha256": tensor_sha256(residual_bank),
        "residual_bank_bytes": int(residual_bank.numel() * residual_bank.element_size()),
        "hidden_bank_sha256": tensor_sha256(hidden_old),
        "hidden_bank_bytes": int(hidden_old.numel() * hidden_old.element_size()),
        "candidate": candidate_meta,
        "target": target,
        "candidate_completed_before_target": candidate_completed_ns
        <= int(target["started_ns"]),
        "candidate_tokens_sha256": canonical_sha256(candidate),
        "target_tokens_sha256": canonical_sha256(target_tokens),
        "candidate_target_prefix": prefix,
        "candidate_target_exact_positions": exact_positions,
        "first_position_controls": first_controls,
        "causal_rank": causal_summary.to_dict(),
        "position_aligned_rank": aligned_summary.to_dict(),
        "single_residual_oracle_rank": single_summary.to_dict(),
        "span_oracle_rank": span_summary.to_dict(),
    }
    del fine_old, coarse_old, hidden_old, residual_bank
    gc.collect()
    return row


def aggregate(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == str(split)]
    if not selected:
        raise RuntimeError(f"empty aggregate {split}")
    causal_ranks = [
        int(value)
        for row in selected
        for value in row["target"]["causal_true_path_ranks"]
    ]
    aligned_ranks = [
        int(value)
        for row in selected
        for value in row["target"]["position_aligned_ranks"]
    ]
    single_ranks = [
        int(value)
        for row in selected
        for value in row["target"]["single_residual_oracle_ranks"]
    ]
    span_ranks = [
        int(value)
        for row in selected
        for value in row["target"]["span_oracle_ranks"]
    ]
    accepted = [int(row["candidate_target_prefix"]) for row in selected]
    exact_positions = [
        int(row["candidate_target_exact_positions"]) for row in selected
    ]
    return {
        "split": split,
        "case_count": len(selected),
        "position_count": len(causal_ranks),
        "accepted_lengths": accepted,
        "accepted_min": min(accepted),
        "accepted_p50": float(np.percentile(accepted, 50, method="linear")),
        "accepted_max": max(accepted),
        "exact_positions": exact_positions,
        "exact_positions_p50": float(
            np.percentile(exact_positions, 50, method="linear")
        ),
        "causal_rank": summarize_ranks(causal_ranks).to_dict(),
        "position_aligned_rank": summarize_ranks(aligned_ranks).to_dict(),
        "single_residual_oracle_rank": summarize_ranks(single_ranks).to_dict(),
        "span_oracle_rank": summarize_ranks(span_ranks).to_dict(),
        "span_effective_ranks": [
            int(row["target"]["span_projection"]["effective_rank"])
            for row in selected
        ],
        "span_projection_relative_l2": [
            float(row["target"]["span_projection"]["projection_relative_l2"])
            for row in selected
        ],
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
        raise RuntimeError("model-free residual affine-hull controls failed")

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
        print(
            f"{row['split']} {row['case_id']}: "
            f"accepted={row['candidate_target_prefix']} exact={row['candidate_target_exact_positions']} "
            f"causal p95/max={row['causal_rank']['p95']}/{row['causal_rank']['maximum']} "
            f"single p95/max={row['single_residual_oracle_rank']['p95']}/{row['single_residual_oracle_rank']['maximum']} "
            f"span p95/max={row['span_oracle_rank']['p95']}/{row['span_oracle_rank']['maximum']}",
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
        int(row["fine_sweep"]["post_sweep_cache_bytes"]) > 0
        and int(row["candidate"]["calls"]) == int(config["block_length"])
        and int(row["coarse_old"]["calls"]) == int(config["block_length"])
        and int(row["target"]["target_calls"]) == int(config["block_length"])
        and int(row["target"]["true_coarse_calls"]) == int(config["block_length"])
        for row in rows
    ):
        integrity_failures.append("incomplete_call_trace")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")
    if not all(
        float(row["target"]["span_projection"]["gram_symmetry_max_abs"])
        <= 1e-6
        for row in rows
    ):
        integrity_failures.append("gram_symmetry_failure")
    integrity_passed = len(integrity_failures) <= int(
        config["success"]["maximum_integrity_mismatches"]
    )

    all_min_accepted = min(int(row["candidate_target_prefix"]) for row in rows)
    exact_chain_signal = (
        integrity_passed
        and all_min_accepted
        >= int(config["success"]["chain_required_accepted_length"])
    )
    causal_branch_signal = (
        integrity_passed
        and not exact_chain_signal
        and float(build["causal_rank"]["p95"])
        <= float(config["success"]["causal_rank_p95_max"])
        and int(build["causal_rank"]["maximum"])
        <= int(config["success"]["causal_rank_max_max"])
        and float(holdout["causal_rank"]["p95"])
        <= float(config["success"]["causal_rank_p95_max"])
        and int(holdout["causal_rank"]["maximum"])
        <= int(config["success"]["causal_rank_max_max"])
    )
    oracle_span_signal = (
        integrity_passed
        and not exact_chain_signal
        and not causal_branch_signal
        and float(build["span_oracle_rank"]["p95"])
        <= float(config["success"]["oracle_rank_p95_max"])
        and int(build["span_oracle_rank"]["maximum"])
        <= int(config["success"]["oracle_rank_max_max"])
        and float(holdout["span_oracle_rank"]["p95"])
        <= float(config["success"]["oracle_rank_p95_max"])
        and int(holdout["span_oracle_rank"]["maximum"])
        <= int(config["success"]["oracle_rank_max_max"])
    )
    if not integrity_passed:
        decision = "INVALID_ONLINE_RESIDUAL_AFFINE_HULL_CONTROL_FAILURE"
    elif exact_chain_signal:
        decision = "PROMOTE_CAUSAL_RESIDUAL_AFFINE_HULL_TO_EXACT_CERTIFICATE_GATE"
    elif causal_branch_signal:
        decision = "PROMOTE_CAUSAL_RESIDUAL_AFFINE_HULL_TO_BRANCH_GATE"
    elif oracle_span_signal:
        decision = "PROMOTE_ONLINE_RESIDUAL_SPAN_TO_CAUSAL_COEFFICIENT_GATE"
    else:
        decision = "REJECT_ONLINE_DEEP_RESIDUAL_AFFINE_HULL_AS_405B_SOURCE"

    target_base = project_target_hot_state(config["target_projection"])
    target_residual_bytes = residual_buffer_bytes(
        block_length=int(config["block_length"]),
        vocabulary_size=int(config["target_projection"]["vocab_size"]),
        value_bytes=8,
    )
    inherited_hot = int(target_base.total_hot_bytes) + target_residual_bytes
    target_hot = project_target_hot_bytes(
        inherited_hot_bytes=inherited_hot,
        block_length=int(config["block_length"]),
        vocabulary_size=int(config["target_projection"]["vocab_size"]),
        hidden_size=int(config["target_projection"]["hidden_size"]),
    )
    hot_limit = int(config["target_projection"]["hot_limit_bytes"])
    target_hot["hot_limit_bytes"] = hot_limit
    target_hot["hot_margin_bytes"] = hot_limit - int(target_hot["total_hot_bytes"])
    target_hot["hot_limit_passed"] = int(target_hot["total_hot_bytes"]) <= hot_limit
    compression = float(
        config["target_projection"]["checkpoint_compression_ratio_favorable"]
    )
    holdout_min = int(holdout["accepted_min"])
    neighbor_count = int(config["causal_transport"]["neighbor_count"])
    causal_scalar_work = (
        neighbor_count * int(config["target_projection"]["vocab_size"])
        + neighbor_count * int(config["target_projection"]["hidden_size"])
        + neighbor_count**3
    )
    target_projection = {
        "base_hot_state": target_base.to_dict(),
        "hot": target_hot,
        "causal_scalar_work_per_token": causal_scalar_work,
        "causal_scalar_work_fraction_of_405b": causal_scalar_work
        / 405_000_000_000.0,
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
        "fine_dense_arithmetic_fraction_per_token": 1.0,
        "sound_token_certificate": "NOT_CONSTRUCTED",
    }

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "one-official-k128-fine-guessed-sweep/one-official-first-layer-q4-head/"
            "complete-online-float64-deep-residual-bank/eight-neighbor-affine-ridge-causal-transport/"
            "best-single-residual-oracle/full-residual-row-span-l2-oracle/"
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
            "float64_residual_bank_bytes": residual_buffer_bytes(
                block_length=int(config["block_length"]),
                vocabulary_size=int(model.config.vocab_size),
                value_bytes=8,
            ),
            "float32_hidden_bank_bytes": int(config["block_length"])
            * int(model.config.hidden_size)
            * 4,
            "fine_block_calls": len(rows),
            "coarse_old_calls": len(rows) * int(config["block_length"]),
            "candidate_coarse_calls": len(rows) * int(config["block_length"]),
            "true_path_coarse_calls": len(rows) * int(config["block_length"]),
            "incremental_target_calls": len(rows) * int(config["block_length"]),
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "all_rows_minimum_accepted_length": all_min_accepted,
            "holdout_minimum_accepted_length": holdout_min,
            "exact_chain_signal": exact_chain_signal,
            "causal_branch_signal": causal_branch_signal,
            "oracle_span_signal": oracle_span_signal,
            "hot_projection_passed": bool(target_hot["hot_limit_passed"]),
            "certificate_constructed": False,
        },
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "runtime": {
            **runtime,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "timing": {
            "model_load_ns": model_load_ns,
            "q4_compile_ns": q4_compile_ns,
            "experiment_ns": experiment_ns,
        },
        "provenance": {
            "measured": [
                "DEV-W guessed fine logits, guessed and corrected shallow states/logits, causal candidate, delayed target logits/tokens/caches, exact finite ranks"
            ],
            "derived": [
                "online residual bank, affine-ridge coefficients, best-single residual ranks, full-row-span L2 projection, resource equations"
            ],
            "granted": [
                "target-seeing single-residual and row-span oracles execute after the delayed target and are nondeployable capacity screens"
            ],
            "unverified": [
                "sound coefficient certificate, TARGET-W behavior, physical kernels, 8-GiB allocation, fine arithmetic reduction, 4B-class latency"
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
                "holdout_causal_rank": holdout["causal_rank"],
                "holdout_single_residual_rank": holdout[
                    "single_residual_oracle_rank"
                ],
                "holdout_span_rank": holdout["span_oracle_rank"],
                "holdout_span_effective_ranks": holdout["span_effective_ranks"],
                "hot_total_gib": target_hot["total_hot_gib"],
                "hot_limit_passed": target_hot["hot_limit_passed"],
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
