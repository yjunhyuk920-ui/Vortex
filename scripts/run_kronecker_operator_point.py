from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import (
    common_prefix_length,
    greedy_tokens,
    teacher_forced_logits,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.candidate_coverage import token_rank
from vortex_runtime.feasibility import default_specs
from vortex_runtime.kronecker_operator import (
    KroneckerFitStats,
    kronecker_operator_budget,
    replace_linears_with_kronecker,
)
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace every TinyLlama linear operation with executable balanced "
            "Kronecker factors and measure exact-prefix behavior."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tiny-rank", type=int, required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument("--embedding-bits", type=int, default=4)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--tree-depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--node-budget", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--embedding-row-chunk", type=int, default=128)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kronecker_operator_point.json"),
    )
    return parser.parse_args()


def teacher_summary(
    *,
    logits: torch.Tensor,
    exact_tokens: torch.Tensor,
) -> dict[str, object]:
    ranks: list[int] = []
    matches = 0
    for position in range(exact_tokens.shape[1]):
        exact_token = int(exact_tokens[0, position].item())
        position_logits = logits[0, position]
        predicted = int(torch.argmax(position_logits).item())
        matches += int(predicted == exact_token)
        ranks.append(token_rank(position_logits, exact_token))
    return {
        "tokens": len(ranks),
        "top1_rate": matches / len(ranks),
        "top4_rate": sum(rank <= 4 for rank in ranks) / len(ranks),
        "top32_rate": sum(rank <= 32 for rank in ranks) / len(ranks),
        "mean_exact_token_rank": sum(ranks) / len(ranks),
        "maximum_exact_token_rank": max(ranks),
        "ranks": ranks,
    }


def quantize_embedding(
    model: nn.Module,
    *,
    bits: int,
    row_chunk: int,
) -> dict[str, int | float | str]:
    embedding = model.model.embed_tokens
    restored, stats = symmetric_per_row_fake_quantize(
        embedding.weight,
        bits=bits,
        source_bits=16,
        name="model.embed_tokens.weight",
        row_chunk=row_chunk,
    )
    with torch.no_grad():
        embedding.weight.copy_(
            restored.to(
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
        )
    del restored
    return stats.to_dict()


def aggregate_fit_stats(
    stats: dict[str, KroneckerFitStats],
) -> dict[str, object]:
    if not stats:
        raise ValueError("at least one fitted operator is required")
    total_original = sum(item.original_elements for item in stats.values())
    total_factors = sum(item.factor_elements for item in stats.values())
    weighted_error = sum(
        item.original_elements * item.relative_l2_error
        for item in stats.values()
    ) / total_original
    maximum_error = max(item.relative_l2_error for item in stats.values())
    worst = sorted(
        (
            {"name": name, **item.to_dict()}
            for name, item in stats.items()
        ),
        key=lambda item: float(item["relative_l2_error"]),
        reverse=True,
    )[:24]
    return {
        "operators": len(stats),
        "original_elements": total_original,
        "factor_elements": total_factors,
        "compression_ratio": total_original / total_factors,
        "element_weighted_relative_l2_error": weighted_error,
        "maximum_operator_relative_l2_error": maximum_error,
        "worst_operators": worst,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.tiny_rank,
        args.target_rank,
        args.factor_bits,
        args.embedding_bits,
        args.tokens,
        args.tree_depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.embedding_row_chunk,
    ) <= 0:
        raise SystemExit("positive integer controls are required")

    target, baseline = default_specs()
    target_budget = kronecker_operator_budget(
        target=target,
        baseline=baseline,
        rank=args.target_rank,
        factor_bits=args.factor_bits,
        embedding_bits=args.embedding_bits,
        active_kv_tokens=256,
        workspace_gib=1.5,
        allocator_reserve_gib=1.0,
        resident_hbm_gib_s=300.0,
        effective_tops=160.0,
    )

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)

    started = time.perf_counter()
    exact_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=max(args.tokens, args.tree_depth),
    ).to("cpu")
    evaluation_tokens = exact_tokens[:, : args.tokens]
    exact_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=evaluation_tokens,
    )

    embedding_stats = quantize_embedding(
        model,
        bits=args.embedding_bits,
        row_chunk=args.embedding_row_chunk,
    )
    fit_started = time.perf_counter()
    per_operator = replace_linears_with_kronecker(
        model,
        rank=args.tiny_rank,
        factor_bits=args.factor_bits,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - fit_started
    fit_summary = aggregate_fit_stats(per_operator)
    gc.collect()

    factorized_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=evaluation_tokens,
    )
    exact_teacher = teacher_summary(
        logits=exact_logits,
        exact_tokens=evaluation_tokens,
    )
    factorized_teacher = teacher_summary(
        logits=factorized_logits,
        exact_tokens=evaluation_tokens,
    )
    factorized_autoregressive = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    autoregressive_prefix = common_prefix_length(
        factorized_autoregressive,
        evaluation_tokens,
    )

    reference = tuple(
        int(token) for token in exact_tokens[0, : args.tree_depth].tolist()
    )
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, tuple())]
    depth_rows: list[dict[str, int | float | bool]] = []
    exact_path_last_alive_depth = 0
    for depth_index in range(args.tree_depth):
        sequences = [sequence for _, sequence in beams]
        logits = _next_logits(
            model=model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            sequences=sequences,
            batch_size=args.eval_batch_size,
        )
        log_probabilities = torch.log_softmax(logits, dim=-1)
        width = min(args.top_k, log_probabilities.shape[-1])
        values, indices = torch.topk(log_probabilities, k=width, dim=-1)
        expanded: list[tuple[float, tuple[int, ...]]] = []
        for beam_index, (score, sequence) in enumerate(beams):
            for child_index in range(width):
                token = int(indices[beam_index, child_index].item())
                expanded.append(
                    (
                        score + float(values[beam_index, child_index].item()),
                        sequence + (token,),
                    )
                )
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = _trim_to_node_budget(
            expanded,
            beam_width=args.beam_width,
            node_budget=args.node_budget,
        )
        if not beams:
            break
        retained = [sequence for _, sequence in beams]
        current_depth = depth_index + 1
        exact_path_alive = reference[:current_depth] in retained
        if exact_path_alive:
            exact_path_last_alive_depth = current_depth
        depth_rows.append(
            {
                "depth": current_depth,
                "retained_branches": len(beams),
                "unique_tree_nodes": unique_prefix_node_count(retained),
                "longest_exact_reference_prefix": longest_reference_prefix(
                    retained,
                    reference,
                ),
                "exact_path_alive": exact_path_alive,
            }
        )

    retained = [sequence for _, sequence in beams]
    tree_nodes = unique_prefix_node_count(retained) if retained else 0
    tree_prefix = longest_reference_prefix(retained, reference) if retained else 0
    qualifies_for_next_rank = bool(
        target_budget.pass_all
        and factorized_teacher["top32_rate"] >= 0.95
        and exact_path_last_alive_depth >= 1
    )
    payload = {
        "evidence_level": "E2 executable all-linear Kronecker replacement",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tiny_rank": args.tiny_rank,
        "projected_405b_rank": args.target_rank,
        "factor_bits": args.factor_bits,
        "embedding_bits": args.embedding_bits,
        "embedding_quantization": embedding_stats,
        "fit": fit_summary,
        "fit_seconds": fit_seconds,
        "teacher_forced_exact_reference": exact_teacher,
        "teacher_forced_factorized": factorized_teacher,
        "autoregressive_exact_prefix": autoregressive_prefix,
        "tree": {
            "requested_depth": args.tree_depth,
            "top_k": args.top_k,
            "beam_width": args.beam_width,
            "node_budget": args.node_budget,
            "retained_tree_nodes": tree_nodes,
            "optimistic_committed_exact_prefix": tree_prefix,
            "exact_path_last_alive_depth": exact_path_last_alive_depth,
            "depth_rows": depth_rows,
        },
        "405b_budget": target_budget.to_dict(),
        "contract": (
            "Every nn.Linear operation, including lm_head, is replaced by the "
            "factorized operator. Factors are built from checkpoint weights only; "
            "no activation calibration or learned correction is used."
        ),
        "qualifies_for_next_rank": qualifies_for_next_rank,
        "decision": (
            "advance Kronecker rank frontier"
            if qualifies_for_next_rank
            else "reject tested Kronecker point"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
