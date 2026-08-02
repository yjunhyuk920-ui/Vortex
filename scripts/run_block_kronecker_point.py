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
from scripts.run_kronecker_operator_point import quantize_embedding, teacher_summary
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.block_kronecker import (
    BlockKroneckerLinear,
    block_kronecker_budget,
    fit_block_kronecker_linear,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace every TinyLlama linear with one Kronecker term per semantic "
            "attention, MLP or vocabulary block."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument("--embedding-bits", type=int, default=4)
    parser.add_argument("--power-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3031)
    parser.add_argument("--mlp-block-size", type=int, default=64)
    parser.add_argument("--lm-head-block-size", type=int, default=64)
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
        default=Path("block_kronecker_point.json"),
    )
    return parser.parse_args()


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def semantic_plan(
    *,
    name: str,
    linear: nn.Linear,
    head_dim: int,
    mlp_block_size: int,
    lm_head_block_size: int,
) -> tuple[str, int]:
    if name.endswith(("q_proj", "k_proj", "v_proj")):
        return "row", head_dim
    if name.endswith("o_proj"):
        return "column", head_dim
    if name.endswith(("gate_proj", "up_proj")):
        return "row", mlp_block_size
    if name.endswith("down_proj"):
        return "column", mlp_block_size
    if name == "lm_head":
        return "row", lm_head_block_size
    raise ValueError(f"unsupported semantic linear: {name}")


def replace_semantic_linears(
    model: nn.Module,
    *,
    factor_bits: int,
    power_iterations: int,
    seed: int,
    mlp_block_size: int,
    lm_head_block_size: int,
) -> dict[str, dict[str, int | float | str]]:
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(suffixes):
            matches.append((name, module))

    result: dict[str, dict[str, int | float | str]] = {}
    for index, (name, linear) in enumerate(matches):
        mode, block_size = semantic_plan(
            name=name,
            linear=linear,
            head_dim=head_dim,
            mlp_block_size=mlp_block_size,
            lm_head_block_size=lm_head_block_size,
        )
        replacement, stats = fit_block_kronecker_linear(
            linear,
            mode=mode,  # type: ignore[arg-type]
            block_size=block_size,
            factor_bits=factor_bits,
            power_iterations=power_iterations,
            seed=seed + index * 1009,
        )
        parent, attribute = _resolve_parent(model, name)
        setattr(parent, attribute, replacement)
        result[name] = stats
    return result


def aggregate_fit_stats(
    stats: dict[str, dict[str, int | float | str]],
) -> dict[str, object]:
    if not stats:
        raise ValueError("at least one semantic operator is required")
    original = sum(int(item["original_elements"]) for item in stats.values())
    factors = sum(int(item["factor_elements"]) for item in stats.values())
    weighted_error = sum(
        int(item["original_elements"]) * float(item["relative_l2_error"])
        for item in stats.values()
    ) / original
    worst = sorted(
        ({"name": name, **item} for name, item in stats.items()),
        key=lambda item: float(item["relative_l2_error"]),
        reverse=True,
    )[:24]
    modes: dict[str, int] = {}
    for item in stats.values():
        key = str(item["mode"])
        modes[key] = modes.get(key, 0) + 1
    return {
        "operators": len(stats),
        "original_elements": original,
        "factor_elements": factors,
        "factor_density": factors / original,
        "compression_ratio": original / factors,
        "element_weighted_relative_l2_error": weighted_error,
        "modes": modes,
        "worst_operators": worst,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.factor_bits,
        args.embedding_bits,
        args.power_iterations,
        args.mlp_block_size,
        args.lm_head_block_size,
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
    target_budget = block_kronecker_budget(
        target=target,
        baseline=baseline,
        factor_bits=args.factor_bits,
        embedding_bits=args.embedding_bits,
        active_kv_tokens=256,
        attention_terms=4,
        mlp_terms=3,
        lm_head_terms=2,
        mlp_block_size=128,
        lm_head_block_size=256,
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
    per_operator = replace_semantic_linears(
        model,
        factor_bits=args.factor_bits,
        power_iterations=args.power_iterations,
        seed=args.seed,
        mlp_block_size=args.mlp_block_size,
        lm_head_block_size=args.lm_head_block_size,
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
    factorized_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    autonomous_prefix = common_prefix_length(factorized_tokens, evaluation_tokens)

    reference = tuple(
        int(token) for token in exact_tokens[0, : args.tree_depth].tolist()
    )
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, tuple())]
    depth_rows: list[dict[str, int | float | bool]] = []
    last_alive_depth = 0
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
                expanded.append(
                    (
                        score + float(values[beam_index, child_index].item()),
                        sequence + (int(indices[beam_index, child_index].item()),),
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
        alive = reference[:current_depth] in retained
        if alive:
            last_alive_depth = current_depth
        depth_rows.append(
            {
                "depth": current_depth,
                "retained_branches": len(beams),
                "unique_tree_nodes": unique_prefix_node_count(retained),
                "longest_exact_reference_prefix": longest_reference_prefix(
                    retained,
                    reference,
                ),
                "exact_path_alive": alive,
            }
        )

    retained = [sequence for _, sequence in beams]
    tree_nodes = unique_prefix_node_count(retained) if retained else 0
    tree_prefix = longest_reference_prefix(retained, reference) if retained else 0
    target_factor_density = target_budget.factor_elements / target.parameters
    qualifies = bool(
        target_budget.pass_all
        and factorized_teacher["top32_rate"] >= 0.95
        and last_alive_depth >= 1
    )
    payload = {
        "evidence_level": "E2 executable semantic block Kronecker replacement",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "factor_bits": args.factor_bits,
        "embedding_bits": args.embedding_bits,
        "tiny_mlp_block_size": args.mlp_block_size,
        "tiny_lm_head_block_size": args.lm_head_block_size,
        "embedding_quantization": embedding_stats,
        "fit": fit_summary,
        "fit_seconds": fit_seconds,
        "density_comparison": {
            "tiny_factor_density": fit_summary["factor_density"],
            "projected_405b_factor_density": target_factor_density,
            "tiny_over_target_density_ratio": (
                float(fit_summary["factor_density"]) / target_factor_density
            ),
        },
        "teacher_forced_exact_reference": exact_teacher,
        "teacher_forced_factorized": factorized_teacher,
        "autoregressive_exact_prefix": autonomous_prefix,
        "tree": {
            "requested_depth": args.tree_depth,
            "top_k": args.top_k,
            "beam_width": args.beam_width,
            "node_budget": args.node_budget,
            "retained_tree_nodes": tree_nodes,
            "optimistic_committed_exact_prefix": tree_prefix,
            "exact_path_last_alive_depth": last_alive_depth,
            "depth_rows": depth_rows,
        },
        "405b_budget": target_budget.to_dict(),
        "contract": (
            "Every target linear is independently replaced. Attention heads, "
            "MLP neuron blocks and vocabulary blocks are never mixed across the "
            "factor fit. No activation calibration or learned correction is used."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance semantic block Kronecker frontier"
            if qualifies
            else "reject tested semantic block Kronecker point"
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
