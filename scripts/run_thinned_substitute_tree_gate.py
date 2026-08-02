from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    speculative_tree_verification_budget,
    unique_prefix_node_count,
)
from vortex_runtime.substitute_draft_budget import (
    maximum_retained_layers,
    select_layer_indices,
    substitute_draft_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a training-free, target-derived layer-thinned Q4 substitute "
            "draft that fits the projected 405B model inside 8 GiB."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--retained-layers", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=("front", "uniform", "edge"),
        required=True,
    )
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--node-budget", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--memory-limit-gib", type=float, default=8.0)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument("--target-hot-effective-tops", type=float, default=160.0)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thinned_substitute_tree_gate.json"),
    )
    return parser.parse_args()


def unique_parameter_count(model: nn.Module) -> int:
    seen: set[tuple[int, int]] = set()
    elements = 0
    for parameter in model.parameters():
        identity = (
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        elements += parameter.numel()
    return elements


def prune_llama_layers(
    model: nn.Module,
    *,
    retained_indices: tuple[int, ...],
) -> tuple[int, int]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError("expected a Llama-like model.model.layers ModuleList")
    original_layers = len(layers)
    if not retained_indices:
        raise ValueError("at least one layer index is required")
    if any(index < 0 or index >= original_layers for index in retained_indices):
        raise ValueError("retained layer index out of range")

    selected = nn.ModuleList([layers[index] for index in retained_indices])
    for new_index, layer in enumerate(selected):
        self_attention = getattr(layer, "self_attn", None)
        if self_attention is not None and hasattr(self_attention, "layer_idx"):
            self_attention.layer_idx = new_index
    backbone.layers = selected
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(selected)
    if hasattr(backbone, "config"):
        backbone.config.num_hidden_layers = len(selected)
    return original_layers, len(selected)


def main() -> None:
    args = parse_args()
    if not 2 <= args.bits < 16:
        raise SystemExit("bits must be in [2, 16)")
    if min(
        args.retained_layers,
        args.depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.row_chunk,
    ) <= 0:
        raise SystemExit("draft, tree and quantization parameters must be positive")

    target, baseline = default_specs()
    target_max_layers = maximum_retained_layers(
        model=target,
        weight_bits=args.bits,
        tie_word_embeddings=False,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
    )
    if args.retained_layers > target_max_layers:
        raise SystemExit(
            f"requested {args.retained_layers} layers but untied 405B projection "
            f"fits at most {target_max_layers}"
        )

    target_untied_budget = substitute_draft_budget(
        model=target,
        retained_layers=args.retained_layers,
        weight_bits=args.bits,
        tie_word_embeddings=False,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
    )
    target_tied_budget = substitute_draft_budget(
        model=target,
        retained_layers=args.retained_layers,
        weight_bits=args.bits,
        tie_word_embeddings=True,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
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
        count=args.depth,
    ).to("cpu")
    reference = tuple(int(token) for token in exact_tokens[0].tolist())

    original_layer_count = len(model.model.layers)
    retained_indices = select_layer_indices(
        total_layers=original_layer_count,
        retained_layers=args.retained_layers,
        strategy=args.strategy,
    )
    original_layers, actual_layers = prune_llama_layers(
        model,
        retained_indices=retained_indices,
    )
    parameters_before_quantization = unique_parameter_count(model)
    precision_stats, _ = fake_quantize_full_rank_modules(
        model,
        bits=args.bits,
        source_bits=16,
        row_chunk=args.row_chunk,
    )
    gc.collect()

    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, tuple())]
    depth_rows: list[dict[str, int | float | bool]] = []
    exact_path_last_alive_depth = 0

    for depth_index in range(args.depth):
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
                child_score = score + float(values[beam_index, child_index].item())
                expanded.append((child_score, sequence + (token,)))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = _trim_to_node_budget(
            expanded,
            beam_width=args.beam_width,
            node_budget=args.node_budget,
        )
        if not beams:
            break

        retained_sequences = [sequence for _, sequence in beams]
        current_depth = depth_index + 1
        exact_prefix = longest_reference_prefix(retained_sequences, reference)
        exact_path_alive = reference[:current_depth] in retained_sequences
        if exact_path_alive:
            exact_path_last_alive_depth = current_depth
        depth_rows.append(
            {
                "depth": current_depth,
                "retained_branches": len(beams),
                "unique_tree_nodes": unique_prefix_node_count(retained_sequences),
                "longest_exact_reference_prefix": exact_prefix,
                "exact_path_alive": exact_path_alive,
                "best_log_probability": beams[0][0],
                "worst_retained_log_probability": beams[-1][0],
            }
        )

    retained_sequences = [sequence for _, sequence in beams]
    tree_nodes = unique_prefix_node_count(retained_sequences) if beams else 0
    reached_depth = max((len(sequence) for sequence in retained_sequences), default=0)
    optimistic_committed = (
        longest_reference_prefix(retained_sequences, reference) if beams else 0
    )

    target_lower_bound = speculative_tree_verification_budget(
        target=target,
        baseline=baseline,
        hot_bits=4,
        tree_nodes=max(1, tree_nodes),
        tree_depth=max(1, reached_depth),
        committed_tokens=optimistic_committed,
        host_to_device_gib_s=args.host_to_device_gib_s,
        hot_effective_tops=args.target_hot_effective_tops,
    )
    path_survives = optimistic_committed == args.depth
    lower_bound_survives = bool(
        path_survives and target_lower_bound.observed_serialized_pass
    )

    actual_weight_gib_at_requested_bits = (
        parameters_before_quantization * args.bits / 8 / (1024**3)
    )
    payload = {
        "evidence_level": (
            "E2 causal target-derived 8GiB substitute tree plus target-side lower bound"
        ),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "strategy": args.strategy,
        "original_layers": original_layers,
        "retained_layers": actual_layers,
        "retained_indices": list(retained_indices),
        "relative_layer_fraction_tinyllama": actual_layers / original_layers,
        "relative_layer_fraction_405b": args.retained_layers / target.layers,
        "optimism_warning": (
            "Retaining three of 22 TinyLlama layers is proportionally much "
            "stronger than retaining three of 126 target layers."
        ),
        "prompt_tokens": int(prompt_ids.shape[1]),
        "requested_depth": args.depth,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "node_budget": args.node_budget,
        "retained_tree_nodes": tree_nodes,
        "reached_depth": reached_depth,
        "optimistic_committed_exact_prefix": optimistic_committed,
        "exact_path_last_alive_depth": exact_path_last_alive_depth,
        "tinyllama_draft": {
            "unique_parameters": parameters_before_quantization,
            "projected_q_weight_gib": actual_weight_gib_at_requested_bits,
            "precision": precision_stats.to_dict(),
        },
        "405b_substitute_memory": {
            "maximum_untied_layers": target_max_layers,
            "untied": target_untied_budget.to_dict(),
            "tied": target_tied_budget.to_dict(),
        },
        "depth_rows": depth_rows,
        "405b_q4_target_side_lower_bound": target_lower_bound.to_dict(),
        "contract": (
            "The draft is built only by selecting existing target layers and "
            "quantizing them; no training or learned adapter is used. Draft "
            "runtime cost is omitted from latency, and Q6/Q8 exact verification "
            "precision is omitted, so this remains an optimistic necessary gate."
        ),
        "qualifies": lower_bound_survives,
        "decision": (
            "survive thinned-draft lower bound; require exact progressive verifier"
            if lower_bound_survives
            else "reject tested 8GiB thinned substitute tree point"
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
