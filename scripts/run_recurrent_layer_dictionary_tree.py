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
from scripts.run_thinned_substitute_tree_gate import unique_parameter_count
from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules
from vortex_runtime.recurrent_layer_dictionary import (
    recurrent_draft_budget,
    recurrent_layer_schedule,
)
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    speculative_tree_verification_budget,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a full-depth Q4 draft whose layer positions reuse a small "
            "resident dictionary of original target layers."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--unique-layers", type=int, default=3)
    parser.add_argument(
        "--representative-strategy",
        choices=("front", "uniform", "edge"),
        required=True,
    )
    parser.add_argument(
        "--assignment-strategy",
        choices=("nearest", "cyclic"),
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
    parser.add_argument("--draft-effective-tops", type=float, default=160.0)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument("--target-hot-effective-tops", type=float, default=160.0)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recurrent_layer_dictionary_tree.json"),
    )
    return parser.parse_args()


def apply_recurrent_schedule(
    model: nn.Module,
    *,
    assignment: tuple[int, ...],
) -> tuple[int, int]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError("expected a Llama-like model.model.layers ModuleList")
    original = list(layers)
    if len(assignment) != len(original):
        raise ValueError("assignment length must equal original model depth")
    if any(index < 0 or index >= len(original) for index in assignment):
        raise ValueError("recurrent assignment index out of range")

    backbone.layers = nn.ModuleList([original[index] for index in assignment])
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(assignment)
    if hasattr(backbone, "config"):
        backbone.config.num_hidden_layers = len(assignment)
    return len(original), len(set(assignment))


def main() -> None:
    args = parse_args()
    if not 2 <= args.bits < 16:
        raise SystemExit("bits must be in [2, 16)")
    if min(
        args.unique_layers,
        args.depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.row_chunk,
    ) <= 0:
        raise SystemExit("dictionary, tree and quantization parameters must be positive")

    target, baseline = default_specs()
    projected_draft = recurrent_draft_budget(
        target=target,
        baseline=baseline,
        unique_layers=args.unique_layers,
        weight_bits=args.bits,
        tie_word_embeddings=False,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
        effective_tops=args.draft_effective_tops,
    )
    if not projected_draft.memory_pass:
        raise SystemExit("projected recurrent draft does not fit the memory limit")

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

    original_depth = len(model.model.layers)
    schedule = recurrent_layer_schedule(
        total_layers=original_depth,
        unique_layers=args.unique_layers,
        representative_strategy=args.representative_strategy,
        assignment_strategy=args.assignment_strategy,
    )
    original_layers, actual_unique_layers = apply_recurrent_schedule(
        model,
        assignment=schedule.assignment,
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
        projected_draft.pass_all
        and path_survives
        and target_lower_bound.observed_serialized_pass
    )

    actual_weight_gib = parameters_before_quantization * args.bits / 8 / (1024**3)
    payload = {
        "evidence_level": (
            "E2 causal full-depth recurrent layer dictionary plus target lower bound"
        ),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "schedule": schedule.to_dict(),
        "original_layers": original_layers,
        "actual_unique_layers": actual_unique_layers,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "requested_depth": args.depth,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "node_budget": args.node_budget,
        "retained_tree_nodes": tree_nodes,
        "reached_depth": reached_depth,
        "optimistic_committed_exact_prefix": optimistic_committed,
        "exact_path_last_alive_depth": exact_path_last_alive_depth,
        "tinyllama_recurrent_draft": {
            "unique_parameters": parameters_before_quantization,
            "projected_q_weight_gib": actual_weight_gib,
            "precision": precision_stats.to_dict(),
        },
        "405b_recurrent_draft_budget": projected_draft.to_dict(),
        "depth_rows": depth_rows,
        "405b_q4_target_side_lower_bound": target_lower_bound.to_dict(),
        "contract": (
            "Only existing target layers are stored. The selected dictionary is "
            "reused across every original depth position without learned adapters. "
            "Target verification remains a Q4 cost lower bound and omits exact "
            "Q6/Q8 precision."
        ),
        "qualifies": lower_bound_survives,
        "decision": (
            "survive recurrent dictionary lower bound; require exact verifier"
            if lower_bound_survives
            else "reject tested recurrent layer dictionary tree point"
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
