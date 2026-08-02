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

from scripts.run_full_rank_precision_point import (
    common_prefix_length,
    greedy_tokens,
    teacher_forced_logits,
)
from scripts.run_kronecker_operator_point import teacher_summary
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.feasibility import default_specs
from vortex_runtime.mlp_neuron_dictionary import (
    MLPDictionaryFitStats,
    compile_swiglu_dictionary,
    mlp_neuron_dictionary_budget,
)
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace every TinyLlama SwiGLU MLP with a permutation-invariant "
            "compiled neuron dictionary while leaving attention and LM head exact."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tiny-prototypes", type=int, required=True)
    parser.add_argument("--target-prototypes", type=int, required=True)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--cluster-iterations", type=int, default=5)
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4049)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--tree-depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--node-budget", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlp_neuron_dictionary_point.json"),
    )
    return parser.parse_args()


def replace_all_mlp_modules(
    model: nn.Module,
    *,
    prototypes: int,
    projection_dim: int,
    iterations: int,
    factor_bits: int,
    seed: int,
) -> dict[str, MLPDictionaryFitStats]:
    layers = model.model.layers
    result: dict[str, MLPDictionaryFitStats] = {}
    for layer_index, layer in enumerate(layers):
        original = layer.mlp
        if not all(
            isinstance(getattr(original, name, None), nn.Linear)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            raise RuntimeError("expected a Llama-style SwiGLU MLP")
        compiled, stats = compile_swiglu_dictionary(
            gate_proj=original.gate_proj,
            up_proj=original.up_proj,
            down_proj=original.down_proj,
            prototypes=prototypes,
            projection_dim=projection_dim,
            iterations=iterations,
            factor_bits=factor_bits,
            seed=seed + layer_index * 1009,
        )
        layer.mlp = compiled
        result[f"model.layers.{layer_index}.mlp"] = stats
        del original
        gc.collect()
    return result


def aggregate_fit_stats(
    stats: dict[str, MLPDictionaryFitStats],
) -> dict[str, object]:
    if not stats:
        raise ValueError("at least one MLP layer is required")
    neurons = sum(item.neurons for item in stats.values())
    prototypes = sum(item.prototypes for item in stats.values())
    factor_elements = sum(item.factor_elements for item in stats.values())
    weighted_error = sum(
        item.neurons * item.gate_up_relative_l2_error
        for item in stats.values()
    ) / neurons
    worst = sorted(
        ({"name": name, **item.to_dict()} for name, item in stats.items()),
        key=lambda item: float(item["gate_up_relative_l2_error"]),
        reverse=True,
    )[:12]
    return {
        "layers": len(stats),
        "neurons": neurons,
        "prototypes": prototypes,
        "neuron_to_prototype_ratio": neurons / prototypes,
        "factor_elements": factor_elements,
        "factor_bytes": sum(item.factor_bytes for item in stats.values()),
        "neuron_weighted_gate_up_relative_l2_error": weighted_error,
        "worst_layers": worst,
    }


def main() -> None:
    args = parse_args()
    if min(
        args.tiny_prototypes,
        args.target_prototypes,
        args.projection_dim,
        args.cluster_iterations,
        args.factor_bits,
        args.tokens,
        args.tree_depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
    ) <= 0:
        raise SystemExit("positive integer controls are required")

    target, baseline = default_specs()
    target_budget = mlp_neuron_dictionary_budget(
        target=target,
        baseline=baseline,
        prototypes_per_layer=args.target_prototypes,
        factor_bits=args.factor_bits,
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

    fit_started = time.perf_counter()
    per_layer = replace_all_mlp_modules(
        model,
        prototypes=args.tiny_prototypes,
        projection_dim=args.projection_dim,
        iterations=args.cluster_iterations,
        factor_bits=args.factor_bits,
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - fit_started
    fit_summary = aggregate_fit_stats(per_layer)
    gc.collect()

    compiled_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=evaluation_tokens,
    )
    exact_teacher = teacher_summary(
        logits=exact_logits,
        exact_tokens=evaluation_tokens,
    )
    compiled_teacher = teacher_summary(
        logits=compiled_logits,
        exact_tokens=evaluation_tokens,
    )
    compiled_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    autonomous_prefix = common_prefix_length(compiled_tokens, evaluation_tokens)

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
    tiny_ratio = int(model.config.intermediate_size) / args.tiny_prototypes
    target_ratio = target.intermediate_size / args.target_prototypes
    qualifies = bool(
        target_budget.memory_pass
        and target_budget.partial_traffic_pass
        and target_budget.partial_latency_pass
        and compiled_teacher["top32_rate"] >= 0.95
        and last_alive_depth >= 1
    )
    payload = {
        "evidence_level": "E2 executable MLP neuron dictionary replacement",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tiny_prototypes_per_layer": args.tiny_prototypes,
        "projected_405b_prototypes_per_layer": args.target_prototypes,
        "factor_bits": args.factor_bits,
        "fit": fit_summary,
        "fit_seconds": fit_seconds,
        "compression_comparison": {
            "tiny_neurons_per_prototype": tiny_ratio,
            "target_neurons_per_prototype": target_ratio,
            "tiny_over_target_favorability": target_ratio / tiny_ratio,
        },
        "teacher_forced_exact_reference": exact_teacher,
        "teacher_forced_compiled": compiled_teacher,
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
        "405b_partial_mlp_budget": target_budget.to_dict(),
        "contract": (
            "Only exact within-layer neuron permutation symmetry is assumed. "
            "Gate/up prototypes are fitted from checkpoint weights and original "
            "down columns are summed by assignment. Attention and LM head remain "
            "exact to isolate MLP compressibility."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance neuron dictionary toward complete architecture"
            if qualifies
            else "reject tested MLP neuron dictionary point"
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
