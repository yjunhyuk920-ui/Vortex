from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.multistep_decision_trace import (
    CausalMLPDecisionTrace,
    collect_causal_multistep_mlp_traces,
)
from vortex_runtime.semantic_program_routing import (
    assign_semantic_states,
    block_perpendicular_ratio,
    build_block_bases,
    deterministic_projection,
    project_and_normalize_signatures,
    routing_run_statistics,
    semantic_program_budget,
    spherical_state_centroids,
    summarize_ratios,
)


def parse_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            states_text, rank_text = item.split(":", maxsplit=1)
            states = int(states_text)
            rank = int(rank_text)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(
                "configurations must use states:rank entries"
            ) from error
        if min(states, rank) <= 0:
            raise argparse.ArgumentTypeError("states and rank must be positive")
        pair = (states, rank)
        if pair not in configs:
            configs.append(pair)
    if not configs:
        raise argparse.ArgumentTypeError("at least one configuration is required")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a causal semantic-state router can select compact "
            "activation/dual programs that cover disjoint warm-decode traces."
        )
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("experiments/semantic_program_prompts.json"),
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--signature-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=[(4, 2), (8, 2), (8, 4), (16, 4), (16, 8)],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tinyllama_1_1b_semantic_program_routing.json"),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install the experiment dependencies with pip install transformers sentencepiece"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def load_prompt_groups(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    build = payload.get("build")
    evaluation = payload.get("eval")
    if not isinstance(build, list) or not isinstance(evaluation, list):
        raise ValueError("prompt file must contain build and eval lists")
    for group_name, group in (("build", build), ("eval", evaluation)):
        if not group:
            raise ValueError(f"{group_name} prompt group must not be empty")
        for item in group:
            if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
                raise ValueError(f"every {group_name} item needs id and prompt")
    build_prompts = [
        {"id": str(item["id"]), "prompt": str(item["prompt"])} for item in build
    ]
    eval_prompts = [
        {"id": str(item["id"]), "prompt": str(item["prompt"])}
        for item in evaluation
    ]
    build_ids = {item["id"] for item in build_prompts}
    eval_ids = {item["id"] for item in eval_prompts}
    if build_ids & eval_ids:
        raise ValueError("build and evaluation prompt ids must be disjoint")
    return build_prompts, eval_prompts


def load_model(
    AutoModelForCausalLM: Any,
    *,
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    model.to(device)
    model.eval()
    return model


def collect_group(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, str]],
    steps: int,
    device: torch.device,
    group_name: str,
) -> tuple[list[CausalMLPDecisionTrace], list[list[CausalMLPDecisionTrace]], list[dict[str, Any]]]:
    flat: list[CausalMLPDecisionTrace] = []
    sequences: list[list[CausalMLPDecisionTrace]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(prompts, start=1):
        print(
            json.dumps(
                {
                    "event": "collect_trace",
                    "group": group_name,
                    "index": index,
                    "total": len(prompts),
                    "id": item["id"],
                }
            ),
            flush=True,
        )
        traces = collect_causal_multistep_mlp_traces(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            steps=steps,
            device=device,
        )
        sequences.append(traces)
        flat.extend(traces)
        diagnostics.append(
            {
                "id": item["id"],
                "prompt_tokens": traces[0].prompt_tokens,
                "steps": len(traces),
                "winner_tokens": [trace.winner_token for trace in traces],
                "minimum_margin": min(trace.exact_margin for trace in traces),
                "mean_margin": sum(trace.exact_margin for trace in traces) / len(traces),
                "maximum_margin": max(trace.exact_margin for trace in traces),
            }
        )
    return flat, sequences, diagnostics


def build_state_bases(
    *,
    build_traces: list[CausalMLPDecisionTrace],
    labels: torch.Tensor,
    states: int,
    layers: int,
    block_size: int,
    rank: int,
) -> tuple[list[list[tuple[list[torch.Tensor], list[torch.Tensor]]]], list[int]]:
    state_programs: list[list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = []
    occupancy: list[int] = []
    for state in range(states):
        member_indices = torch.nonzero(labels == state, as_tuple=False).reshape(-1)
        members = [build_traces[int(index.item())] for index in member_indices]
        occupancy.append(len(members))
        if not members:
            raise RuntimeError("router produced an empty build state")
        layer_programs: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        for layer_index in range(layers):
            activation_bases = build_block_bases(
                [trace.activations[layer_index] for trace in members],
                block_size=block_size,
                rank=rank,
            )
            dual_bases = build_block_bases(
                [trace.output_duals[layer_index] for trace in members],
                block_size=block_size,
                rank=rank,
            )
            layer_programs.append((activation_bases, dual_bases))
        state_programs.append(layer_programs)
    return state_programs, occupancy


def evaluate_configuration(
    *,
    build_traces: list[CausalMLPDecisionTrace],
    eval_traces: list[CausalMLPDecisionTrace],
    eval_sequences: list[list[CausalMLPDecisionTrace]],
    build_signatures: torch.Tensor,
    eval_signatures: torch.Tensor,
    states: int,
    rank: int,
    block_size: int,
    layers: int,
) -> dict[str, Any]:
    if states > len(build_traces):
        raise ValueError("state count exceeds build traces")
    centroids, build_labels = spherical_state_centroids(
        build_signatures,
        states=states,
    )
    eval_labels, eval_similarity = assign_semantic_states(eval_signatures, centroids)
    programs, build_occupancy = build_state_bases(
        build_traces=build_traces,
        labels=build_labels,
        states=states,
        layers=layers,
        block_size=block_size,
        rank=rank,
    )

    activation_ratios: list[float] = []
    dual_ratios: list[float] = []
    per_prompt: list[dict[str, Any]] = []
    labels_by_sequence: list[list[int]] = []
    eval_occupancy = [0 for _ in range(states)]
    offset = 0
    for sequence in eval_sequences:
        sequence_labels = eval_labels[offset : offset + len(sequence)].tolist()
        labels_by_sequence.append([int(label) for label in sequence_labels])
        prompt_activation: list[float] = []
        prompt_dual: list[float] = []
        for local_index, trace in enumerate(sequence):
            state = int(sequence_labels[local_index])
            eval_occupancy[state] += 1
            layer_programs = programs[state]
            for layer_index in range(layers):
                activation_bases, dual_bases = layer_programs[layer_index]
                activation_ratio = block_perpendicular_ratio(
                    trace.activations[layer_index],
                    activation_bases,
                    block_size=block_size,
                )
                dual_ratio = block_perpendicular_ratio(
                    trace.output_duals[layer_index],
                    dual_bases,
                    block_size=block_size,
                )
                activation_ratios.append(activation_ratio)
                dual_ratios.append(dual_ratio)
                prompt_activation.append(activation_ratio)
                prompt_dual.append(dual_ratio)
        per_prompt.append(
            {
                "prompt": sequence[0].prompt,
                "routed_states": [int(label) for label in sequence_labels],
                "activation": summarize_ratios(prompt_activation).to_dict(),
                "dual": summarize_ratios(prompt_dual).to_dict(),
            }
        )
        offset += len(sequence)
    if offset != len(eval_traces):
        raise RuntimeError("evaluation sequence accounting mismatch")

    budget = semantic_program_budget(block_size=block_size, rank=rank)
    routing = routing_run_statistics(
        labels_by_sequence,
        active_program_gib=budget.active_program_gib,
    )
    activation = summarize_ratios(activation_ratios)
    dual = summarize_ratios(dual_ratios)
    router_distance = summarize_ratios(
        [max(0.0, 1.0 - float(value)) for value in eval_similarity.tolist()]
    )
    signature_compute_gflop = (
        2 * 16_384 * build_signatures.shape[1]
        + 2 * states * build_signatures.shape[1]
    ) / 1e9
    qualifies = bool(
        budget.active_program_pass
        and routing.projected_switch_traffic_gib_per_token
        <= budget.switch_traffic_limit_gib_per_token
        and activation.mean <= 0.10
        and dual.mean <= 0.10
        and activation.p95 <= 0.20
        and dual.p95 <= 0.20
        and min(build_occupancy) > 0
    )
    return {
        "states": states,
        "rank": rank,
        "block_size": block_size,
        "budget": budget.to_dict(),
        "host_bank_storage_gib": states * budget.active_program_gib,
        "signature_and_routing_compute_gflop_per_token": signature_compute_gflop,
        "build_state_occupancy": build_occupancy,
        "eval_state_occupancy": eval_occupancy,
        "router_cosine_distance": router_distance.to_dict(),
        "routing": routing.to_dict(),
        "activation_perpendicular": activation.to_dict(),
        "dual_perpendicular": dual.to_dict(),
        "per_prompt": per_prompt,
        "qualifies": qualifies,
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.signature_size, args.block_size) <= 0:
        raise SystemExit("steps, signature size, and block size must be positive")

    build_prompts, eval_prompts = load_prompt_groups(args.prompts)
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    model = load_model(
        AutoModelForCausalLM,
        model_name=args.model,
        dtype=dtype,
        device=device,
    )

    build_traces, build_sequences, build_diagnostics = collect_group(
        model=model,
        tokenizer=tokenizer,
        prompts=build_prompts,
        steps=args.steps,
        device=device,
        group_name="build",
    )
    eval_traces, eval_sequences, eval_diagnostics = collect_group(
        model=model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        steps=args.steps,
        device=device,
        group_name="eval",
    )
    del model
    gc.collect()

    hidden_size = int(build_traces[0].routing_hidden.numel())
    layers = len(build_traces[0].activations)
    if args.signature_size > hidden_size:
        raise SystemExit("signature size exceeds model hidden size")
    projection = deterministic_projection(hidden_size, args.signature_size)
    build_signatures = project_and_normalize_signatures(
        [trace.routing_hidden for trace in build_traces],
        projection,
    )
    eval_signatures = project_and_normalize_signatures(
        [trace.routing_hidden for trace in eval_traces],
        projection,
    )

    results: list[dict[str, Any]] = []
    for states, rank in args.configs:
        print(
            json.dumps(
                {"event": "evaluate_config", "states": states, "rank": rank}
            ),
            flush=True,
        )
        result = evaluate_configuration(
            build_traces=build_traces,
            eval_traces=eval_traces,
            eval_sequences=eval_sequences,
            build_signatures=build_signatures,
            eval_signatures=eval_signatures,
            states=states,
            rank=rank,
            block_size=args.block_size,
            layers=layers,
        )
        results.append(result)
        gc.collect()

    promoted = [
        {"states": result["states"], "rank": result["rank"]}
        for result in results
        if result["qualifies"]
    ]
    best_coverage = min(
        results,
        key=lambda item: (
            item["activation_perpendicular"]["mean"]
            + item["dual_perpendicular"]["mean"],
            item["routing"]["projected_switch_traffic_gib_per_token"],
        ),
    )
    payload = {
        "evidence_level": "E1/E2 causal semantic-state routing capacity gate",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "build_prompts": build_diagnostics,
        "eval_prompts": eval_diagnostics,
        "build_trace_count": len(build_traces),
        "eval_trace_count": len(eval_traces),
        "decode_steps_per_prompt": args.steps,
        "model_hidden_size": hidden_size,
        "decoder_layers": layers,
        "signature_size": args.signature_size,
        "block_size": args.block_size,
        "configurations": results,
        "promoted_configurations": promoted,
        "best_coverage_configuration": {
            "states": best_coverage["states"],
            "rank": best_coverage["rank"],
            "activation_mean": best_coverage["activation_perpendicular"]["mean"],
            "dual_mean": best_coverage["dual_perpendicular"]["mean"],
            "switch_traffic_gib_per_token": best_coverage["routing"][
                "projected_switch_traffic_gib_per_token"
            ],
        },
        "qualifies": bool(promoted),
        "decision": (
            "advance semantic router to full quantized signed residual compilation"
            if promoted
            else "reject tested precompiled semantic-state routing capacity"
        ),
        "next_candidate_if_rejected": (
            "multi-token decision program with an explicit >=247-token compute "
            "amortization gate; do not increase static K/rank without a new reuse mechanism"
        ),
        "contract": (
            "Routing uses only the previous completed token's final hidden state. "
            "Current MLP activations and exact top-two output duals are optimistic "
            "evaluation oracles. Program memory includes 4-bit signed coefficients, "
            "8-bit remainder norms, two 16-bit row scales, and 8-bit bases. Every "
            "initial program load and routed state switch is charged."
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
