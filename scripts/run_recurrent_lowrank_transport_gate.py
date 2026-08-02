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

from scripts.run_diagonal_transport_gate import (
    quantize_io_weights,
    quantized_representative_snapshots,
    teacher_forced_summary,
)
from scripts.run_full_rank_precision_point import (
    common_prefix_length,
    greedy_tokens,
    teacher_forced_logits,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.diagonal_transport import (
    fit_diagonal_transport,
    materialize_diagonal_transport,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.lowrank_transport import (
    fit_randomized_low_rank_residual,
    materialize_low_rank_correction,
)
from vortex_runtime.recurrent_layer_dictionary import recurrent_layer_schedule
from vortex_runtime.recurrent_lowrank_budget import recurrent_low_rank_budget
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    speculative_tree_verification_budget,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a resident Q4 recurrent dictionary with weight-only "
            "diagonal plus low-rank per-layer residual transports."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dictionary-bits", type=int, default=4)
    parser.add_argument("--residual-bits", type=int, default=8)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--unique-layers", type=int, default=3)
    parser.add_argument("--fit-iterations", type=int, default=2)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--power-iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--tree-depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--node-budget", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--memory-limit-gib", type=float, default=8.0)
    parser.add_argument("--effective-tops", type=float, default=160.0)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recurrent_lowrank_transport_gate.json"),
    )
    return parser.parse_args()


def apply_diagonal_lowrank_transports(
    *,
    layers: torch.nn.ModuleList,
    assignment: tuple[int, ...],
    snapshots: dict[int, dict[str, torch.Tensor]],
    diagonal_iterations: int,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
    residual_bits: int,
) -> dict[str, object]:
    if len(layers) != len(assignment):
        raise ValueError("assignment must cover every model layer")

    rows: list[dict[str, Any]] = []
    total_elements = 0
    weighted_representative_error = 0.0
    weighted_diagonal_error = 0.0
    weighted_corrected_error = 0.0
    factor_bytes = 0

    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            representative_index = assignment[layer_index]
            representative = snapshots[representative_index]
            for parameter_index, (name, parameter) in enumerate(
                layer.named_parameters()
            ):
                if parameter.ndim != 2:
                    continue
                representative_weight = representative[name]
                input_scale, output_scale, diagonal_stats = fit_diagonal_transport(
                    target_weight=parameter,
                    representative_weight=representative_weight,
                    iterations=diagonal_iterations,
                    metadata_bits=16,
                )
                diagonal_base = materialize_diagonal_transport(
                    representative_weight=representative_weight,
                    input_scale=input_scale,
                    output_scale=output_scale,
                )
                effective_rank = min(rank, min(parameter.shape))
                left, right, lowrank_stats = fit_randomized_low_rank_residual(
                    target_weight=parameter,
                    base_weight=diagonal_base,
                    rank=effective_rank,
                    oversample=oversample,
                    power_iterations=power_iterations,
                    seed=seed + layer_index * 101 + parameter_index,
                    factor_bits=residual_bits,
                )
                corrected = materialize_low_rank_correction(
                    base_weight=diagonal_base,
                    left=left,
                    right=right,
                )
                parameter.copy_(corrected.to(parameter.dtype))

                elements = parameter.numel()
                total_elements += elements
                weighted_representative_error += (
                    elements * diagonal_stats.baseline_relative_l2_error
                )
                weighted_diagonal_error += (
                    elements * diagonal_stats.adapted_relative_l2_error
                )
                weighted_corrected_error += (
                    elements * lowrank_stats.corrected_relative_l2_error
                )
                factor_bytes += lowrank_stats.factor_bytes
                row = {
                    "layer": layer_index,
                    "representative_layer": representative_index,
                    "name": name,
                    "elements": elements,
                    "representative_error": diagonal_stats.baseline_relative_l2_error,
                    "diagonal_error": diagonal_stats.adapted_relative_l2_error,
                    "corrected_error": lowrank_stats.corrected_relative_l2_error,
                    "error_reduction_from_diagonal": lowrank_stats.relative_error_reduction,
                    "rank": lowrank_stats.rank,
                    "factor_bytes": lowrank_stats.factor_bytes,
                }
                rows.append(row)
                del input_scale, output_scale, diagonal_base, left, right, corrected

    rows.sort(key=lambda item: float(item["corrected_error"]), reverse=True)
    denominator = max(1, total_elements)
    return {
        "tensors": len(rows),
        "elements": total_elements,
        "factor_bytes_tinyllama": factor_bytes,
        "element_weighted_representative_error": (
            weighted_representative_error / denominator
        ),
        "element_weighted_diagonal_error": weighted_diagonal_error / denominator,
        "element_weighted_corrected_error": weighted_corrected_error / denominator,
        "worst_tensors": rows[:24],
    }


def main() -> None:
    args = parse_args()
    if min(
        args.rank,
        args.unique_layers,
        args.fit_iterations,
        args.tokens,
        args.tree_depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.row_chunk,
    ) <= 0:
        raise SystemExit("positive integer controls are required")

    target, baseline = default_specs()
    projected_budget = recurrent_low_rank_budget(
        target=target,
        baseline=baseline,
        unique_layers=args.unique_layers,
        rank=args.rank,
        dictionary_bits=args.dictionary_bits,
        residual_bits=args.residual_bits,
        metadata_bits=16,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
        effective_tops=args.effective_tops,
    )
    if not projected_budget.pass_all:
        raise SystemExit("requested low-rank point exceeds the 405B resident envelope")

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

    layers = model.model.layers
    schedule = recurrent_layer_schedule(
        total_layers=len(layers),
        unique_layers=args.unique_layers,
        representative_strategy="uniform",
        assignment_strategy="nearest",
    )
    snapshots, representative_stats = quantized_representative_snapshots(
        layers=layers,
        representative_indices=schedule.representative_indices,
        bits=args.dictionary_bits,
        row_chunk=args.row_chunk,
    )
    io_stats = quantize_io_weights(
        model=model,
        bits=args.dictionary_bits,
        row_chunk=args.row_chunk,
    )
    fit_stats = apply_diagonal_lowrank_transports(
        layers=layers,
        assignment=schedule.assignment,
        snapshots=snapshots,
        diagonal_iterations=args.fit_iterations,
        rank=args.rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
        residual_bits=args.residual_bits,
    )
    del snapshots
    gc.collect()

    adapted_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=evaluation_tokens,
    )
    adapted_teacher = teacher_forced_summary(
        logits=adapted_logits,
        exact_tokens=evaluation_tokens,
    )
    exact_teacher = teacher_forced_summary(
        logits=exact_logits,
        exact_tokens=evaluation_tokens,
    )
    adapted_autoregressive = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    autoregressive_prefix = common_prefix_length(
        adapted_autoregressive,
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
    reached_depth = max((len(sequence) for sequence in retained), default=0)
    optimistic_committed = (
        longest_reference_prefix(retained, reference) if retained else 0
    )
    target_lower_bound = speculative_tree_verification_budget(
        target=target,
        baseline=baseline,
        hot_bits=4,
        tree_nodes=max(1, tree_nodes),
        tree_depth=max(1, reached_depth),
        committed_tokens=optimistic_committed,
        host_to_device_gib_s=args.host_to_device_gib_s,
        hot_effective_tops=args.effective_tops,
    )

    qualifies = bool(
        optimistic_committed == args.tree_depth
        and target_lower_bound.observed_serialized_pass
    )
    payload = {
        "evidence_level": (
            "E2 recurrent Q4 dictionary plus diagonal and FP8 low-rank residuals"
        ),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "rank": args.rank,
        "dictionary_bits": args.dictionary_bits,
        "residual_bits": args.residual_bits,
        "schedule": schedule.to_dict(),
        "representative_quantization": representative_stats,
        "io_quantization": io_stats,
        "fit": fit_stats,
        "teacher_forced_exact_reference": exact_teacher,
        "teacher_forced_corrected": adapted_teacher,
        "autoregressive_exact_prefix": autoregressive_prefix,
        "tree": {
            "requested_depth": args.tree_depth,
            "retained_tree_nodes": tree_nodes,
            "reached_depth": reached_depth,
            "optimistic_committed_exact_prefix": optimistic_committed,
            "exact_path_last_alive_depth": exact_path_last_alive_depth,
            "depth_rows": depth_rows,
        },
        "405b_projection": {
            "resident_budget": projected_budget.to_dict(),
            "q4_target_side_lower_bound": target_lower_bound.to_dict(),
        },
        "contract": (
            "All correction factors are built automatically from checkpoint "
            "weights only. No activations, labels, gradients or learned adapters "
            "are used. TinyLlama materializes corrected weights only to test the "
            "factorized resident representation."
        ),
        "qualifies": qualifies,
        "decision": (
            "survive resident low-rank path gate"
            if qualifies
            else "reject tested resident low-rank point"
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
