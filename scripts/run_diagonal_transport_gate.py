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

from scripts.run_full_rank_precision_point import (
    common_prefix_length,
    greedy_tokens,
    teacher_forced_logits,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from scripts.run_q4_speculative_tree_gate import _next_logits, _trim_to_node_budget
from vortex_runtime.candidate_coverage import token_rank
from vortex_runtime.diagonal_transport import (
    diagonal_transport_metadata_budget,
    fit_diagonal_transport,
    materialize_diagonal_transport,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize
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
            "Fit training-free per-layer diagonal transports around three shared "
            "Q4 representative layers and measure causal target preservation."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--unique-layers", type=int, default=3)
    parser.add_argument(
        "--representative-strategy",
        choices=("front", "uniform", "edge"),
        default="uniform",
    )
    parser.add_argument(
        "--assignment-strategy",
        choices=("nearest", "cyclic"),
        default="nearest",
    )
    parser.add_argument("--fit-iterations", type=int, default=2)
    parser.add_argument("--scale-limit", type=float, default=16.0)
    parser.add_argument("--metadata-bits", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--tree-depth", type=int, default=12)
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
        default=Path("diagonal_transport_gate.json"),
    )
    return parser.parse_args()


def quantized_representative_snapshots(
    *,
    layers: torch.nn.ModuleList,
    representative_indices: tuple[int, ...],
    bits: int,
    row_chunk: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, object]]:
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    tensor_stats: list[dict[str, Any]] = []
    for representative_index in representative_indices:
        layer = layers[representative_index]
        snapshot: dict[str, torch.Tensor] = {}
        for name, parameter in layer.named_parameters():
            if parameter.ndim != 2:
                continue
            restored, stats = symmetric_per_row_fake_quantize(
                parameter,
                bits=bits,
                source_bits=16,
                name=f"layers.{representative_index}.{name}",
                row_chunk=row_chunk,
            )
            snapshot[name] = restored.to("cpu", torch.float32).contiguous()
            tensor_stats.append(stats.to_dict())
        snapshots[representative_index] = snapshot
    return snapshots, {
        "representatives": list(representative_indices),
        "quantized_tensors": len(tensor_stats),
        "tensor_stats": tensor_stats,
    }


def quantize_io_weights(
    *,
    model: torch.nn.Module,
    bits: int,
    row_chunk: int,
) -> list[dict[str, Any]]:
    modules = (
        ("model.embed_tokens", model.model.embed_tokens),
        ("lm_head", model.lm_head),
    )
    seen: set[tuple[int, int]] = set()
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in modules:
            weight = module.weight
            identity = (
                weight.untyped_storage().data_ptr(),
                weight.storage_offset(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            restored, stats = symmetric_per_row_fake_quantize(
                weight,
                bits=bits,
                source_bits=16,
                name=f"{name}.weight",
                row_chunk=row_chunk,
            )
            weight.copy_(restored.to(device=weight.device, dtype=weight.dtype))
            result.append(stats.to_dict())
            del restored
    return result


def apply_diagonal_transports(
    *,
    layers: torch.nn.ModuleList,
    assignment: tuple[int, ...],
    snapshots: dict[int, dict[str, torch.Tensor]],
    iterations: int,
    scale_limit: float,
    metadata_bits: int,
) -> dict[str, object]:
    if len(layers) != len(assignment):
        raise ValueError("assignment must cover every model layer")

    rows: list[dict[str, Any]] = []
    weighted_baseline = 0.0
    weighted_adapted = 0.0
    total_elements = 0
    metadata_bytes = 0
    exact_vector_bytes = 0

    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            representative_index = assignment[layer_index]
            representative = snapshots[representative_index]
            for name, parameter in layer.named_parameters():
                if parameter.ndim == 2:
                    if name not in representative:
                        raise RuntimeError(
                            f"missing representative tensor {representative_index}:{name}"
                        )
                    input_scale, output_scale, stats = fit_diagonal_transport(
                        target_weight=parameter,
                        representative_weight=representative[name],
                        iterations=iterations,
                        scale_limit=scale_limit,
                        metadata_bits=metadata_bits,
                    )
                    adapted = materialize_diagonal_transport(
                        representative_weight=representative[name],
                        input_scale=input_scale,
                        output_scale=output_scale,
                    )
                    parameter.copy_(adapted.to(parameter.dtype))
                    elements = parameter.numel()
                    total_elements += elements
                    weighted_baseline += elements * stats.baseline_relative_l2_error
                    weighted_adapted += elements * stats.adapted_relative_l2_error
                    metadata_bytes += stats.metadata_bytes
                    row = stats.to_dict()
                    row.update(
                        {
                            "layer": layer_index,
                            "representative_layer": representative_index,
                            "name": name,
                            "elements": elements,
                        }
                    )
                    rows.append(row)
                    del adapted, input_scale, output_scale
                elif parameter.ndim == 1:
                    exact_vector_bytes += parameter.numel() * metadata_bits // 8

    rows.sort(
        key=lambda item: float(item["adapted_relative_l2_error"]),
        reverse=True,
    )
    return {
        "tensors": len(rows),
        "elements": total_elements,
        "metadata_bytes": metadata_bytes,
        "exact_vector_bytes": exact_vector_bytes,
        "total_metadata_bytes": metadata_bytes + exact_vector_bytes,
        "element_weighted_baseline_relative_error": (
            weighted_baseline / max(1, total_elements)
        ),
        "element_weighted_adapted_relative_error": (
            weighted_adapted / max(1, total_elements)
        ),
        "worst_tensors": rows[:24],
    }


def teacher_forced_summary(
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
        "top1_rate": matches / len(ranks),
        "top32_rate": sum(rank <= 32 for rank in ranks) / len(ranks),
        "mean_exact_token_rank": sum(ranks) / len(ranks),
        "maximum_exact_token_rank": max(ranks),
        "ranks": ranks,
    }


def main() -> None:
    args = parse_args()
    if not 2 <= args.bits < 16:
        raise SystemExit("bits must be in [2, 16)")
    if min(
        args.unique_layers,
        args.fit_iterations,
        args.metadata_bits,
        args.tokens,
        args.tree_depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.row_chunk,
    ) <= 0:
        raise SystemExit("all integer controls must be positive")

    target, baseline = default_specs()
    recurrent_budget = recurrent_draft_budget(
        target=target,
        baseline=baseline,
        unique_layers=args.unique_layers,
        weight_bits=args.bits,
        tie_word_embeddings=False,
        workspace_gib=args.workspace_gib,
        memory_limit_gib=args.memory_limit_gib,
        effective_tops=args.draft_effective_tops,
    )
    metadata_budget = diagonal_transport_metadata_budget(
        model=target,
        metadata_bits=args.metadata_bits,
    )
    total_projected_gib = (
        recurrent_budget.memory.total_gib + metadata_budget.metadata_gib
    )
    projected_memory_pass = total_projected_gib <= args.memory_limit_gib

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
        representative_strategy=args.representative_strategy,
        assignment_strategy=args.assignment_strategy,
    )
    snapshots, representative_stats = quantized_representative_snapshots(
        layers=layers,
        representative_indices=schedule.representative_indices,
        bits=args.bits,
        row_chunk=args.row_chunk,
    )
    io_stats = quantize_io_weights(
        model=model,
        bits=args.bits,
        row_chunk=args.row_chunk,
    )
    transport_stats = apply_diagonal_transports(
        layers=layers,
        assignment=schedule.assignment,
        snapshots=snapshots,
        iterations=args.fit_iterations,
        scale_limit=args.scale_limit,
        metadata_bits=args.metadata_bits,
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
        hot_effective_tops=args.target_hot_effective_tops,
    )

    qualifies = bool(
        projected_memory_pass
        and recurrent_budget.compute_pass
        and optimistic_committed == args.tree_depth
        and target_lower_bound.observed_serialized_pass
    )
    payload = {
        "evidence_level": (
            "E2 full-depth Q4 dictionary with training-free diagonal transports"
        ),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "schedule": schedule.to_dict(),
        "fit_iterations": args.fit_iterations,
        "metadata_bits": args.metadata_bits,
        "representative_quantization": representative_stats,
        "io_quantization": io_stats,
        "transport_fit": transport_stats,
        "teacher_forced_exact_reference": exact_teacher,
        "teacher_forced_adapted": adapted_teacher,
        "autoregressive_exact_prefix": autoregressive_prefix,
        "tree": {
            "requested_depth": args.tree_depth,
            "top_k": args.top_k,
            "beam_width": args.beam_width,
            "node_budget": args.node_budget,
            "retained_tree_nodes": tree_nodes,
            "reached_depth": reached_depth,
            "optimistic_committed_exact_prefix": optimistic_committed,
            "exact_path_last_alive_depth": exact_path_last_alive_depth,
            "depth_rows": depth_rows,
        },
        "405b_projection": {
            "recurrent_dictionary": recurrent_budget.to_dict(),
            "transport_metadata": metadata_budget.to_dict(),
            "total_gib_including_workspace": total_projected_gib,
            "memory_pass": projected_memory_pass,
            "q4_target_side_lower_bound": target_lower_bound.to_dict(),
        },
        "contract": (
            "Representative matrices are Q4 target weights. Every original layer "
            "receives only deterministic input/output scale vectors fitted from "
            "its unchanged weights. No activation data, labels or training are used."
        ),
        "qualifies": qualifies,
        "decision": (
            "survive diagonal transport lower bound; build factorized runtime"
            if qualifies
            else "reject tested diagonal transport tree point"
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
