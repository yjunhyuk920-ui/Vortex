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

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_mlp_heavy_hitter_oracle import (
    common_prefix_length,
    teacher_forced_logits,
    teacher_summary,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.adjoint_heavy_hitter import (
    AdjointAllocation,
    allocate_global_neuron_budget,
    replace_llama_mlp_with_count_allocation,
    uniform_neuron_allocation,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.mlp_heavy_hitter import (
    aggregate_heavy_hitter_stats,
    mlp_heavy_hitter_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate a fixed exact-neuron budget across Llama MLP layers using "
            "exact top-two logit-margin adjoints from a disjoint calibration prompt."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--total-fraction", type=float, required=True)
    parser.add_argument("--calibration-tokens", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=16)
    parser.add_argument("--minimum-per-layer", type=int, default=1)
    parser.add_argument(
        "--calibration-prompt",
        default=(
            "Explain how a database transaction preserves consistency when two "
            "concurrent services update related records."
        ),
    )
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adjoint_heavy_hitter_allocation.json"),
    )
    return parser.parse_args()


def freeze_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def combined_teacher_inputs(
    *,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prompt_ids = encoded["input_ids"]
    continuation = exact_tokens[:, :-1].to(prompt_ids.device)
    result = {"input_ids": torch.cat((prompt_ids, continuation), dim=1)}
    prompt_mask = encoded.get("attention_mask")
    if prompt_mask is not None:
        result["attention_mask"] = torch.cat(
            (
                prompt_mask,
                torch.ones_like(
                    continuation,
                    dtype=prompt_mask.dtype,
                    device=prompt_mask.device,
                ),
            ),
            dim=1,
        )
    return result


def calibration_margin_scores(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")

    activated_inputs: list[torch.Tensor | None] = [None] * len(layers)
    down_outputs: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[Any] = []

    for layer_index, layer in enumerate(layers):
        down = layer.mlp.down_proj

        def pre_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            index: int = layer_index,
        ) -> None:
            del module
            activated_inputs[index] = inputs[0]

        def output_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> torch.Tensor:
            del module, inputs
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()
            down_outputs[index] = output
            return output

        handles.append(down.register_forward_pre_hook(pre_hook))
        handles.append(down.register_forward_hook(output_hook))

    teacher_inputs = combined_teacher_inputs(
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    model.zero_grad(set_to_none=True)
    output = model(
        **teacher_inputs,
        use_cache=False,
        return_dict=True,
    )
    prompt_length = encoded["input_ids"].shape[1]
    start = prompt_length - 1
    end = start + exact_tokens.shape[1]
    continuation_logits = output.logits[:, start:end, :]
    if continuation_logits.shape[1] != exact_tokens.shape[1]:
        raise RuntimeError("failed to align calibration logits")

    top_two = torch.topk(continuation_logits.detach(), k=2, dim=-1).indices
    top_one_logits = torch.gather(
        continuation_logits,
        dim=-1,
        index=top_two[..., :1],
    ).squeeze(-1)
    runner_up_logits = torch.gather(
        continuation_logits,
        dim=-1,
        index=top_two[..., 1:2],
    ).squeeze(-1)
    margins = top_one_logits - runner_up_logits
    scalar = margins.sum()
    scalar.backward()

    layer_scores: list[torch.Tensor] = []
    layer_diagnostics: list[dict[str, int | float]] = []
    for layer_index, layer in enumerate(layers):
        activated = activated_inputs[layer_index]
        down_output = down_outputs[layer_index]
        if activated is None or down_output is None or down_output.grad is None:
            raise RuntimeError(f"missing calibration tensors for layer {layer_index}")
        activation_value = activated.detach().to("cpu", torch.float32)
        output_gradient = down_output.grad.detach().to("cpu", torch.float32)
        down_weight = layer.mlp.down_proj.weight.detach().to("cpu", torch.float32)
        adjoint_drive = torch.matmul(output_gradient, down_weight)
        contribution = torch.abs(activation_value * adjoint_drive)
        neuron_scores = contribution.reshape(-1, contribution.shape[-1]).sum(dim=0)
        layer_scores.append(neuron_scores)
        layer_diagnostics.append(
            {
                "layer": layer_index,
                "neurons": int(neuron_scores.numel()),
                "score_sum": float(neuron_scores.sum().item()),
                "score_max": float(neuron_scores.max().item()),
                "nonzero_scores": int(torch.count_nonzero(neuron_scores).item()),
                "mlp_output_gradient_norm": float(
                    torch.linalg.vector_norm(output_gradient).item()
                ),
            }
        )

    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    diagnostics: dict[str, Any] = {
        "calibration_positions": int(exact_tokens.shape[1]),
        "mean_exact_top2_margin": float(margins.detach().mean().item()),
        "minimum_exact_top2_margin": float(margins.detach().min().item()),
        "maximum_exact_top2_margin": float(margins.detach().max().item()),
        "layers": layer_diagnostics,
    }
    return layer_scores, diagnostics


def evaluate_count_allocation(
    *,
    AutoModelForCausalLM: Any,
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
    layer_counts: tuple[int, ...],
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    modules = replace_llama_mlp_with_count_allocation(
        model,
        layer_counts=layer_counts,
    )
    for module in modules:
        module.reset_statistics()
    logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    teacher = teacher_summary(logits=logits, exact_tokens=exact_tokens)
    autonomous = greedy_tokens(
        model=model,
        encoded=encoded,
        count=exact_tokens.shape[1],
    ).to("cpu")
    prefix = common_prefix_length(autonomous, exact_tokens)
    aggregate = aggregate_heavy_hitter_stats(modules)
    payload: dict[str, Any] = {
        "layer_counts": list(layer_counts),
        "teacher_forced": teacher,
        "autonomous_exact_prefix": prefix,
        "autonomous_exact_match_rate": float(
            torch.eq(autonomous, exact_tokens).float().mean().item()
        ),
        "oracle_statistics": aggregate.to_dict(),
    }
    del model, modules, logits, autonomous
    gc.collect()
    return payload


def main() -> None:
    args = parse_args()
    if not 0 < args.total_fraction <= 1:
        raise SystemExit("total fraction must be in (0, 1]")
    if min(args.calibration_tokens, args.eval_tokens, args.minimum_per_layer) <= 0:
        raise SystemExit("token counts and minimum-per-layer must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration_encoded = encode_prompt(tokenizer, args.calibration_prompt, device)
    eval_encoded = encode_prompt(tokenizer, args.eval_prompt, device)

    started = time.perf_counter()
    calibration_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    calibration_model.to(device)
    calibration_model.eval()
    freeze_parameters(calibration_model)
    calibration_tokens = greedy_tokens(
        model=calibration_model,
        encoded=calibration_encoded,
        count=args.calibration_tokens,
    )
    layer_scores, calibration_diagnostics = calibration_margin_scores(
        model=calibration_model,
        encoded=calibration_encoded,
        exact_tokens=calibration_tokens,
    )
    layers = len(layer_scores)
    intermediate = int(layer_scores[0].numel())
    total_neurons = layers * max(1, int(torch.ceil(torch.tensor(
        intermediate * args.total_fraction
    )).item()))
    allocation: AdjointAllocation = allocate_global_neuron_budget(
        layer_scores,
        total_neurons=total_neurons,
        minimum_per_layer=args.minimum_per_layer,
    )
    uniform_counts = uniform_neuron_allocation(
        layers=layers,
        intermediate_neurons=intermediate,
        total_neurons=total_neurons,
    )
    del calibration_model, calibration_tokens, layer_scores
    gc.collect()

    exact_eval_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    exact_eval_model.to(device)
    exact_eval_model.eval()
    exact_eval_tokens = greedy_tokens(
        model=exact_eval_model,
        encoded=eval_encoded,
        count=args.eval_tokens,
    ).to("cpu")
    del exact_eval_model
    gc.collect()

    uniform = evaluate_count_allocation(
        AutoModelForCausalLM=AutoModelForCausalLM,
        model_name=args.model,
        dtype=dtype,
        device=device,
        encoded=eval_encoded,
        exact_tokens=exact_eval_tokens,
        layer_counts=uniform_counts,
    )
    adjoint = evaluate_count_allocation(
        AutoModelForCausalLM=AutoModelForCausalLM,
        model_name=args.model,
        dtype=dtype,
        device=device,
        encoded=eval_encoded,
        exact_tokens=exact_eval_tokens,
        layer_counts=allocation.layer_counts,
    )

    actual_fraction = total_neurons / (layers * intermediate)
    target, _ = default_specs()
    budget = mlp_heavy_hitter_budget(
        target=target,
        selected_fraction=actual_fraction,
        source_bits=16,
        selector_bits_per_neuron=32,
        partial_traffic_limit_gib=1.6,
    )
    improves_top32 = (
        adjoint["teacher_forced"]["top32_rate"]
        > uniform["teacher_forced"]["top32_rate"]
    )
    improves_prefix = (
        adjoint["autonomous_exact_prefix"]
        > uniform["autonomous_exact_prefix"]
    )
    qualifies = bool(
        budget.partial_traffic_pass
        and adjoint["teacher_forced"]["top32_rate"] >= 0.95
        and adjoint["autonomous_exact_prefix"] >= 4
    )
    payload = {
        "evidence_level": "E2 disjoint-prompt adjoint heavy-hitter allocation oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "calibration_prompt_tokens": int(calibration_encoded["input_ids"].shape[1]),
        "eval_prompt_tokens": int(eval_encoded["input_ids"].shape[1]),
        "calibration_tokens": args.calibration_tokens,
        "eval_tokens": args.eval_tokens,
        "requested_total_fraction": args.total_fraction,
        "actual_total_fraction": actual_fraction,
        "total_selected_neurons": total_neurons,
        "calibration": calibration_diagnostics,
        "adjoint_allocation": allocation.to_dict(),
        "uniform_allocation": list(uniform_counts),
        "uniform_eval": uniform,
        "adjoint_eval": adjoint,
        "comparison": {
            "top1_delta": (
                adjoint["teacher_forced"]["top1_rate"]
                - uniform["teacher_forced"]["top1_rate"]
            ),
            "top32_delta": (
                adjoint["teacher_forced"]["top32_rate"]
                - uniform["teacher_forced"]["top32_rate"]
            ),
            "autonomous_prefix_delta": (
                adjoint["autonomous_exact_prefix"]
                - uniform["autonomous_exact_prefix"]
            ),
            "improves_top32": improves_top32,
            "improves_prefix": improves_prefix,
        },
        "projected_405b_partial_mlp_budget": budget.to_dict(),
        "contract": (
            "Only layer counts are calibrated. Neuron allocation uses exact top-two "
            "margin adjoints from a calibration prompt, while all quality metrics "
            "are measured on a disjoint evaluation prompt. The evaluation selector "
            "remains the optimistic exact-activation oracle, so the result is an "
            "upper bound for a compact causal runtime."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance nonuniform heavy-hitter selector and tail proof"
            if qualifies
            else "reject tested adjoint allocation or revise the unit-cost model"
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
