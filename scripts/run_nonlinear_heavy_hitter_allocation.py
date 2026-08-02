from __future__ import annotations

import argparse
import gc
import json
from math import ceil
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_mlp_heavy_hitter_oracle import (
    common_prefix_length,
    teacher_forced_logits,
    teacher_summary,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.adjoint_heavy_hitter import (
    replace_llama_mlp_with_count_allocation,
    uniform_neuron_allocation,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.mlp_heavy_hitter import (
    OracleHeavyHitterSwiGLU,
    aggregate_heavy_hitter_stats,
    mlp_heavy_hitter_budget,
)
from vortex_runtime.nonlinear_heavy_hitter import (
    LayerDamagePoint,
    solve_nonlinear_allocation,
)


def parse_int_list(value: str) -> list[int]:
    items = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("count options must be positive integers")
    return items


def parse_float_list(value: str) -> list[float]:
    items = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not items or any(not 0 < item <= 1 for item in items):
        raise argparse.ArgumentTypeError("fractions must lie in (0, 1]")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact final-logit damage curves for one sparse MLP layer at "
            "a time, solve a discrete neuron allocation, and validate it on a "
            "disjoint prompt against a uniform allocation with the same total."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count-options", type=parse_int_list, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--total-fractions", type=parse_float_list, default=[0.001, 0.0025, 0.005])
    parser.add_argument("--calibration-tokens", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=16)
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
        default=Path("nonlinear_heavy_hitter_allocation.json"),
    )
    return parser.parse_args()


def exact_token_cross_entropy(
    logits: torch.Tensor,
    exact_tokens: torch.Tensor,
) -> float:
    vocabulary = logits.shape[-1]
    return float(
        F.cross_entropy(
            logits.reshape(-1, vocabulary),
            exact_tokens.to(logits.device).reshape(-1),
            reduction="mean",
        ).item()
    )


def measure_layer_damage_curves(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
    count_options: list[int],
) -> tuple[list[list[LayerDamagePoint]], dict[str, Any]]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")

    exact_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    baseline_teacher = teacher_summary(
        logits=exact_logits,
        exact_tokens=exact_tokens,
    )
    baseline_ce = exact_token_cross_entropy(exact_logits, exact_tokens)
    curves: list[list[LayerDamagePoint]] = []
    raw_layers: list[dict[str, Any]] = []

    for layer_index, layer in enumerate(layers):
        original_mlp = layer.mlp
        intermediate = int(original_mlp.gate_proj.out_features)
        valid_counts = sorted({min(count, intermediate) for count in count_options})
        layer_curve: list[LayerDamagePoint] = []
        raw_points: list[dict[str, Any]] = []
        for count in valid_counts:
            replacement = OracleHeavyHitterSwiGLU(
                gate_proj=original_mlp.gate_proj,
                up_proj=original_mlp.up_proj,
                down_proj=original_mlp.down_proj,
                act_fn=original_mlp.act_fn,
                selected_fraction=count / intermediate,
            )
            layer.mlp = replacement
            replacement.reset_statistics()
            logits = teacher_forced_logits(
                model=model,
                encoded=encoded,
                exact_tokens=exact_tokens,
            )
            teacher = teacher_summary(logits=logits, exact_tokens=exact_tokens)
            cross_entropy = exact_token_cross_entropy(logits, exact_tokens)
            statistics = replacement.statistics()
            damage = max(0.0, cross_entropy - baseline_ce)
            point = LayerDamagePoint(
                selected_neurons=replacement.selected_neurons,
                damage=damage,
                top1_rate=float(teacher["top1_rate"]),
                top32_rate=float(teacher["top32_rate"]),
                output_error=float(statistics["mean_output_relative_l2_error"]),
            )
            layer_curve.append(point)
            raw_points.append(
                {
                    **point.to_dict(),
                    "cross_entropy": cross_entropy,
                    "mean_exact_token_rank": teacher["mean_exact_token_rank"],
                    "maximum_exact_token_rank": teacher["maximum_exact_token_rank"],
                    "score_coverage": statistics["mean_score_coverage"],
                    "unique_neuron_fraction": statistics["unique_neuron_fraction"],
                }
            )
            layer.mlp = original_mlp
            del replacement, logits
            gc.collect()
        curves.append(layer_curve)
        raw_layers.append(
            {
                "layer": layer_index,
                "intermediate_neurons": intermediate,
                "points": raw_points,
                "damage_at_minimum": layer_curve[0].damage,
                "damage_at_maximum": layer_curve[-1].damage,
                "damage_reduction": layer_curve[0].damage - layer_curve[-1].damage,
            }
        )

    diagnostics: dict[str, Any] = {
        "baseline_cross_entropy": baseline_ce,
        "baseline_teacher": baseline_teacher,
        "count_options": count_options,
        "layers": raw_layers,
    }
    return curves, diagnostics


def evaluate_allocation(
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
        "used_neurons": sum(layer_counts),
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
    if min(args.calibration_tokens, args.eval_tokens) <= 0:
        raise SystemExit("token counts must be positive")

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
    calibration_tokens = greedy_tokens(
        model=calibration_model,
        encoded=calibration_encoded,
        count=args.calibration_tokens,
    ).to("cpu")
    curves, curve_diagnostics = measure_layer_damage_curves(
        model=calibration_model,
        encoded=calibration_encoded,
        exact_tokens=calibration_tokens,
        count_options=args.count_options,
    )
    layers = len(curves)
    intermediate = int(
        calibration_model.model.layers[0].mlp.gate_proj.out_features
    )
    del calibration_model, calibration_tokens
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

    target, _ = default_specs()
    budget_results: list[dict[str, Any]] = []
    for requested_fraction in args.total_fractions:
        requested_total = layers * max(1, ceil(intermediate * requested_fraction))
        nonlinear_allocation = solve_nonlinear_allocation(
            curves,
            total_budget=requested_total,
        )
        uniform_counts = uniform_neuron_allocation(
            layers=layers,
            intermediate_neurons=intermediate,
            total_neurons=nonlinear_allocation.used_neurons,
        )
        uniform = evaluate_allocation(
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_name=args.model,
            dtype=dtype,
            device=device,
            encoded=eval_encoded,
            exact_tokens=exact_eval_tokens,
            layer_counts=uniform_counts,
        )
        nonlinear = evaluate_allocation(
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_name=args.model,
            dtype=dtype,
            device=device,
            encoded=eval_encoded,
            exact_tokens=exact_eval_tokens,
            layer_counts=nonlinear_allocation.layer_counts,
        )
        actual_fraction = nonlinear_allocation.used_neurons / (layers * intermediate)
        budget = mlp_heavy_hitter_budget(
            target=target,
            selected_fraction=actual_fraction,
            source_bits=16,
            selector_bits_per_neuron=32,
            partial_traffic_limit_gib=1.6,
        )
        qualifies = bool(
            budget.partial_traffic_pass
            and nonlinear["teacher_forced"]["top32_rate"] >= 0.95
            and nonlinear["autonomous_exact_prefix"] >= 4
        )
        budget_results.append(
            {
                "requested_fraction": requested_fraction,
                "requested_total_neurons": requested_total,
                "actual_fraction": actual_fraction,
                "allocation": nonlinear_allocation.to_dict(),
                "uniform": uniform,
                "nonlinear": nonlinear,
                "comparison": {
                    "top1_delta": (
                        nonlinear["teacher_forced"]["top1_rate"]
                        - uniform["teacher_forced"]["top1_rate"]
                    ),
                    "top32_delta": (
                        nonlinear["teacher_forced"]["top32_rate"]
                        - uniform["teacher_forced"]["top32_rate"]
                    ),
                    "autonomous_prefix_delta": (
                        nonlinear["autonomous_exact_prefix"]
                        - uniform["autonomous_exact_prefix"]
                    ),
                },
                "projected_405b_partial_mlp_budget": budget.to_dict(),
                "qualifies": qualifies,
            }
        )

    promoted = [point for point in budget_results if point["qualifies"]]
    improves_any = any(
        point["comparison"]["top32_delta"] > 0
        or point["comparison"]["autonomous_prefix_delta"] > 0
        for point in budget_results
    )
    payload = {
        "evidence_level": "E2 disjoint-prompt nonlinear heavy-hitter allocation",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "calibration_prompt_tokens": int(calibration_encoded["input_ids"].shape[1]),
        "eval_prompt_tokens": int(eval_encoded["input_ids"].shape[1]),
        "calibration_tokens": args.calibration_tokens,
        "eval_tokens": args.eval_tokens,
        "count_options": args.count_options,
        "total_fractions": args.total_fractions,
        "damage_curves": curve_diagnostics,
        "budgets": budget_results,
        "promoted_fractions": [point["requested_fraction"] for point in promoted],
        "improves_any": improves_any,
        "contract": (
            "Every layer-count damage point is measured by actually replacing one "
            "MLP layer and observing final exact-token logits on the calibration "
            "prompt. A discrete allocator chooses counts under a fixed total. The "
            "chosen allocation is validated on a disjoint prompt against a uniform "
            "allocation with exactly the same used neuron count. Evaluation remains "
            "an optimistic exact-activation oracle."
        ),
        "qualifies": bool(promoted),
        "decision": (
            "advance nonlinear allocation to causal selection and proof"
            if promoted
            else (
                "retain nonlinear allocation signal but reject tested quality gate"
                if improves_any
                else "close exact-neuron heavy-hitter allocation family"
            )
        ),
        "next_candidate_if_rejected": (
            "return to decision-level programs; exact-neuron subsets are too large "
            "or too interaction-sensitive under the 405B traffic envelope"
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
