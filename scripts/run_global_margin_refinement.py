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

from vortex_runtime.block_signed_residual_code import (
    build_block_signed_residual_terms,
    compile_block_signed_residual_kernel,
    signed_residual_code_budget,
)
from vortex_runtime.decision_trace import collect_one_step_mlp_decision_trace
from vortex_runtime.feasibility import default_specs
from vortex_runtime.global_margin_refinement import (
    compare_equal_layer_and_global_refinement,
)
from vortex_runtime.signed_dual_mlp import signed_dual_refinement_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--build-prompts", type=Path, required=True)
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--margin-share", type=float, default=0.5)
    parser.add_argument("--price-steps", type=int, default=41)
    parser.add_argument("--metadata-limit-gib", type=float, default=6.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"{path} must contain a non-empty list")
    return prompts


def require_transformers() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    args = parse_args()
    build_prompts = load_prompts(args.build_prompts)
    eval_prompts = load_prompts(args.eval_prompts)
    if set(build_prompts) & set(eval_prompts):
        raise SystemExit("build and evaluation prompts must be disjoint")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    layers = model.model.layers

    started = time.perf_counter()
    build_traces = [
        collect_one_step_mlp_decision_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )
        for prompt in build_prompts
    ]
    eval_traces = [
        collect_one_step_mlp_decision_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )
        for prompt in eval_prompts
    ]
    terms_by_prompt: list[list] = [[] for _ in eval_traces]
    diagnostics_by_prompt: list[list[dict]] = [[] for _ in eval_traces]

    for layer_index, layer in enumerate(layers):
        mlp = layer.mlp
        kernel = compile_block_signed_residual_kernel(
            gate_weight=mlp.gate_proj.weight,
            up_weight=mlp.up_proj.weight,
            down_weight=mlp.down_proj.weight,
            activation_build_vectors=[trace.activations[layer_index] for trace in build_traces],
            dual_build_vectors=[trace.output_duals[layer_index] for trace in build_traces],
            bits=args.bits,
            block_size=args.block_size,
            rank=args.rank,
        )
        for prompt_index, trace in enumerate(eval_traces):
            terms, diagnostics = build_block_signed_residual_terms(
                kernel,
                activation=trace.activations[layer_index],
                output_dual=trace.output_duals[layer_index],
            )
            terms_by_prompt[prompt_index].append(terms)
            diagnostics_by_prompt[prompt_index].append(diagnostics.to_dict())
        del kernel
        gc.collect()

    target_model, _ = default_specs()
    metadata = signed_residual_code_budget(
        target=target_model,
        block_size=args.block_size,
        rank=args.rank,
        coefficient_bits=32,
        remainder_bits=32,
        basis_bits=32,
        metadata_limit_gib=args.metadata_limit_gib,
    )
    results = []
    for trace, layer_terms, diagnostics in zip(
        eval_traces,
        terms_by_prompt,
        diagnostics_by_prompt,
    ):
        target_error = abs(trace.exact_margin) * args.margin_share
        comparison = compare_equal_layer_and_global_refinement(
            layer_terms,
            total_absolute_error=target_error,
            price_steps=args.price_steps,
        )
        traffic = signed_dual_refinement_budget(
            target=target_model,
            selected_fraction=comparison.dual_price_refined_fraction,
            source_bits=16,
            partial_limit_gib=1.6,
        )
        qualifies = bool(
            comparison.dual_price_certificate.target_error_met
            and comparison.dual_price_certificate.interval_contains_exact
            and not comparison.dual_price_certificate.unsafe_certificate
            and traffic.partial_traffic_pass
            and metadata.metadata_pass
        )
        results.append(
            {
                "prompt": trace.prompt,
                "exact_margin": trace.exact_margin,
                "target_error": target_error,
                "comparison": comparison.to_dict(),
                "mean_activation_perpendicular_ratio": sum(x["activation_perpendicular_ratio"] for x in diagnostics) / len(diagnostics),
                "mean_dual_perpendicular_ratio": sum(x["dual_perpendicular_ratio"] for x in diagnostics) / len(diagnostics),
                "projected_405b_exact_refinement": traffic.to_dict(),
                "qualifies": qualifies,
            }
        )

    payload = {
        "evidence_level": "E2 fixed-dual global margin refinement",
        "model": args.model,
        "bits": args.bits,
        "block_size": args.block_size,
        "rank": args.rank,
        "price_steps": args.price_steps,
        "metadata_budget": metadata.to_dict(),
        "results": results,
        "summary": {
            "mean_equal_layer_fraction": sum(x["comparison"]["equal_layer_refined_fraction"] for x in results) / len(results),
            "mean_width_global_fraction": sum(x["comparison"]["width_global_refined_fraction"] for x in results) / len(results),
            "mean_dual_price_fraction": sum(x["comparison"]["dual_price_refined_fraction"] for x in results) / len(results),
            "maximum_dual_price_fraction": max(x["comparison"]["dual_price_refined_fraction"] for x in results),
            "maximum_exact_gib_per_token": max(x["projected_405b_exact_refinement"]["exact_refinement_gib_per_token"] for x in results),
            "qualifies": all(x["qualifies"] for x in results),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["qualifies"] = payload["summary"]["qualifies"]
    payload["decision"] = (
        "advance global margin program to causal dual transport"
        if payload["qualifies"]
        else "reject global allocation as insufficient for signed residual codes"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
