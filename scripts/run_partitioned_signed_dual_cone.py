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

from vortex_runtime.decision_trace import collect_one_step_mlp_decision_trace
from vortex_runtime.feasibility import default_specs
from vortex_runtime.partitioned_signed_dual import (
    build_partitioned_signed_dual_terms,
    compile_partitioned_signed_dual_kernel,
    partitioned_cone_metadata_budget,
)
from vortex_runtime.signed_dual_mlp import (
    refine_signed_dual_certificate,
    signed_dual_refinement_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--metadata-bits", type=int, default=8)
    parser.add_argument("--metadata-limit-gib", type=float, default=2.5)
    parser.add_argument("--margin-share", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit("prompts must be a non-empty JSON list")
    if args.bits < 2 or args.block_size <= 0 or not 0 < args.margin_share <= 1:
        raise SystemExit("invalid precision, block size, or margin share")

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
    traces = [
        collect_one_step_mlp_decision_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )
        for prompt in prompts
    ]
    states = [
        {
            "trace": trace,
            "target": abs(trace.exact_margin) * args.margin_share / len(layers),
            "layers": [],
            "total": 0,
            "error_refined": 0,
            "sign_refined": 0,
            "unsafe": 0,
            "containment_failures": 0,
        }
        for trace in traces
    ]

    for layer_index, layer in enumerate(layers):
        mlp = layer.mlp
        kernel = compile_partitioned_signed_dual_kernel(
            gate_weight=mlp.gate_proj.weight,
            up_weight=mlp.up_proj.weight,
            down_weight=mlp.down_proj.weight,
            bits=args.bits,
            block_size=args.block_size,
        )
        for state in states:
            trace = state["trace"]
            terms, diagnostics = build_partitioned_signed_dual_terms(
                kernel,
                activation=trace.activations[layer_index],
                output_dual=trace.output_duals[layer_index],
            )
            sign_cert = refine_signed_dual_certificate(terms, require_sign=True)
            error_cert = refine_signed_dual_certificate(
                terms,
                target_absolute_error=state["target"],
                require_sign=False,
            )
            state["total"] += error_cert.total_neurons
            state["error_refined"] += error_cert.refined_neurons
            state["sign_refined"] += sign_cert.refined_neurons
            state["unsafe"] += int(sign_cert.unsafe_certificate)
            state["unsafe"] += int(error_cert.unsafe_certificate)
            state["containment_failures"] += int(not sign_cert.interval_contains_exact)
            state["containment_failures"] += int(not error_cert.interval_contains_exact)
            state["layers"].append(
                {
                    "layer": layer_index,
                    "sign": sign_cert.to_dict(),
                    "margin_share": error_cert.to_dict(),
                    "diagnostics": diagnostics.to_dict(),
                }
            )
        del kernel
        gc.collect()

    target_model, _ = default_specs()
    metadata = partitioned_cone_metadata_budget(
        target=target_model,
        block_size=args.block_size,
        metadata_bits=args.metadata_bits,
        metadata_limit_gib=args.metadata_limit_gib,
    )
    results = []
    for state in states:
        trace = state["trace"]
        fraction = state["error_refined"] / max(state["total"], 1)
        traffic = signed_dual_refinement_budget(
            target=target_model,
            selected_fraction=fraction,
            source_bits=16,
            partial_limit_gib=1.6,
        )
        def ratio(partitioned: str, global_name: str) -> float:
            return sum(
                item["diagnostics"][partitioned]
                / max(item["diagnostics"][global_name], 1e-24)
                for item in state["layers"]
            ) / len(state["layers"])

        qualifies = bool(
            state["unsafe"] == 0
            and state["containment_failures"] == 0
            and traffic.partial_traffic_pass
            and metadata.metadata_pass
        )
        results.append(
            {
                "prompt": trace.prompt,
                "prompt_tokens": trace.prompt_tokens,
                "winner_token": trace.winner_token,
                "competitor_token": trace.competitor_token,
                "exact_margin": trace.exact_margin,
                "per_layer_error_target": state["target"],
                "mean_error_refined_fraction": fraction,
                "mean_sign_refined_fraction": state["sign_refined"] / max(state["total"], 1),
                "gate_radius_ratio": ratio("partitioned_gate_radius_sum", "global_gate_radius_sum"),
                "up_radius_ratio": ratio("partitioned_up_radius_sum", "global_up_radius_sum"),
                "directional_radius_ratio": ratio(
                    "partitioned_directional_radius_sum",
                    "global_directional_radius_sum",
                ),
                "unsafe_certificates": state["unsafe"],
                "interval_failures": state["containment_failures"],
                "projected_405b_exact_refinement": traffic.to_dict(),
                "layers": state["layers"],
                "qualifies": qualifies,
            }
        )

    payload = {
        "evidence_level": "E1/E2 optimistic partitioned signed dual cone",
        "model": args.model,
        "bits": args.bits,
        "block_size": args.block_size,
        "metadata_budget": metadata.to_dict(),
        "prompts": results,
        "summary": {
            "mean_refined_fraction": sum(item["mean_error_refined_fraction"] for item in results) / len(results),
            "maximum_refined_fraction": max(item["mean_error_refined_fraction"] for item in results),
            "maximum_exact_gib_per_token": max(
                item["projected_405b_exact_refinement"]["exact_refinement_gib_per_token"]
                for item in results
            ),
            "mean_gate_radius_ratio": sum(item["gate_radius_ratio"] for item in results) / len(results),
            "mean_up_radius_ratio": sum(item["up_radius_ratio"] for item in results) / len(results),
            "mean_directional_radius_ratio": sum(item["directional_radius_ratio"] for item in results) / len(results),
            "unsafe_certificates": sum(item["unsafe_certificates"] for item in results),
            "interval_failures": sum(item["interval_failures"] for item in results),
            "qualifies": all(item["qualifies"] for item in results),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["qualifies"] = payload["summary"]["qualifies"]
    payload["decision"] = (
        "advance partitioned cone to runtime dual transport"
        if payload["qualifies"]
        else "reject tested block granularity or refine residual metadata"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
