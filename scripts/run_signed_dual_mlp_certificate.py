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

from vortex_runtime.feasibility import default_specs
from vortex_runtime.signed_dual_mlp import (
    build_signed_dual_terms,
    refine_signed_dual_certificate,
    signed_dual_refinement_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many exact original SwiGLU neurons are required to "
            "close sound signed local decision intervals under a fixed exact "
            "top-two output dual."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margin-share", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised in workflow
        raise RuntimeError("transformers is required for this experiment") from error
    return AutoModelForCausalLM, AutoTokenizer


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _register_mlp_captures(model: torch.nn.Module) -> tuple[list[Any], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[Any] = []
    for layer_index, layer in enumerate(layers):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            if not inputs or not isinstance(output, torch.Tensor):
                raise RuntimeError("unexpected Llama MLP hook signature")
            captures[index] = (inputs[0].detach(), output)

        handles.append(layer.mlp.register_forward_hook(hook))
    return handles, captures


def analyze_prompt(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    bits: int,
    margin_share: float,
    device: torch.device,
) -> dict[str, Any]:
    encoded = encode_prompt(tokenizer, prompt, device)
    with torch.no_grad():
        prefill = model(**encoded, use_cache=True)
        first_token = torch.argmax(prefill.logits[:, -1, :], dim=-1)
        past_key_values = prefill.past_key_values

    handles, captures = _register_mlp_captures(model)
    model.zero_grad(set_to_none=True)
    decode = model(
        input_ids=first_token[:, None],
        past_key_values=past_key_values,
        use_cache=False,
    )
    for handle in handles:
        handle.remove()

    logits = decode.logits[:, -1, :]
    top_values, top_indices = torch.topk(logits, k=2, dim=-1)
    winner = int(top_indices[0, 0].item())
    competitor = int(top_indices[0, 1].item())
    margin_tensor = top_values[0, 0] - top_values[0, 1]
    margin = float(margin_tensor.detach().item())

    root = getattr(model, "model")
    layers = getattr(root, "layers")
    if len(captures) != len(layers):
        raise RuntimeError("did not capture every decoder-layer MLP output")
    outputs = [captures[index][1] for index in range(len(layers))]
    gradients = torch.autograd.grad(
        margin_tensor,
        outputs,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    per_layer_target = abs(margin) * margin_share / max(len(layers), 1)
    layer_results: list[dict[str, Any]] = []
    total_error_refined = 0
    total_sign_refined = 0
    total_neurons = 0
    unsafe_certificates = 0
    interval_failures = 0

    for layer_index, (layer, gradient) in enumerate(zip(layers, gradients)):
        mlp = layer.mlp
        activation = captures[layer_index][0][0, -1].to("cpu")
        output_dual = gradient.detach()[0, -1].to("cpu")
        terms = build_signed_dual_terms(
            gate_weight=mlp.gate_proj.weight,
            up_weight=mlp.up_proj.weight,
            down_weight=mlp.down_proj.weight,
            activation=activation,
            output_dual=output_dual,
            bits=bits,
        )
        sign_certificate = refine_signed_dual_certificate(
            terms,
            require_sign=True,
        )
        error_certificate = refine_signed_dual_certificate(
            terms,
            target_absolute_error=per_layer_target,
            require_sign=False,
        )
        total_sign_refined += sign_certificate.refined_neurons
        total_error_refined += error_certificate.refined_neurons
        total_neurons += error_certificate.total_neurons
        unsafe_certificates += int(sign_certificate.unsafe_certificate)
        unsafe_certificates += int(error_certificate.unsafe_certificate)
        interval_failures += int(not sign_certificate.interval_contains_exact)
        interval_failures += int(not error_certificate.interval_contains_exact)
        layer_results.append(
            {
                "layer": layer_index,
                "dual_norm": float(torch.linalg.vector_norm(output_dual).item()),
                "activation_norm": float(torch.linalg.vector_norm(activation).item()),
                "sign_certificate": sign_certificate.to_dict(),
                "margin_share_certificate": error_certificate.to_dict(),
            }
        )
        del terms
        gc.collect()

    selected_fraction = total_error_refined / max(total_neurons, 1)
    target, _ = default_specs()
    traffic = signed_dual_refinement_budget(
        target=target,
        selected_fraction=selected_fraction,
        source_bits=16,
        partial_limit_gib=1.6,
    )
    prompt_qualifies = bool(
        unsafe_certificates == 0
        and interval_failures == 0
        and all(
            layer["margin_share_certificate"]["target_error_met"]
            for layer in layer_results
        )
        and traffic.partial_traffic_pass
    )

    payload = {
        "prompt": prompt,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "winner_token": winner,
        "competitor_token": competitor,
        "exact_margin": margin,
        "margin_share_fraction": margin_share,
        "per_layer_absolute_error_target": per_layer_target,
        "layers": layer_results,
        "total_neurons": total_neurons,
        "total_error_refined_neurons": total_error_refined,
        "total_sign_refined_neurons": total_sign_refined,
        "mean_error_refined_fraction": selected_fraction,
        "mean_sign_refined_fraction": total_sign_refined / max(total_neurons, 1),
        "unsafe_certificates": unsafe_certificates,
        "interval_failures": interval_failures,
        "projected_405b_exact_refinement": traffic.to_dict(),
        "qualifies": prompt_qualifies,
    }

    del decode, logits, top_values, top_indices, margin_tensor, outputs, gradients
    del prefill, past_key_values, captures
    gc.collect()
    return payload


def main() -> None:
    args = parse_args()
    if args.bits < 2:
        raise SystemExit("bits must be at least 2")
    if not 0 < args.margin_share <= 1:
        raise SystemExit("margin-share must lie in (0, 1]")

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(item, str) for item in prompts):
        raise SystemExit("prompts must be a non-empty JSON string list")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    started = time.perf_counter()
    prompt_results = [
        analyze_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            bits=args.bits,
            margin_share=args.margin_share,
            device=device,
        )
        for prompt in prompts
    ]
    unsafe = sum(point["unsafe_certificates"] for point in prompt_results)
    interval_failures = sum(point["interval_failures"] for point in prompt_results)
    mean_fraction = sum(
        point["mean_error_refined_fraction"] for point in prompt_results
    ) / len(prompt_results)
    max_fraction = max(
        point["mean_error_refined_fraction"] for point in prompt_results
    )
    all_qualify = all(point["qualifies"] for point in prompt_results)

    payload = {
        "evidence_level": "E1/E2 optimistic exact-dual local MLP certificate",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "margin_share": args.margin_share,
        "prompts": prompt_results,
        "summary": {
            "prompt_count": len(prompt_results),
            "mean_error_refined_fraction": mean_fraction,
            "maximum_error_refined_fraction": max_fraction,
            "unsafe_certificates": unsafe,
            "interval_failures": interval_failures,
            "qualifies": all_qualify,
        },
        "contract": (
            "For one real warm-decode step, exact top-two margin gradients define "
            "an optimistic fixed dual at each MLP output. Gate/up rows and down "
            "columns are independently low-bit approximated. Global SiLU and "
            "Cauchy bounds create a sound interval for each local signed neuron "
            "contribution. Widest intervals are exact-refined until each layer's "
            "allocated absolute margin error closes. This certifies each fixed "
            "local operator only; later nonlinear dual drift remains unproven."
        ),
        "qualifies": all_qualify,
        "decision": (
            "advance signed dual cone to multi-layer interval transport"
            if all_qualify
            else "reject tested signed local interval density or revise the hot approximation"
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
