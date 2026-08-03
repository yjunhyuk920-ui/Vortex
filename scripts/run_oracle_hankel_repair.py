from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_prompt_hankel_decision_program import (
    load_model,
    load_prompt,
    parse_configs,
    require_transformers,
)
from vortex_runtime.final_hidden_trace import collect_prompt_continuation_trace
from vortex_runtime.hankel_decision_program import (
    fit_hankel_decision_program,
    hankel_program_budget,
    reduce_hidden,
)
from vortex_runtime.oracle_hankel_repair import (
    oracle_repair_hankel_rollout,
    repair_envelope_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the minimum exact-target repair rate using an impossible "
            "oracle that repairs only recurrence token mismatches."
        )
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("experiments/hankel_decision_prompts.json"),
    )
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=[
            (8, 8, 1, "linear"),
            (16, 8, 2, "linear"),
            (16, 16, 2, "full"),
            (32, 16, 2, "full"),
            (32, 16, 4, "full"),
            (64, 16, 2, "full"),
        ],
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.to(torch.float64), probability).item())


def evaluate_config(
    *,
    trace,
    embedding_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    state_rank: int,
    control_rank: int,
    order: int,
    lift: str,
    ridge: float,
) -> dict[str, Any]:
    program = fit_hankel_decision_program(
        hidden_states=trace.prompt_hidden_states,
        token_ids=trace.prompt_token_ids,
        embedding_weight=embedding_weight,
        lm_head_weight=lm_head_weight,
        lm_head_bias=lm_head_bias,
        state_rank=state_rank,
        control_rank=control_rank,
        order=order,
        lift=lift,
        ridge=ridge,
    )
    initial_history = [
        reduce_hidden(program, trace.prompt_hidden_states[-1 - lag])
        for lag in range(program.order)
    ]
    exact_controls = trace.continuation_token_ids[:-1]
    exact_targets = trace.continuation_token_ids[1:]
    rollout = oracle_repair_hankel_rollout(
        program,
        initial_history=initial_history,
        exact_control_tokens=exact_controls,
        exact_target_tokens=exact_targets,
        exact_hidden_states=trace.continuation_hidden_states,
    )
    hidden_difference = rollout.hidden_states - trace.continuation_hidden_states
    hidden_relative = torch.linalg.vector_norm(hidden_difference, dim=1) / torch.clamp(
        torch.linalg.vector_norm(trace.continuation_hidden_states, dim=1),
        min=1e-12,
    )
    predicted_norm = torch.linalg.vector_norm(rollout.hidden_states, dim=1)
    exact_norm = torch.linalg.vector_norm(trace.continuation_hidden_states, dim=1)
    cosine = (
        rollout.hidden_states * trace.continuation_hidden_states
    ).sum(dim=1) / torch.clamp(predicted_norm * exact_norm, min=1e-12)
    cosine_error = 1.0 - cosine

    budget = hankel_program_budget(
        state_rank=state_rank,
        control_rank=control_rank,
        order=order,
        lift=lift,
    )
    qualifies, amortized_compute = repair_envelope_passes(
        rollout.statistics,
        program_hot_compute_gflop_per_token=budget.hot_compute_gflop_per_token,
        program_build_compute_gflop=budget.build_compute_gflop,
        horizon_tokens=trace.continuation_steps,
    )
    return {
        "state_rank": state_rank,
        "control_rank": control_rank,
        "order": order,
        "lift": lift,
        "fit": program.diagnostics.to_dict(),
        "budget": budget.to_dict(),
        "repair": rollout.statistics.to_dict(),
        "amortized_total_compute_gflop_per_token": amortized_compute,
        "post_repair_hidden": {
            "relative_l2_mean": float(hidden_relative.mean().item()),
            "relative_l2_p95": _quantile(hidden_relative, 0.95),
            "cosine_error_mean": float(cosine_error.mean().item()),
            "cosine_error_p95": _quantile(cosine_error, 0.95),
        },
        "qualifies": qualifies,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 247:
        raise SystemExit("the repair Gate requires at least 247 tokens")
    if args.ridge < 0:
        raise SystemExit("ridge must be nonnegative")

    prompt = load_prompt(args.prompts, args.prompt_id)
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(
        AutoModelForCausalLM,
        model_name=args.model,
        dtype=dtype,
        device=device,
    )
    started = time.perf_counter()
    trace = collect_prompt_continuation_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        continuation_steps=args.steps,
        device=device,
    )
    embedding_weight = model.get_input_embeddings().weight.detach().to(
        "cpu", torch.float32
    )
    output_module = model.get_output_embeddings()
    if output_module is None or not hasattr(output_module, "weight"):
        raise RuntimeError("model has no output embedding/LM head weight")
    lm_head_weight = output_module.weight.detach().to("cpu", torch.float32)
    lm_head_bias = getattr(output_module, "bias", None)
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.detach().to("cpu", torch.float32)
    del model
    gc.collect()

    results: list[dict[str, Any]] = []
    for state_rank, control_rank, order, lift in args.configs:
        print(
            json.dumps(
                {
                    "event": "oracle_repair_config",
                    "prompt_id": args.prompt_id,
                    "state_rank": state_rank,
                    "control_rank": control_rank,
                    "order": order,
                    "lift": lift,
                }
            ),
            flush=True,
        )
        results.append(
            evaluate_config(
                trace=trace,
                embedding_weight=embedding_weight,
                lm_head_weight=lm_head_weight,
                lm_head_bias=lm_head_bias,
                state_rank=state_rank,
                control_rank=control_rank,
                order=order,
                lift=lift,
                ridge=args.ridge,
            )
        )
        gc.collect()

    promoted = [
        {
            "state_rank": item["state_rank"],
            "control_rank": item["control_rank"],
            "order": item["order"],
            "lift": item["lift"],
        }
        for item in results
        if item["qualifies"]
    ]
    best = min(
        results,
        key=lambda item: (
            item["repair"]["repairs"],
            -item["repair"]["mean_repair_interval"],
            item["amortized_total_compute_gflop_per_token"],
        ),
    )
    payload = {
        "evidence_level": "E2 optimistic oracle sparse Hankel repair",
        "model": args.model,
        "prompt_id": args.prompt_id,
        "prompt_tokens": trace.prompt_tokens,
        "continuation_steps": trace.continuation_steps,
        "device": str(device),
        "dtype": str(dtype),
        "configurations": results,
        "promoted_configurations": promoted,
        "best_configuration": {
            "state_rank": best["state_rank"],
            "control_rank": best["control_rank"],
            "order": best["order"],
            "lift": best["lift"],
            "repairs": best["repair"]["repairs"],
            "mean_repair_interval": best["repair"]["mean_repair_interval"],
            "repair_traffic_gib_per_token": best["repair"][
                "projected_repair_traffic_gib_per_token"
            ],
            "repair_compute_gflop_per_token": best["repair"][
                "projected_repair_compute_gflop_per_token"
            ],
        },
        "qualifies": bool(promoted),
        "decision": (
            "advance oracle-sparse repair to a sound causal detector"
            if promoted
            else "close Hankel recurrence plus sparse exact repair"
        ),
        "contract": (
            "The repair oracle sees the exact target token and therefore gives "
            "a lower bound on all deployable detector repair rates. Every mismatch "
            "or non-finite recurrence state is charged as one full 405B target interaction."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    safe_payload = _sanitize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            safe_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            safe_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
