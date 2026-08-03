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

from vortex_runtime.final_hidden_trace import collect_prompt_continuation_trace
from vortex_runtime.hankel_decision_program import (
    fit_hankel_decision_program,
    hankel_program_budget,
    reduce_hidden,
    rollout_hankel_program,
)


def parse_configs(value: str) -> list[tuple[int, int, int, str]]:
    configs: list[tuple[int, int, int, str]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "configs must use state:control:order:lift"
            )
        try:
            state_rank = int(parts[0])
            control_rank = int(parts[1])
            order = int(parts[2])
        except ValueError as error:
            raise argparse.ArgumentTypeError("ranks and order must be integers") from error
        lift = parts[3]
        if min(state_rank, control_rank, order) <= 0:
            raise argparse.ArgumentTypeError("ranks and order must be positive")
        if lift not in {"linear", "quadratic", "bilinear", "full"}:
            raise argparse.ArgumentTypeError("invalid lift")
        config = (state_rank, control_rank, order, lift)
        if config not in configs:
            configs.append(config)
    if not configs:
        raise argparse.ArgumentTypeError("at least one config is required")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one exact prompt hidden trajectory into low-rank Hankel "
            "decision programs and test 256 future greedy decisions."
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


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install transformers and sentencepiece for the real-model gate"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def load_prompt(path: Path, prompt_id: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("prompt file must contain a prompts list")
    matches = [item for item in prompts if item.get("id") == prompt_id]
    if len(matches) != 1:
        raise ValueError(f"prompt id {prompt_id!r} was not found exactly once")
    prompt = matches[0].get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt text is empty")
    return prompt


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


def common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    count = min(left.numel(), right.numel())
    for index in range(count):
        if int(left[index].item()) != int(right[index].item()):
            return index
    return count


def maximum_identical_run(tokens: torch.Tensor) -> int:
    values = tokens.reshape(-1).tolist()
    if not values:
        return 0
    best = 1
    current = 1
    for previous, value in zip(values, values[1:]):
        if value == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def ngram_repetition_rate(tokens: torch.Tensor, n: int = 4) -> float:
    values = tokens.reshape(-1).tolist()
    if len(values) < n:
        return 0.0
    grams = [tuple(values[index : index + n]) for index in range(len(values) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def tensor_quantile(values: torch.Tensor, quantile: float) -> float:
    return float(torch.quantile(values.to(torch.float64), quantile).item())


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
    eos_token_id: int | None,
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
    exact_controls = trace.continuation_token_ids[:-1].tolist()
    exact_targets = trace.continuation_token_ids[1:]
    if len(exact_controls) != trace.continuation_steps:
        raise RuntimeError("continuation control accounting mismatch")

    autonomous = rollout_hankel_program(
        program,
        initial_history=initial_history,
        first_control_token=trace.first_generated_token,
        steps=trace.continuation_steps,
    )
    teacher = rollout_hankel_program(
        program,
        initial_history=initial_history,
        first_control_token=trace.first_generated_token,
        forced_control_tokens=exact_controls,
        steps=trace.continuation_steps,
    )

    exact_hidden = trace.continuation_hidden_states
    hidden_difference = teacher.hidden_states - exact_hidden
    hidden_relative = torch.linalg.vector_norm(hidden_difference, dim=1) / torch.clamp(
        torch.linalg.vector_norm(exact_hidden, dim=1), min=1e-12
    )
    teacher_norm = torch.linalg.vector_norm(teacher.hidden_states, dim=1)
    exact_norm = torch.linalg.vector_norm(exact_hidden, dim=1)
    cosine = (teacher.hidden_states * exact_hidden).sum(dim=1) / torch.clamp(
        teacher_norm * exact_norm,
        min=1e-12,
    )
    cosine_error = 1.0 - cosine

    teacher_top1 = torch.argmax(teacher.logits, dim=1)
    top32 = torch.topk(
        teacher.logits,
        k=min(32, teacher.logits.shape[1]),
        dim=1,
    ).indices
    top32_contains = torch.eq(top32, exact_targets[:, None]).any(dim=1)
    autonomous_match = torch.eq(autonomous.token_ids, exact_targets)
    prefix = common_prefix_length(autonomous.token_ids, exact_targets)

    exact_unique_fraction = len(set(exact_targets.tolist())) / max(
        int(exact_targets.numel()), 1
    )
    eos_position: int | None = None
    if eos_token_id is not None:
        matches = torch.nonzero(exact_targets == eos_token_id, as_tuple=False)
        if matches.numel():
            eos_position = int(matches[0, 0].item()) + 1
    exact_max_run = maximum_identical_run(exact_targets)
    exact_ngram_repetition = ngram_repetition_rate(exact_targets)

    budget = hankel_program_budget(
        state_rank=state_rank,
        control_rank=control_rank,
        order=order,
        lift=lift,
    )
    amortized_compute_at_247 = (
        budget.hot_compute_gflop_per_token
        + budget.build_compute_gflop / 247.0
    )
    qualifies = bool(
        prefix >= 247
        and float(torch.eq(teacher_top1, exact_targets).float().mean().item()) >= 0.99
        and bool(top32_contains.all().item())
        and tensor_quantile(cosine_error, 0.95) <= 0.05
        and budget.total_program_gib <= 0.25
        and amortized_compute_at_247 <= 9.6
        and (eos_position is None or eos_position > 247)
        and exact_max_run <= 8
        and exact_unique_fraction >= 0.10
    )
    return {
        "state_rank": state_rank,
        "control_rank": control_rank,
        "order": order,
        "lift": lift,
        "fit": program.diagnostics.to_dict(),
        "budget": budget.to_dict(),
        "amortized_compute_gflop_per_token_at_247": amortized_compute_at_247,
        "autonomous": {
            "exact_prefix": prefix,
            "exact_match_rate": float(autonomous_match.float().mean().item()),
            "first_predicted_tokens": autonomous.token_ids[:32].tolist(),
        },
        "teacher_forced": {
            "top1_rate": float(torch.eq(teacher_top1, exact_targets).float().mean().item()),
            "top32_rate": float(top32_contains.float().mean().item()),
            "hidden_relative_l2_mean": float(hidden_relative.mean().item()),
            "hidden_relative_l2_p95": tensor_quantile(hidden_relative, 0.95),
            "hidden_cosine_error_mean": float(cosine_error.mean().item()),
            "hidden_cosine_error_p95": tensor_quantile(cosine_error, 0.95),
        },
        "exact_continuation": {
            "unique_token_fraction": exact_unique_fraction,
            "maximum_identical_token_run": exact_max_run,
            "four_gram_repetition_rate": exact_ngram_repetition,
            "eos_position": eos_position,
            "first_target_tokens": exact_targets[:32].tolist(),
        },
        "qualifies": qualifies,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 247:
        raise SystemExit("the Gate requires at least 247 future decisions")
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
                    "event": "fit_config",
                    "prompt_id": args.prompt_id,
                    "state_rank": state_rank,
                    "control_rank": control_rank,
                    "order": order,
                    "lift": lift,
                }
            ),
            flush=True,
        )
        result = evaluate_config(
            trace=trace,
            embedding_weight=embedding_weight,
            lm_head_weight=lm_head_weight,
            lm_head_bias=lm_head_bias,
            state_rank=state_rank,
            control_rank=control_rank,
            order=order,
            lift=lift,
            ridge=args.ridge,
            eos_token_id=tokenizer.eos_token_id,
        )
        results.append(result)
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
    best = max(
        results,
        key=lambda item: (
            item["autonomous"]["exact_prefix"],
            item["teacher_forced"]["top1_rate"],
            -item["teacher_forced"]["hidden_cosine_error_mean"],
        ),
    )
    payload = {
        "evidence_level": "E1/E2 prompt-compiled Hankel decision program",
        "model": args.model,
        "prompt_id": args.prompt_id,
        "prompt_tokens": trace.prompt_tokens,
        "continuation_steps": trace.continuation_steps,
        "device": str(device),
        "dtype": str(dtype),
        "ridge": args.ridge,
        "configurations": results,
        "promoted_configurations": promoted,
        "best_configuration": {
            "state_rank": best["state_rank"],
            "control_rank": best["control_rank"],
            "order": best["order"],
            "lift": best["lift"],
            "autonomous_exact_prefix": best["autonomous"]["exact_prefix"],
            "teacher_top1": best["teacher_forced"]["top1_rate"],
            "teacher_top32": best["teacher_forced"]["top32_rate"],
            "hidden_cosine_error_mean": best["teacher_forced"][
                "hidden_cosine_error_mean"
            ],
        },
        "qualifies": bool(promoted),
        "decision": (
            "advance Hankel decision program to sound error certification"
            if promoted
            else "reject prompt-compiled low-rank Hankel decision programs"
        ),
        "contract": (
            "Only exact prompt token ids and final hidden states are used to fit "
            "the program. The exact first token from prefill is the sole rollout "
            "anchor. All later exact tokens and hidden states are evaluation-only."
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
