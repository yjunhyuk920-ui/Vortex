from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.feasibility import default_specs
from vortex_runtime.weight_stationary_block import (
    StreamedBlockHardware,
    certified_fixed_prefix,
    jacobi_token_update,
    longest_common_prefix,
    streamed_exact_block_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure an exact, training-free Jacobi fixed-point block decoder. "
            "Every target pass evaluates all draft positions in parallel; only "
            "a prefix unchanged by the causal update is certified and committed."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument(
        "--initializer",
        choices=("repeat-next", "repeat-last", "prompt-tail", "eos"),
        default="repeat-next",
    )
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument("--target-tensor-tflops", type=float, default=80.0)
    parser.add_argument("--baseline-memory-gib-s", type=float, default=300.0)
    parser.add_argument("--baseline-tensor-tflops", type=float, default=40.0)
    parser.add_argument("--target-ratio", type=float, default=1.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weight_stationary_fixed_point.json"),
    )
    return parser.parse_args()


def exact_greedy_tokens(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        output = model(**encoded, use_cache=True, return_dict=True)
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
        generated = [next_token]
        past = output.past_key_values
        current = next_token.reshape(-1, 1)
        for _ in range(1, count):
            output = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            current = torch.argmax(output.logits[:, -1, :], dim=-1).reshape(-1, 1)
            generated.append(current.reshape(-1))
    return torch.stack(generated, dim=1), generated[0]


def initial_draft(
    *,
    initializer: str,
    input_ids: torch.Tensor,
    prompt_next: torch.Tensor,
    eos_token_id: int,
    positions: int,
) -> torch.Tensor:
    batch = input_ids.shape[0]
    if initializer == "repeat-next":
        return prompt_next.reshape(batch, 1).expand(batch, positions).clone()
    if initializer == "repeat-last":
        return input_ids[:, -1:].expand(batch, positions).clone()
    if initializer == "eos":
        return torch.full(
            (batch, positions),
            eos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    if initializer == "prompt-tail":
        tail_count = min(positions, input_ids.shape[1])
        tail = input_ids[:, -tail_count:]
        repeats = (positions + tail_count - 1) // tail_count
        return tail.repeat(1, repeats)[:, :positions].clone()
    raise ValueError(f"unsupported initializer: {initializer}")


def jacobi_logits(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    draft: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    combined_ids = torch.cat((input_ids, draft), dim=1)
    combined_mask = None
    if attention_mask is not None:
        combined_mask = torch.cat(
            (
                attention_mask,
                torch.ones_like(draft, dtype=attention_mask.dtype),
            ),
            dim=1,
        )
    kwargs: dict[str, Any] = {
        "input_ids": combined_ids,
        "use_cache": False,
        "return_dict": True,
    }
    if combined_mask is not None:
        kwargs["attention_mask"] = combined_mask
    with torch.inference_mode():
        output = model(**kwargs)
    prompt_length = input_ids.shape[1]
    prompt_next_logits = output.logits[:, prompt_length - 1, :]
    draft_logits = output.logits[:, prompt_length : prompt_length + draft.shape[1], :]
    return prompt_next_logits, draft_logits


def longest_reference_match_from_any_position(
    *,
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[int, int]:
    """Return the longest reference-prefix n-gram found inside a trajectory row."""

    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    best_length = 0
    best_start = 0
    for start in range(candidate_flat.numel()):
        limit = min(reference_flat.numel(), candidate_flat.numel() - start)
        if limit <= best_length:
            continue
        unequal = torch.nonzero(
            candidate_flat[start : start + limit] != reference_flat[:limit],
            as_tuple=False,
        )
        length = limit if unequal.numel() == 0 else int(unequal[0, 0].item())
        if length > best_length:
            best_length = length
            best_start = start
    return best_length, best_start


def main() -> None:
    args = parse_args()
    if args.block_tokens <= 0 or args.max_iterations <= 0:
        raise SystemExit("block tokens and max iterations must be positive")

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
    reference, prompt_next = exact_greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.block_tokens,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer must define eos_token_id or pad_token_id")

    draft = initial_draft(
        initializer=args.initializer,
        input_ids=encoded["input_ids"],
        prompt_next=prompt_next,
        eos_token_id=int(eos_token_id),
        positions=args.block_tokens,
    )

    iterations: list[dict[str, int | float | bool]] = []
    best_prefix = 0
    best_iteration = 0
    best_rate = 0.0
    converged = False
    best_trajectory_match = {"length": 0, "iteration": 0, "start": 0}
    for iteration in range(1, args.max_iterations + 1):
        prompt_logits, draft_logits = jacobi_logits(
            model=model,
            encoded=encoded,
            draft=draft,
        )
        updated = jacobi_token_update(
            draft_tokens=draft,
            prompt_next_token=torch.argmax(prompt_logits, dim=-1),
            draft_logits=draft_logits,
        )
        certified = certified_fixed_prefix(draft, updated)
        exact_prefix = longest_common_prefix(updated, reference)
        exact_matches = int(torch.count_nonzero(updated == reference).item())
        trajectory_length, trajectory_start = longest_reference_match_from_any_position(
            candidate=updated,
            reference=reference,
        )
        if trajectory_length > int(best_trajectory_match["length"]):
            best_trajectory_match = {
                "length": trajectory_length,
                "iteration": iteration,
                "start": trajectory_start,
            }
        rate = certified / iteration
        if certified > 0 and rate > best_rate:
            best_prefix = certified
            best_iteration = iteration
            best_rate = rate
        iterations.append(
            {
                "iteration": iteration,
                "certified_fixed_prefix": certified,
                "diagnostic_exact_prefix": exact_prefix,
                "diagnostic_exact_matches": exact_matches,
                "diagnostic_longest_reference_ngram": trajectory_length,
                "diagnostic_reference_ngram_start": trajectory_start,
                "certified_tokens_per_target_pass": rate,
                "full_window_stable": certified == args.block_tokens,
                "full_window_matches_reference": bool(
                    torch.equal(updated, reference)
                ),
            }
        )
        draft = updated
        if certified == args.block_tokens:
            converged = True
            break

    last_measured_certified = int(iterations[-1]["certified_fixed_prefix"])
    if best_prefix == 0:
        best_prefix = last_measured_certified
        best_iteration = len(iterations)

    target, baseline = default_specs()
    hardware = StreamedBlockHardware(
        host_to_device_gib_s=args.host_to_device_gib_s,
        target_tensor_tflops=args.target_tensor_tflops,
        baseline_gpu_memory_gib_s=args.baseline_memory_gib_s,
        baseline_tensor_tflops=args.baseline_tensor_tflops,
    )
    best_budget = None
    if best_prefix > 0:
        best_budget = streamed_exact_block_budget(
            target=target,
            baseline=baseline,
            draft_positions=args.block_tokens,
            committed_tokens=best_prefix,
            target_passes=best_iteration,
            hardware=hardware,
            target_ratio=args.target_ratio,
        ).to_dict()
    full_budget = None
    if converged:
        full_budget = streamed_exact_block_budget(
            target=target,
            baseline=baseline,
            draft_positions=args.block_tokens,
            committed_tokens=args.block_tokens,
            target_passes=len(iterations),
            hardware=hardware,
            target_ratio=args.target_ratio,
        ).to_dict()

    certificate_matches_reference = (
        best_prefix == 0
        or bool(torch.equal(draft[:, :best_prefix], reference[:, :best_prefix]))
    )
    decision = "reject tested fixed-point block point"
    qualifies = False
    if best_budget is not None:
        qualifies = bool(
            certificate_matches_reference
            and best_budget["serialized_pass"]
            and best_prefix >= best_budget["minimum_committed_tokens_serialized"]
        )
        if qualifies:
            decision = "advance exact weight-stationary fixed-point block decoder"

    payload = {
        "evidence_level": "E1 exact weight-stationary fixed-point block frontier",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "block_tokens": args.block_tokens,
        "initializer": args.initializer,
        "max_iterations": args.max_iterations,
        "iterations_run": len(iterations),
        "converged_full_window": converged,
        "best_certified_prefix": best_prefix,
        "best_certification_iteration": best_iteration,
        "best_certified_tokens_per_target_pass": best_rate,
        "certificate_matches_exact_reference": certificate_matches_reference,
        "best_trajectory_reference_ngram": best_trajectory_match,
        "trajectory_candidate_tokens_examined": len(iterations) * args.block_tokens,
        "iterations": iterations,
        "hardware": {
            "host_to_device_gib_s": args.host_to_device_gib_s,
            "target_tensor_tflops": args.target_tensor_tflops,
            "baseline_memory_gib_s": args.baseline_memory_gib_s,
            "baseline_tensor_tflops": args.baseline_tensor_tflops,
            "target_ratio": args.target_ratio,
        },
        "best_prefix_roofline_budget": best_budget,
        "full_window_roofline_budget": full_budget,
        "contract": (
            "Target weights are never approximated. One target pass evaluates "
            "the whole draft window. A token prefix is committed only when it "
            "is unchanged by the deterministic causal Jacobi update."
        ),
        "decision_rule": (
            "advance only when the causal fixed-point certificate matches the "
            "exact greedy reference and the conservative serialized 405B "
            "roofline reaches the native-4B time envelope"
        ),
        "qualifies": qualifies,
        "decision": decision,
        "next_candidate_if_rejected": (
            "collect and verify multiple Jacobi trajectory n-grams in one "
            "weight stream, then add rejection recycling across adjacent blocks"
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
