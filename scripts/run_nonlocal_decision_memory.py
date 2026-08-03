from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.final_hidden_trace import (
    collect_prompt_continuation_trace,
    continuation_queries_after_anchor,
)
from vortex_runtime.nonlocal_decision_memory import (
    build_nonlocal_decision_memory,
    evaluate_nonlocal_decision_memory,
)


def parse_ranks(value: str) -> list[int]:
    ranks: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rank = int(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError("ranks must be integers") from error
        if rank <= 0:
            raise argparse.ArgumentTypeError("ranks must be positive")
        if rank not in ranks:
            ranks.append(rank)
    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank is required")
    return ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-only exact decision-block memory and test nearest, "
            "top-k, and global-oracle reuse on a held-out continuation."
        )
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("experiments/nonlocal_decision_memory_prompts.json"),
    )
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=256)
    parser.add_argument(
        "--ranks",
        type=parse_ranks,
        default=[8, 16, 32, 64, 128, 256],
    )
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
    device: torch.device,
) -> torch.nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    model.to(device)
    model.eval()
    return model


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


def eos_position(tokens: torch.Tensor, eos_token_id: int | None) -> int | None:
    if eos_token_id is None:
        return None
    matches = torch.nonzero(tokens == eos_token_id, as_tuple=False)
    if matches.numel() == 0:
        return None
    return int(matches[0, 0].item())


def main() -> None:
    args = parse_args()
    if args.steps < 247:
        raise ValueError("steps must be at least the 247-token promotion horizon")
    if args.block_length < 247:
        raise ValueError("block length must be at least 247")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    prompt = load_prompt(args.prompts, args.prompt_id)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(
        AutoModelForCausalLM,
        model_name=args.model,
        device=device,
    )

    trace = collect_prompt_continuation_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        continuation_steps=args.steps,
        device=device,
    )
    query_hidden, target_tokens = continuation_queries_after_anchor(
        trace,
        steps=args.steps,
    )

    continuation_eos = eos_position(target_tokens, tokenizer.eos_token_id)
    maximum_run = maximum_identical_run(target_tokens)
    unique_fraction = len(set(target_tokens.tolist())) / max(target_tokens.numel(), 1)
    repetition_4gram = ngram_repetition_rate(target_tokens, n=4)
    degeneration_pass = bool(
        (continuation_eos is None or continuation_eos >= 247)
        and maximum_run <= 16
        and unique_fraction >= 0.05
    )

    frontiers: list[dict[str, Any]] = []
    for rank in args.ranks:
        print(
            json.dumps(
                {
                    "event": "evaluate_rank",
                    "prompt_id": args.prompt_id,
                    "rank": rank,
                }
            ),
            flush=True,
        )
        memory = build_nonlocal_decision_memory(
            prompt_hidden_states=trace.prompt_hidden_states,
            prompt_token_ids=trace.prompt_token_ids,
            key_rank=rank,
            block_length=args.block_length,
        )
        frontier = evaluate_nonlocal_decision_memory(
            memory,
            query_hidden_states=query_hidden,
            continuation_token_ids=target_tokens,
            topk_values=(4, 16, 64),
            scaled_entries=65536,
            target_hidden_size=16384,
        )
        item = frontier.to_dict()
        item["nearest_qualifies"] = bool(
            degeneration_pass
            and frontier.nearest.maximum >= 247
            and frontier.budget_scaled_entries.memory_pass
            and frontier.budget_scaled_entries.lookup_pass
        )
        frontiers.append(item)

    global_maxima = [item["global_oracle"]["maximum"] for item in frontiers]
    if len(set(global_maxima)) != 1:
        raise RuntimeError("rank-independent global oracle changed across ranks")
    global_maximum = int(global_maxima[0])
    representation_capacity_pass = bool(degeneration_pass and global_maximum >= 247)
    promoted_ranks = [
        item["requested_rank"] for item in frontiers if item["nearest_qualifies"]
    ]
    best = max(
        frontiers,
        key=lambda item: (
            item["nearest"]["maximum"],
            item["topk_oracles"]["64"]["maximum"],
            -item["requested_rank"],
        ),
    )

    payload = {
        "experiment": "039_nonlocal_exact_decision_memory",
        "evidence_level": "E1/E2 optimistic nonlocal exact-block reuse gate",
        "model": args.model,
        "prompt_id": args.prompt_id,
        "prompt_tokens": trace.prompt_tokens,
        "memory_entries": trace.prompt_tokens - 1,
        "continuation_steps": args.steps,
        "block_length": args.block_length,
        "exact_boundary_anchor_token": trace.first_generated_token,
        "build_contract": {
            "prompt_tokens_only": True,
            "prompt_hidden_only": True,
            "continuation_tokens_in_build": False,
            "continuation_hidden_in_build": False,
            "final_prompt_position_excluded_from_memory": True,
            "first_continuation_token_charged_as_boundary_anchor": True,
            "replayed_targets_begin_after_anchor": True,
        },
        "continuation_diagnostics": {
            "eos_position_after_anchor": continuation_eos,
            "maximum_identical_token_run": maximum_run,
            "unique_token_fraction": unique_fraction,
            "four_gram_repetition_rate": repetition_4gram,
            "degeneration_pass": degeneration_pass,
        },
        "frontiers": frontiers,
        "global_oracle_maximum_exact_prefix": global_maximum,
        "representation_capacity_pass": representation_capacity_pass,
        "promoted_nearest_ranks": promoted_ranks,
        "best_nearest_configuration": {
            "requested_rank": best["requested_rank"],
            "effective_rank": best["effective_rank"],
            "nearest": best["nearest"],
            "top64_oracle": best["topk_oracles"]["64"],
            "global_oracle": best["global_oracle"],
            "scaled_budget": best["budget_scaled_entries"],
        },
        "qualifies": bool(representation_capacity_pass and promoted_ranks),
        "decision": (
            "advance nonlocal decision memory to exact block validation"
            if representation_capacity_pass and promoted_ranks
            else (
                "global token oracle exposes capacity but hidden retrieval fails"
                if representation_capacity_pass
                else "reject prompt-only nonlocal exact token-block memory"
            )
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
