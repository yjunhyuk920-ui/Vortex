from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.feasibility import default_specs
from vortex_runtime.mlp_heavy_hitter import (
    aggregate_heavy_hitter_stats,
    mlp_heavy_hitter_budget,
    replace_llama_mlp_with_oracle_heavy_hitters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure an optimistic exact-activation SwiGLU heavy-hitter oracle. "
            "Full gate/up activations select original neurons; only selected "
            "neurons contribute through down projection."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selected-fraction", type=float, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlp_heavy_hitter_oracle.json"),
    )
    return parser.parse_args()


def teacher_forced_logits(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> torch.Tensor:
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    continuation_input = exact_tokens[:, :-1].to(prompt_ids.device)
    combined_ids = torch.cat((prompt_ids, continuation_input), dim=1)
    kwargs: dict[str, object] = {
        "input_ids": combined_ids,
        "use_cache": False,
        "return_dict": True,
    }
    if prompt_mask is not None:
        combined_mask = torch.cat(
            (
                prompt_mask,
                torch.ones_like(
                    continuation_input,
                    dtype=prompt_mask.dtype,
                    device=prompt_mask.device,
                ),
            ),
            dim=1,
        )
        kwargs["attention_mask"] = combined_mask
    with torch.inference_mode():
        logits = model(**kwargs).logits
    start = prompt_ids.shape[1] - 1
    end = start + exact_tokens.shape[1]
    selected = logits[:, start:end, :]
    if selected.shape[1] != exact_tokens.shape[1]:
        raise RuntimeError("failed to align teacher-forced logits")
    return selected.detach().to("cpu", torch.float32)


def token_rank(logits: torch.Tensor, token: int) -> int:
    value = logits[token]
    return int(torch.count_nonzero(logits > value).item()) + 1


def teacher_summary(
    *,
    logits: torch.Tensor,
    exact_tokens: torch.Tensor,
) -> dict[str, int | float | list[int]]:
    ranks: list[int] = []
    for position in range(exact_tokens.shape[1]):
        ranks.append(
            token_rank(
                logits[0, position],
                int(exact_tokens[0, position].item()),
            )
        )
    return {
        "tokens": len(ranks),
        "top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "top4_rate": sum(rank <= 4 for rank in ranks) / len(ranks),
        "top32_rate": sum(rank <= 32 for rank in ranks) / len(ranks),
        "mean_exact_token_rank": sum(ranks) / len(ranks),
        "maximum_exact_token_rank": max(ranks),
        "ranks": ranks,
    }


def common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    limit = min(left_flat.numel(), right_flat.numel())
    for index in range(limit):
        if int(left_flat[index].item()) != int(right_flat[index].item()):
            return index
    return limit


def main() -> None:
    args = parse_args()
    if not 0 < args.selected_fraction <= 1:
        raise SystemExit("selected fraction must be in (0, 1]")
    if args.tokens <= 0:
        raise SystemExit("tokens must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)

    started = time.perf_counter()
    exact_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    exact_model.to(device)
    exact_model.eval()
    exact_tokens = greedy_tokens(
        model=exact_model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    del exact_model
    gc.collect()

    oracle_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    oracle_model.to(device)
    oracle_model.eval()
    modules = replace_llama_mlp_with_oracle_heavy_hitters(
        oracle_model,
        selected_fraction=args.selected_fraction,
    )
    for module in modules:
        module.reset_statistics()

    logits = teacher_forced_logits(
        model=oracle_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    teacher = teacher_summary(logits=logits, exact_tokens=exact_tokens)
    autonomous = greedy_tokens(
        model=oracle_model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    prefix = common_prefix_length(autonomous, exact_tokens)
    aggregate = aggregate_heavy_hitter_stats(modules)

    target, _ = default_specs()
    budget = mlp_heavy_hitter_budget(
        target=target,
        selected_fraction=args.selected_fraction,
        source_bits=16,
        selector_bits_per_neuron=32,
        partial_traffic_limit_gib=1.6,
    )
    qualifies = bool(
        budget.partial_traffic_pass
        and teacher["top32_rate"] >= 0.95
        and prefix >= 4
    )
    payload = {
        "evidence_level": "E2 exact-activation MLP heavy-hitter oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "tokens": args.tokens,
        "requested_selected_fraction": args.selected_fraction,
        "oracle_statistics": aggregate.to_dict(),
        "teacher_forced": teacher,
        "autonomous_exact_prefix": prefix,
        "autonomous_exact_match_rate": float(
            torch.eq(autonomous, exact_tokens).float().mean().item()
        ),
        "projected_405b_partial_mlp_budget": budget.to_dict(),
        "contract": (
            "This is an optimistic oracle: full exact gate/up activations and down "
            "column norms choose the best original neuron subset for each vector. "
            "The measured quality is an upper bound for any compact causal selector. "
            "Promotion still requires a selector that avoids reading unselected rows "
            "and a sound omitted-tail proof."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance certified causal heavy-hitter selection"
            if qualifies
            else "reject tested exact-neuron fraction or increase it under budget"
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
