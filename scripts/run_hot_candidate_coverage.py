from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import (
    DEFAULT_BUILD_PROMPTS,
    encode_prompt,
    generate,
    require_transformers,
)
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.decision_tile_repair import replace_with_decision_tile_modules


DEFAULT_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the exact next token remains inside the rank-32 "
            "hot path's top-K set while both models are evaluated on the same "
            "authoritative exact prefix."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hot_candidate_coverage.json"),
    )
    return parser.parse_args()


def last_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[0, -1].detach().to("cpu", torch.float32)


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("token counts must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    suffixes = tuple(args.suffixes or ("self_attn.o_proj", "mlp.down_proj"))
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    for module in replacements.values():
        module.set_mode("learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.build_new_tokens,
        )

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    prompt_tokens = int(prefix.shape[-1])
    rows: list[CandidateCoverageRow] = []
    started = time.perf_counter()

    for offset in range(args.tokens):
        for module in replacements.values():
            module.set_mode("exact")
        exact_logits = last_logits(model, prefix)
        exact_token = int(torch.argmax(exact_logits).item())

        for module in replacements.values():
            module.set_mode("project")
        hot_logits = last_logits(model, prefix)
        hot_token = int(torch.argmax(hot_logits).item())
        rank = token_rank(hot_logits, exact_token)
        rows.append(
            CandidateCoverageRow(
                position=offset,
                exact_token=exact_token,
                hot_token=hot_token,
                exact_token_rank=rank,
                hot_top1_margin=top1_margin(hot_logits),
                exact_logit_gap_from_hot_top1=float(
                    (hot_logits[hot_token] - hot_logits[exact_token]).item()
                ),
            )
        )
        next_token = torch.tensor([[exact_token]], dtype=torch.long, device=device)
        prefix = torch.cat((prefix, next_token), dim=-1)

    first_divergence = next((row for row in rows if not row.exact_match), None)
    ranks = [row.exact_token_rank for row in rows]
    result = {
        "evidence_level": "E1 exact-prefix candidate-coverage diagnostic",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "prompt_tokens": prompt_tokens,
        "evaluated_tokens": len(rows),
        "same_context_contract": (
            "Exact and hot paths are evaluated on the same authoritative exact "
            "prefix at every position. Exact targets are used only to measure "
            "candidate-set coverage."
        ),
        "exact_top1_match_rate": sum(row.exact_match for row in rows) / len(rows),
        "coverage_at_k": coverage_at_k(rows, DEFAULT_WIDTHS),
        "rank_statistics": {
            "minimum": min(ranks),
            "maximum": max(ranks),
            "mean": sum(ranks) / len(ranks),
        },
        "first_divergence": (
            None
            if first_divergence is None
            else {
                "position": first_divergence.position,
                "exact_token": first_divergence.exact_token,
                "hot_token": first_divergence.hot_token,
                "exact_token_rank": first_divergence.exact_token_rank,
                "hot_top1_margin": first_divergence.hot_top1_margin,
                "exact_logit_gap_from_hot_top1": (
                    first_divergence.exact_logit_gap_from_hot_top1
                ),
            }
        ),
        "rows": [row.__dict__ | {"exact_match": row.exact_match} for row in rows],
        "elapsed_seconds": time.perf_counter() - started,
        "decision_rule": {
            "advance_top_k_uncertainty": (
                "first divergence exact-token rank <= 32 and top-32 coverage "
                ">= 0.95"
            ),
            "reject_current_hot_representation": (
                "first divergence exact-token rank > 32 or top-32 coverage < 0.95"
            ),
        },
    }
    first_rank = None if first_divergence is None else first_divergence.exact_token_rank
    result["decision"] = (
        "advance multi-hypothesis top-K uncertainty experiment"
        if first_rank is not None
        and first_rank <= 32
        and result["coverage_at_k"]["32"] >= 0.95
        else "reject rank-32 hot representation for top-K uncertainty coverage"
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "exact_top1_match_rate": result["exact_top1_match_rate"],
                "coverage_at_k": result["coverage_at_k"],
                "first_divergence": result["first_divergence"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
