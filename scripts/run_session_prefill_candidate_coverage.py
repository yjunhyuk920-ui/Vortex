from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import (
    encode_prompt,
    require_transformers,
)
from scripts.run_hot_candidate_coverage import DEFAULT_WIDTHS, last_logits
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.decision_tile_repair import replace_with_decision_tile_modules
from vortex_runtime.rank_frontier import rank_budget_point
from vortex_runtime.session_prefill_basis import (
    compile_session_response_bases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a session-specific response basis using only the exact "
            "user-prompt prefill inputs and outputs, then measure exact-token "
            "candidate coverage on the unseen continuation."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-rank", type=int, default=72)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("session_prefill_candidate_coverage.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.max_rank <= 0:
        raise SystemExit("token count and max rank must be positive")
    if args.rank_rtol < 0:
        raise SystemExit("rank tolerance must be non-negative")

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

    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    for name, module in replacements.items():
        def pre_hook(
            _module: torch.nn.Module,
            hook_args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captured_inputs[key] = hook_args[0].detach().to("cpu")

        def output_hook(
            _module: torch.nn.Module,
            _hook_args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            captured_outputs[key] = output.detach().to("cpu")

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    prompt_tokens = int(prefix.shape[-1])
    for module in replacements.values():
        module.set_mode("exact")
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            model(
                **encoded,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    compile_stats = compile_session_response_bases(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        max_rank=args.max_rank,
        rank_rtol=args.rank_rtol,
    )
    compiled_ranks = [item.compiled_rank for item in compile_stats.values()]
    capsule_bytes = sum(item.capsule_bytes for item in compile_stats.values())

    rows: list[CandidateCoverageRow] = []
    for offset in range(args.tokens):
        for module in replacements.values():
            module.set_mode("exact")
        exact_logits = last_logits(model, prefix)
        exact_token = int(torch.argmax(exact_logits).item())

        for module in replacements.values():
            module.set_mode("project")
        hot_logits = last_logits(model, prefix)
        hot_token = int(torch.argmax(hot_logits).item())
        rows.append(
            CandidateCoverageRow(
                position=offset,
                exact_token=exact_token,
                hot_token=hot_token,
                exact_token_rank=token_rank(hot_logits, exact_token),
                hot_top1_margin=top1_margin(hot_logits),
                exact_logit_gap_from_hot_top1=float(
                    (hot_logits[hot_token] - hot_logits[exact_token]).item()
                ),
            )
        )
        prefix = torch.cat(
            (
                prefix,
                torch.tensor([[exact_token]], dtype=torch.long, device=device),
            ),
            dim=-1,
        )

    first_divergence = next((row for row in rows if not row.exact_match), None)
    ranks = [row.exact_token_rank for row in rows]
    coverage = coverage_at_k(rows, DEFAULT_WIDTHS)
    first_rank = (
        None if first_divergence is None else first_divergence.exact_token_rank
    )
    advance = (
        (first_rank is None or first_rank <= 32)
        and coverage["32"] >= 0.95
    )
    budget = rank_budget_point(args.max_rank)

    result = {
        "evidence_level": "E1 causal exact-prompt session-basis diagnostic",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "requested_max_rank": args.max_rank,
        "rank_rtol": args.rank_rtol,
        "prompt_tokens": prompt_tokens,
        "evaluated_continuation_tokens": len(rows),
        "compiler_contract": (
            "Only exact prompt-prefill module inputs and outputs are used to "
            "recover U and WU. No continuation token, continuation logit, "
            "continuation gradient, or second exact weight scan is used."
        ),
        "compiled_rank_statistics": {
            "minimum": min(compiled_ranks),
            "maximum": max(compiled_ranks),
            "mean": sum(compiled_ranks) / len(compiled_ranks),
        },
        "capsule_bytes": capsule_bytes,
        "prompt_reconstruction": {
            "maximum_input_relative_error": max(
                item.input_reconstruction_relative_error
                for item in compile_stats.values()
            ),
            "maximum_output_relative_error": max(
                item.output_reconstruction_relative_error
                for item in compile_stats.values()
            ),
        },
        "per_module_compile_stats": {
            name: item.to_dict() for name, item in compile_stats.items()
        },
        "budget_at_requested_rank": budget.to_dict(),
        "same_context_contract": (
            "Exact and compiled paths are evaluated on the same authoritative "
            "exact prefix at every unseen continuation position."
        ),
        "exact_top1_match_rate": sum(row.exact_match for row in rows) / len(rows),
        "coverage_at_k": coverage,
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
        "decision_rule": (
            "advance only when first-divergence exact-token rank is at most 32 "
            "and top-32 coverage is at least 0.95"
        ),
        "decision": (
            "advance session-prefill multi-hypothesis certificate"
            if advance
            else "reject prompt-only session response basis at this rank"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "compiled_rank_statistics": result[
                    "compiled_rank_statistics"
                ],
                "prompt_reconstruction": result["prompt_reconstruction"],
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
