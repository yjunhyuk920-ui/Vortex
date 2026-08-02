from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hot_candidate_coverage import DEFAULT_WIDTHS, last_logits
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
from vortex_runtime.capsule_quantization import (
    fake_quantize_response_capsules,
)
from vortex_runtime.decision_tile_repair import replace_with_decision_tile_modules
from vortex_runtime.hybrid_response_basis import (
    augment_response_bases_from_prompt_io,
)
from vortex_runtime.rank_frontier import rank_budget_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a generic global response capsule, augment it with directions "
            "derived only from exact user-prompt prefill residuals, quantize the "
            "combined capsule, and measure exact-token coverage on unseen "
            "continuation positions."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--global-rank", type=int, default=88)
    parser.add_argument("--total-rank", type=int, default=136)
    parser.add_argument("--capsule-bits", type=int, default=4)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hybrid_session_candidate_coverage.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("token counts must be positive")
    if args.global_rank <= 0 or args.total_rank < args.global_rank:
        raise SystemExit("ranks must be positive and total rank >= global rank")
    if not 2 <= args.capsule_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")
    if args.rank_rtol < 0:
        raise SystemExit("rank tolerance must be non-negative")

    budget = rank_budget_point(
        args.total_rank,
        capsule_bits=args.capsule_bits,
    )
    if not budget.pass_all:
        raise SystemExit(
            "hybrid total rank violates the fixed 405B envelope: "
            f"{budget.to_dict()}"
        )

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
        max_rank=args.total_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    for module in replacements.values():
        module.atlas.max_rank = args.global_rank
        module.set_mode("learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.build_new_tokens,
        )
    global_ranks = [module.atlas.rank for module in replacements.values()]
    for module in replacements.values():
        module.atlas.max_rank = args.total_rank

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

    augmentation = augment_response_bases_from_prompt_io(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        total_rank=args.total_rank,
        rank_rtol=args.rank_rtol,
    )
    final_ranks = [item.final_rank for item in augmentation.values()]
    added_ranks = [item.added_session_rank for item in augmentation.values()]
    quantization, per_module_quantization = fake_quantize_response_capsules(
        replacements,
        bits=args.capsule_bits,
    )

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
    exact_ranks = [row.exact_token_rank for row in rows]
    coverage = coverage_at_k(rows, DEFAULT_WIDTHS)
    first_rank = (
        None if first_divergence is None else first_divergence.exact_token_rank
    )
    advance = (
        budget.pass_all
        and (first_rank is None or first_rank <= 32)
        and coverage["32"] >= 0.95
    )

    result = {
        "evidence_level": "E1 causal quantized global-plus-session hybrid",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "global_rank_limit": args.global_rank,
        "total_rank_limit": args.total_rank,
        "capsule_bits": args.capsule_bits,
        "rank_rtol": args.rank_rtol,
        "prompt_tokens": prompt_tokens,
        "evaluated_continuation_tokens": len(rows),
        "compiler_contract": (
            "The global prior is built from fixed disjoint prompts. Session "
            "directions and their response images use only exact user-prompt "
            "module inputs and outputs. No continuation target, continuation "
            "gradient, or second exact-weight scan is used."
        ),
        "global_rank_statistics": {
            "minimum": min(global_ranks),
            "maximum": max(global_ranks),
            "mean": sum(global_ranks) / len(global_ranks),
        },
        "added_session_rank_statistics": {
            "minimum": min(added_ranks),
            "maximum": max(added_ranks),
            "mean": sum(added_ranks) / len(added_ranks),
        },
        "final_rank_statistics": {
            "minimum": min(final_ranks),
            "maximum": max(final_ranks),
            "mean": sum(final_ranks) / len(final_ranks),
        },
        "prompt_reconstruction": {
            "maximum_global_output_relative_error": max(
                item.global_output_reconstruction_relative_error
                for item in augmentation.values()
            ),
            "maximum_final_output_relative_error": max(
                item.final_output_reconstruction_relative_error
                for item in augmentation.values()
            ),
            "maximum_final_input_relative_error": max(
                item.final_input_reconstruction_relative_error
                for item in augmentation.values()
            ),
        },
        "per_module_augmentation": {
            name: item.to_dict() for name, item in augmentation.items()
        },
        "quantization": {
            "aggregate": quantization.to_dict(),
            "per_module": per_module_quantization,
        },
        "budget": budget.to_dict(),
        "same_context_contract": (
            "Exact and hybrid paths are evaluated on the same authoritative "
            "exact prefix at every unseen continuation position."
        ),
        "exact_top1_match_rate": sum(row.exact_match for row in rows) / len(rows),
        "coverage_at_k": coverage,
        "rank_statistics": {
            "minimum": min(exact_ranks),
            "maximum": max(exact_ranks),
            "mean": sum(exact_ranks) / len(exact_ranks),
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
            "advance only when the fixed 405B budget passes, first-divergence "
            "exact-token rank is at most 32, and top-32 coverage is at least 0.95"
        ),
        "decision": (
            "advance hybrid multi-hypothesis certificate"
            if advance
            else "reject tested global-plus-session hybrid allocation"
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
                "global_rank_statistics": result["global_rank_statistics"],
                "added_session_rank_statistics": result[
                    "added_session_rank_statistics"
                ],
                "final_rank_statistics": result["final_rank_statistics"],
                "prompt_reconstruction": result["prompt_reconstruction"],
                "quantization": result["quantization"]["aggregate"],
                "exact_top1_match_rate": result["exact_top1_match_rate"],
                "top32_coverage": result["coverage_at_k"]["32"],
                "rank_statistics": result["rank_statistics"],
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
