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
    fake_quantize_mixed_response_capsules,
)
from vortex_runtime.decision_tile_repair import replace_with_decision_tile_modules
from vortex_runtime.hybrid_response_basis import (
    augment_response_bases_from_prompt_io,
)
from vortex_runtime.rank_frontier import mixed_rank_budget_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a generic global response prior, augment it from exact user-"
            "prompt residuals, preserve session columns at higher precision, "
            "and measure unseen continuation candidate coverage."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--global-rank", type=int, default=58)
    parser.add_argument("--session-rank", type=int, default=45)
    parser.add_argument("--global-bits", type=int, default=4)
    parser.add_argument("--session-bits", type=int, default=8)
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
        default=Path("mixed_precision_hybrid_candidate_coverage.json"),
    )
    return parser.parse_args()


def relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference32 = reference.detach().to("cpu", torch.float32)
    estimate32 = estimate.detach().to("cpu", torch.float32)
    numerator = torch.linalg.vector_norm(reference32 - estimate32)
    denominator = torch.linalg.vector_norm(reference32)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("token counts must be positive")
    if args.global_rank <= 0 or args.session_rank <= 0:
        raise SystemExit("global and session ranks must be positive")
    if not 2 <= args.global_bits <= 16 or not 2 <= args.session_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")
    if args.rank_rtol < 0:
        raise SystemExit("rank tolerance must be non-negative")

    budget = mixed_rank_budget_point(
        global_rank=args.global_rank,
        session_rank=args.session_rank,
        global_bits=args.global_bits,
        session_bits=args.session_bits,
    )
    if not budget.pass_all:
        raise SystemExit(
            "mixed hybrid violates the fixed 405B envelope: "
            f"{budget.to_dict()}"
        )
    total_rank = args.global_rank + args.session_rank

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
        max_rank=total_rank,
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
    global_ranks = {
        name: module.atlas.rank for name, module in replacements.items()
    }
    for module in replacements.values():
        module.atlas.max_rank = total_rank

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
        total_rank=total_rank,
        rank_rtol=args.rank_rtol,
    )
    actual_global_ranks = {
        name: item.global_rank for name, item in augmentation.items()
    }
    final_ranks = [item.final_rank for item in augmentation.values()]
    added_ranks = [item.added_session_rank for item in augmentation.values()]

    quantization, per_module_quantization = (
        fake_quantize_mixed_response_capsules(
            replacements,
            global_ranks=actual_global_ranks,
            global_bits=args.global_bits,
            session_bits=args.session_bits,
        )
    )

    post_quantization_module_errors: dict[str, float] = {}
    for name, module in replacements.items():
        module.set_mode("project")
        input_tensor = captured_inputs[name].to(
            device=module.exact.weight.device,
            dtype=module.exact.weight.dtype,
        )
        with torch.inference_mode():
            estimate = module(input_tensor)
        post_quantization_module_errors[name] = relative_error(
            captured_outputs[name],
            estimate,
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
        "evidence_level": "E1 causal mixed-precision global-session hybrid",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "global_rank_limit": args.global_rank,
        "session_rank_limit": args.session_rank,
        "total_rank_limit": total_rank,
        "global_bits": args.global_bits,
        "session_bits": args.session_bits,
        "rank_rtol": args.rank_rtol,
        "prompt_tokens": prompt_tokens,
        "evaluated_continuation_tokens": len(rows),
        "compiler_contract": (
            "The generic prior uses only fixed disjoint build prompts. Session "
            "directions use only exact user-prompt module inputs and outputs. "
            "Global and session columns are quantized independently; no unseen "
            "continuation target, gradient, or exact-weight rescan is used."
        ),
        "global_rank_statistics": {
            "minimum": min(global_ranks.values()),
            "maximum": max(global_ranks.values()),
            "mean": sum(global_ranks.values()) / len(global_ranks),
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
        "prompt_reconstruction_before_quantization": {
            "maximum_final_output_relative_error": max(
                item.final_output_reconstruction_relative_error
                for item in augmentation.values()
            ),
            "maximum_final_input_relative_error": max(
                item.final_input_reconstruction_relative_error
                for item in augmentation.values()
            ),
        },
        "prompt_reconstruction_after_quantization": {
            "maximum_module_output_relative_error": max(
                post_quantization_module_errors.values()
            ),
            "mean_module_output_relative_error": sum(
                post_quantization_module_errors.values()
            ) / len(post_quantization_module_errors),
            "per_module": post_quantization_module_errors,
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
            "Exact and mixed-hybrid paths receive the same authoritative exact "
            "prefix at every unseen continuation position."
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
            "advance only when the fixed mixed 405B budget passes, first-"
            "divergence exact-token rank is at most 32, and top-32 coverage is "
            "at least 0.95"
        ),
        "decision": (
            "advance mixed-precision hybrid certificate"
            if advance
            else "reject tested mixed-precision hybrid allocation"
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
                "post_quant_prompt_error": result[
                    "prompt_reconstruction_after_quantization"
                ],
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
