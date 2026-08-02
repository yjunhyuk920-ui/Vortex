from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hot_candidate_coverage import DEFAULT_WIDTHS, last_logits
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.capsule_quantization import fake_quantize_response_capsules
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    replace_with_decision_tile_modules,
)
from vortex_runtime.nonuniform_rank_allocator import (
    ModuleRankProfile,
    allocate_nonuniform_ranks,
    profile_module_rank_value,
    uniform_equivalent_byte_budget,
)
from vortex_runtime.rank_frontier import rank_budget_point
from vortex_runtime.session_prefill_basis import compile_session_response_basis


DEFAULT_BUILD_PROMPTS = (
    "Explain how a hash table handles collisions and compare chaining with open addressing.",
    "Write a Python function that topologically sorts a directed acyclic graph.",
    "다음 요구사항을 만족하는 재고 관리 API의 데이터 모델을 설계해줘.",
    "Translate into Korean: The deployment was rolled back after tail latency increased.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate one fixed 405B-equivalent response-capsule byte budget "
            "non-uniformly across real model modules using prompt-only marginal "
            "output benefit, then measure unseen-continuation candidate coverage."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--uniform-equivalent-rank", type=int, default=72)
    parser.add_argument("--profile-max-rank", type=int, default=144)
    parser.add_argument("--minimum-rank", type=int, default=1)
    parser.add_argument("--capsule-bits", type=int, default=8)
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
        default=Path("nonuniform_rank_frontier.json"),
    )
    return parser.parse_args()


def capture_exact_prompt_io(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    prompts: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    captured_inputs: dict[str, list[torch.Tensor]] = {
        name: [] for name in replacements
    }
    captured_outputs: dict[str, list[torch.Tensor]] = {
        name: [] for name in replacements
    }
    handles: list[Any] = []

    for name, module in replacements.items():
        def pre_hook(
            _module: torch.nn.Module,
            hook_args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            value = hook_args[0].detach().to("cpu", torch.float32)
            captured_inputs[key].append(value.reshape(-1, value.shape[-1]))

        def output_hook(
            _module: torch.nn.Module,
            _hook_args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            value = output.detach().to("cpu", torch.float32)
            captured_outputs[key].append(value.reshape(-1, value.shape[-1]))

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    for module in replacements.values():
        module.set_mode("exact")
    try:
        with torch.inference_mode():
            for prompt in prompts:
                encoded = encode_prompt(tokenizer, prompt, device)
                model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        name
        for name in replacements
        if not captured_inputs[name] or not captured_outputs[name]
    ]
    if missing:
        raise RuntimeError(f"missing exact prompt captures for {missing[:5]}")
    return (
        {name: torch.cat(parts, dim=0) for name, parts in captured_inputs.items()},
        {name: torch.cat(parts, dim=0) for name, parts in captured_outputs.items()},
    )


def compile_allocated_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    captured_inputs: Mapping[str, torch.Tensor],
    captured_outputs: Mapping[str, torch.Tensor],
    allocated_ranks: Mapping[str, int],
    rank_rtol: float,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    for name, module in replacements.items():
        requested_rank = int(allocated_ranks[name])
        if requested_rank <= 0:
            raise RuntimeError(f"allocated rank for {name} must be positive")
        module.atlas.max_rank = requested_rank
        item = compile_session_response_basis(
            module,
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            max_rank=requested_rank,
            rank_rtol=rank_rtol,
        )
        stats[name] = item
    return stats


def suffix_summary(ranks: Mapping[str, int]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = {}
    for name, rank in ranks.items():
        suffix = "mlp.down_proj" if name.endswith("mlp.down_proj") else "self_attn.o_proj"
        groups.setdefault(suffix, []).append(int(rank))
    return {
        suffix: {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "total": sum(values),
        }
        for suffix, values in groups.items()
    }


def main() -> None:
    args = parse_args()
    if args.tokens <= 0:
        raise SystemExit("token count must be positive")
    if args.uniform_equivalent_rank <= 0 or args.profile_max_rank <= 0:
        raise SystemExit("rank limits must be positive")
    if args.minimum_rank <= 0:
        raise SystemExit("minimum rank must be positive")
    if not 2 <= args.capsule_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")
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
        max_rank=args.profile_max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    build_prompts = list(args.build_prompts or DEFAULT_BUILD_PROMPTS)
    if args.eval_prompt in build_prompts:
        raise ValueError("generic build prompts and evaluation prompt must be disjoint")
    causal_compile_prompts = [*build_prompts, args.eval_prompt]
    started = time.perf_counter()
    captured_inputs, captured_outputs = capture_exact_prompt_io(
        model=model,
        tokenizer=tokenizer,
        replacements=replacements,
        prompts=causal_compile_prompts,
        device=device,
    )

    profiles: dict[str, ModuleRankProfile] = {}
    for name, module in replacements.items():
        profiles[name] = profile_module_rank_value(
            name=name,
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            bias=module.exact.bias,
            maximum_rank=args.profile_max_rank,
            bits=args.capsule_bits,
            rank_rtol=args.rank_rtol,
        )

    byte_budget = uniform_equivalent_byte_budget(
        profiles,
        rank=args.uniform_equivalent_rank,
    )
    allocation = allocate_nonuniform_ranks(
        profiles,
        byte_budget=byte_budget,
        minimum_rank=args.minimum_rank,
    )
    compile_stats = compile_allocated_capsules(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        allocated_ranks=allocation.ranks,
        rank_rtol=args.rank_rtol,
    )
    quantization, per_module_quantization = fake_quantize_response_capsules(
        replacements,
        bits=args.capsule_bits,
    )

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    rows: list[CandidateCoverageRow] = []
    for position in range(args.tokens):
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
                position=position,
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
    first_rank = None if first_divergence is None else first_divergence.exact_token_rank
    budget = rank_budget_point(
        args.uniform_equivalent_rank,
        capsule_bits=args.capsule_bits,
    )
    coverage_pass = (
        (first_rank is None or first_rank <= 32)
        and coverage["32"] >= 0.95
    )
    advance = budget.pass_all and coverage_pass

    allocated_values = list(allocation.ranks.values())
    compiled_values = [int(item.compiled_rank) for item in compile_stats.values()]
    result = {
        "evidence_level": "E1 causal prompt-valued nonuniform rank frontier",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "generic_build_prompt_count": len(build_prompts),
        "session_prompt_included_for_compilation": True,
        "evaluated_unseen_tokens": len(rows),
        "causal_contract": (
            "Rank values, input bases, and response images use only fixed "
            "disjoint generic prompt prefills plus the exact current user-prompt "
            "prefill. Continuation tokens and continuation activations are "
            "evaluation-only."
        ),
        "allocation_contract": (
            "The total logical per-column byte budget equals a uniform-rank "
            "candidate at the same precision. Contiguous module ranks are chosen "
            "greedily by exact prompt output-energy gain per logical byte."
        ),
        "uniform_equivalent": {
            "rank": args.uniform_equivalent_rank,
            "capsule_bits": args.capsule_bits,
            "byte_budget": byte_budget,
            "gate0_budget": budget.to_dict(),
        },
        "allocation": {
            **allocation.to_dict(),
            "minimum_rank": min(allocated_values),
            "maximum_rank": max(allocated_values),
            "mean_rank": sum(allocated_values) / len(allocated_values),
            "total_rank": sum(allocated_values),
            "suffix_summary": suffix_summary(allocation.ranks),
        },
        "compiled_rank_statistics": {
            "minimum": min(compiled_values),
            "maximum": max(compiled_values),
            "mean": sum(compiled_values) / len(compiled_values),
            "total": sum(compiled_values),
        },
        "profiles": {
            name: profile.to_dict() for name, profile in profiles.items()
        },
        "compile_stats": {
            name: item.to_dict() for name, item in compile_stats.items()
        },
        "quantization": {
            "aggregate": quantization.to_dict(),
            "per_module": per_module_quantization,
        },
        "same_context_contract": (
            "Exact and hot paths receive the same authoritative exact prefix at "
            "every unseen continuation position."
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
            "advance only when the uniform-equivalent 405B budget passes, "
            "first-divergence exact-token rank is at most 32, and top-32 "
            "coverage is at least 0.95"
        ),
        "qualifies": advance,
        "decision": (
            "advance prompt-valued nonuniform rank allocation"
            if advance
            else "reject tested prompt-valued nonuniform allocation point"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "uniform_equivalent": result["uniform_equivalent"],
                "allocation": result["allocation"],
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
