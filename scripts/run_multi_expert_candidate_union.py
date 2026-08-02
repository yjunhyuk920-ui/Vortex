from __future__ import annotations

import argparse
from dataclasses import asdict
from math import ceil
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hot_candidate_coverage import last_logits
from scripts.run_oracle_block_shared_adjoint import (
    DEFAULT_BUILD_PROMPTS,
    encode_prompt,
    generate,
    require_transformers,
)
from vortex_runtime.candidate_coverage import token_rank, top1_margin
from vortex_runtime.capsule_quantization import fake_quantize_response_capsules
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    replace_with_decision_tile_modules,
)
from vortex_runtime.multi_expert_candidates import (
    ExpertCandidateRow,
    best_budget_compatible_fallback,
    margin_fallback_frontier,
    summarize_fixed_union,
    topk_token_ids,
)
from vortex_runtime.rank_frontier import rank_budget_point
from vortex_runtime.session_prefill_basis import compile_session_response_bases


CapsuleSnapshot = dict[str, tuple[torch.Tensor, torch.Tensor]]
FIXED_ALLOCATIONS = ((0, 32), (8, 24), (16, 16), (24, 8), (32, 0))
ROUTER_ALLOCATIONS = ((8, 24), (16, 16), (24, 8))
DIAGNOSTIC_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep generic and exact-prompt response capsules as independent "
            "experts. Measure their real token-ID candidate unions and causal "
            "top-1-margin fallback frontier under the fixed 405B budget."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--generic-rank", type=int, default=56)
    parser.add_argument("--session-rank", type=int, default=45)
    parser.add_argument("--generic-bits", type=int, default=8)
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
        default=Path("multi_expert_candidate_union.json"),
    )
    return parser.parse_args()


def snapshot_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
) -> CapsuleSnapshot:
    return {
        name: (
            module.atlas.input_basis.detach().to("cpu").clone().contiguous(),
            module.atlas.output_image.detach().to("cpu").clone().contiguous(),
        )
        for name, module in replacements.items()
    }


def restore_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    snapshot: CapsuleSnapshot,
) -> None:
    if set(replacements) != set(snapshot):
        raise ValueError("capsule snapshot does not match replacement modules")
    for name, module in replacements.items():
        basis, image = snapshot[name]
        module.atlas.input_basis = basis
        module.atlas.output_image = image
        module.atlas.max_rank = max(module.atlas.max_rank, basis.shape[1])
        module.set_mode("project")


def clear_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    max_rank: int,
) -> None:
    for module in replacements.values():
        module.atlas.input_basis = torch.empty(
            (module.exact.in_features, 0),
            dtype=module.atlas.basis_dtype,
        )
        module.atlas.output_image = torch.empty(
            (module.exact.out_features, 0),
            dtype=module.atlas.basis_dtype,
        )
        module.atlas.max_rank = max_rank


def capture_exact_prompt_io(
    *,
    model: torch.nn.Module,
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    encoded: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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

    for module in replacements.values():
        module.set_mode("exact")
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
    return captured_inputs, captured_outputs


def maximum_secondary_fraction(
    *,
    primary: Any,
    secondary: Any,
) -> dict[str, float]:
    traffic_headroom = (
        primary.traffic_limit_gib_per_token
        - primary.hot_traffic_gib_per_token
    )
    compute_headroom = (
        primary.compute_limit_gflop_per_token
        - primary.hot_compute_gflop_per_token
    )
    traffic_fraction = max(
        0.0,
        traffic_headroom / secondary.hot_traffic_gib_per_token,
    )
    compute_fraction = max(
        0.0,
        compute_headroom / secondary.hot_compute_gflop_per_token,
    )
    return {
        "traffic": min(1.0, traffic_fraction),
        "compute": min(1.0, compute_fraction),
        "combined": min(1.0, traffic_fraction, compute_fraction),
    }


def enrich_fallback(
    item: Any,
    *,
    primary_budget: Any,
    secondary_budget: Any,
) -> dict[str, Any]:
    fraction = item.secondary_invocation_fraction
    traffic = (
        primary_budget.hot_traffic_gib_per_token
        + fraction * secondary_budget.hot_traffic_gib_per_token
    )
    compute = (
        primary_budget.hot_compute_gflop_per_token
        + fraction * secondary_budget.hot_compute_gflop_per_token
    )
    return {
        **item.to_dict(),
        "projected_traffic_gib_per_token": traffic,
        "traffic_limit_gib_per_token": primary_budget.traffic_limit_gib_per_token,
        "traffic_pass": traffic <= primary_budget.traffic_limit_gib_per_token,
        "projected_compute_gflop_per_token": compute,
        "compute_limit_gflop_per_token": primary_budget.compute_limit_gflop_per_token,
        "compute_pass": compute <= primary_budget.compute_limit_gflop_per_token,
    }


def build_router_direction(
    *,
    name: str,
    rows: list[ExpertCandidateRow],
    primary_budget: Any,
    secondary_budget: Any,
) -> dict[str, Any]:
    fractions = maximum_secondary_fraction(
        primary=primary_budget,
        secondary=secondary_budget,
    )
    allocation_results: list[dict[str, Any]] = []
    for primary_k, secondary_k in ROUTER_ALLOCATIONS:
        frontier = margin_fallback_frontier(
            rows,
            primary_k=primary_k,
            secondary_k=secondary_k,
        )
        best = best_budget_compatible_fallback(
            frontier,
            maximum_secondary_fraction=fractions["combined"],
        )
        allocation_results.append(
            {
                "primary_k": primary_k,
                "secondary_k": secondary_k,
                "best_budget_compatible": (
                    None
                    if best is None
                    else enrich_fallback(
                        best,
                        primary_budget=primary_budget,
                        secondary_budget=secondary_budget,
                    )
                ),
                "frontier": [
                    enrich_fallback(
                        item,
                        primary_budget=primary_budget,
                        secondary_budget=secondary_budget,
                    )
                    for item in frontier
                ],
            }
        )

    candidates = [
        item["best_budget_compatible"]
        for item in allocation_results
        if item["best_budget_compatible"] is not None
    ]
    best_overall = (
        None
        if not candidates
        else max(
            candidates,
            key=lambda item: (
                float(item["coverage"]),
                -float(item["secondary_invocation_fraction"]),
                -float(item["mean_candidate_count"]),
            ),
        )
    )
    return {
        "name": name,
        "maximum_secondary_fraction": fractions,
        "primary_budget": primary_budget.to_dict(),
        "secondary_budget": secondary_budget.to_dict(),
        "allocations": allocation_results,
        "best_budget_compatible": best_overall,
    }


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("token counts must be positive")
    if args.generic_rank <= 0 or args.session_rank <= 0:
        raise SystemExit("expert ranks must be positive")
    if not 2 <= args.generic_bits <= 16 or not 2 <= args.session_bits <= 16:
        raise SystemExit("expert capsule bits must be between 2 and 16")
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

    maximum_rank = max(args.generic_rank, args.session_rank)
    suffixes = tuple(args.suffixes or ("self_attn.o_proj", "mlp.down_proj"))
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=maximum_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    clear_capsules(replacements, max_rank=args.generic_rank)
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
    generic_ranks = [module.atlas.rank for module in replacements.values()]
    generic_quantization, _ = fake_quantize_response_capsules(
        replacements,
        bits=args.generic_bits,
    )
    generic_snapshot = snapshot_capsules(replacements)

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    captured_inputs, captured_outputs = capture_exact_prompt_io(
        model=model,
        replacements=replacements,
        encoded=encoded,
    )
    clear_capsules(replacements, max_rank=args.session_rank)
    session_compile = compile_session_response_bases(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        max_rank=args.session_rank,
        rank_rtol=args.rank_rtol,
    )
    session_ranks = [item.compiled_rank for item in session_compile.values()]
    session_quantization, _ = fake_quantize_response_capsules(
        replacements,
        bits=args.session_bits,
    )
    session_snapshot = snapshot_capsules(replacements)

    prefix = encoded["input_ids"]
    maximum_k = max(DIAGNOSTIC_WIDTHS)
    session_primary_rows: list[ExpertCandidateRow] = []
    generic_primary_rows: list[ExpertCandidateRow] = []
    token_diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()

    for position in range(args.tokens):
        for module in replacements.values():
            module.set_mode("exact")
        exact_logits = last_logits(model, prefix)
        exact_token = int(torch.argmax(exact_logits).item())

        restore_capsules(replacements, generic_snapshot)
        generic_logits = last_logits(model, prefix)
        generic_ids = topk_token_ids(generic_logits, maximum_k)

        restore_capsules(replacements, session_snapshot)
        session_logits = last_logits(model, prefix)
        session_ids = topk_token_ids(session_logits, maximum_k)

        session_primary_rows.append(
            ExpertCandidateRow(
                position=position,
                exact_token=exact_token,
                primary_margin=top1_margin(session_logits),
                primary_candidates=session_ids,
                secondary_candidates=generic_ids,
            )
        )
        generic_primary_rows.append(
            ExpertCandidateRow(
                position=position,
                exact_token=exact_token,
                primary_margin=top1_margin(generic_logits),
                primary_candidates=generic_ids,
                secondary_candidates=session_ids,
            )
        )
        token_diagnostics.append(
            {
                "position": position,
                "exact_token": exact_token,
                "generic_exact_token_rank": token_rank(
                    generic_logits,
                    exact_token,
                ),
                "session_exact_token_rank": token_rank(
                    session_logits,
                    exact_token,
                ),
                "generic_top1_margin": top1_margin(generic_logits),
                "session_top1_margin": top1_margin(session_logits),
                "generic_top1_token": int(torch.argmax(generic_logits).item()),
                "session_top1_token": int(torch.argmax(session_logits).item()),
            }
        )
        prefix = torch.cat(
            (
                prefix,
                torch.tensor([[exact_token]], dtype=torch.long, device=device),
            ),
            dim=-1,
        )

    fixed_allocations = [
        summarize_fixed_union(
            session_primary_rows,
            primary_k=session_k,
            secondary_k=generic_k,
        ).to_dict()
        for session_k, generic_k in FIXED_ALLOCATIONS
    ]
    equal_width_diagnostics = [
        summarize_fixed_union(
            session_primary_rows,
            primary_k=k,
            secondary_k=k,
        ).to_dict()
        for k in DIAGNOSTIC_WIDTHS
    ]

    generic_budget = rank_budget_point(
        args.generic_rank,
        capsule_bits=args.generic_bits,
    )
    session_budget = rank_budget_point(
        args.session_rank,
        capsule_bits=args.session_bits,
    )
    stored_equivalent_rank_8bit = ceil(
        (
            args.generic_rank * args.generic_bits
            + args.session_rank * args.session_bits
        )
        / 8
    )
    dictionary_budget = rank_budget_point(
        stored_equivalent_rank_8bit,
        capsule_bits=8,
    )

    routers = [
        build_router_direction(
            name="session_primary_generic_fallback",
            rows=session_primary_rows,
            primary_budget=session_budget,
            secondary_budget=generic_budget,
        ),
        build_router_direction(
            name="generic_primary_session_fallback",
            rows=generic_primary_rows,
            primary_budget=generic_budget,
            secondary_budget=session_budget,
        ),
    ]
    router_candidates = [
        router["best_budget_compatible"]
        for router in routers
        if router["best_budget_compatible"] is not None
    ]
    best_router = (
        None
        if not router_candidates
        else max(
            router_candidates,
            key=lambda item: (
                float(item["coverage"]),
                -float(item["secondary_invocation_fraction"]),
                -float(item["mean_candidate_count"]),
            ),
        )
    )
    router_pass = bool(
        dictionary_budget.memory_pass
        and best_router is not None
        and best_router["traffic_pass"]
        and best_router["compute_pass"]
        and best_router["maximum_candidate_count"] <= 32
        and best_router["coverage"] >= 0.95
    )

    result = {
        "evidence_level": "E1 causal independent-capsule candidate union",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "generic_expert": {
            "rank_limit": args.generic_rank,
            "capsule_bits": args.generic_bits,
            "built_rank_statistics": {
                "minimum": min(generic_ranks),
                "maximum": max(generic_ranks),
                "mean": sum(generic_ranks) / len(generic_ranks),
            },
            "quantization": generic_quantization.to_dict(),
            "budget": generic_budget.to_dict(),
        },
        "session_expert": {
            "rank_limit": args.session_rank,
            "capsule_bits": args.session_bits,
            "compiled_rank_statistics": {
                "minimum": min(session_ranks),
                "maximum": max(session_ranks),
                "mean": sum(session_ranks) / len(session_ranks),
            },
            "quantization": session_quantization.to_dict(),
            "budget": session_budget.to_dict(),
        },
        "dictionary_budget": {
            "stored_equivalent_rank_8bit": stored_equivalent_rank_8bit,
            **dictionary_budget.to_dict(),
        },
        "causal_contract": (
            "The generic capsule uses fixed disjoint prompts. The session "
            "capsule uses only exact user-prompt prefill inputs and outputs. "
            "Fallback triggers use only the primary expert top-1 margin. Exact "
            "continuation tokens are evaluation-only."
        ),
        "evaluated_tokens": len(session_primary_rows),
        "fixed_total_candidate_allocations": fixed_allocations,
        "equal_per_expert_width_diagnostics": equal_width_diagnostics,
        "router_directions": routers,
        "best_budget_compatible_router": best_router,
        "token_diagnostics": token_diagnostics,
        "decision_rule": (
            "advance only when both capsules fit dictionary memory, the causal "
            "margin fallback fits average 405B traffic and compute, uses at most "
            "32 distinct candidates, and reaches at least 0.95 exact-token coverage"
        ),
        "decision": (
            "advance independent-capsule causal certificate"
            if router_pass
            else "reject tested margin-routed two-capsule certificate"
        ),
        "next_candidate": (
            "construct the sound token certificate for the passing margin router"
            if router_pass
            else (
                "add a recent-certified-prefix local capsule and learn a causal "
                "module/expert router without merging response bases"
            )
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
                "dictionary_budget": result["dictionary_budget"],
                "fixed_allocations": fixed_allocations,
                "equal_width_diagnostics": equal_width_diagnostics,
                "best_router": best_router,
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
