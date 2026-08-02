from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hot_candidate_coverage import DEFAULT_WIDTHS, last_logits
from scripts.run_nonuniform_rank_frontier import (
    DEFAULT_BUILD_PROMPTS,
    capture_exact_prompt_io,
    compile_allocated_capsules,
    suffix_summary,
)
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
from vortex_runtime.rolling_refresh_budget import (
    full_model_refresh_cost,
    managed_o_down_refresh_cost,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a causal rolling nonuniform response-capsule dictionary. "
            "Only exact prompt prefills and scheduled exact anchor states may "
            "change the capsules; unseen continuation states remain evaluation-only."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--refresh-interval", type=int, required=True)
    parser.add_argument("--anchor-window", type=int, default=16)
    parser.add_argument("--uniform-equivalent-rank", type=int, default=72)
    parser.add_argument("--profile-max-rank", type=int, default=192)
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
        default=Path("rolling_nonuniform_frontier.json"),
    )
    return parser.parse_args()


def capture_exact_anchor(
    *,
    model: torch.nn.Module,
    encoded: Mapping[str, torch.Tensor],
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
            value = hook_args[0].detach().to("cpu", torch.float32)
            captured_inputs[key] = value.reshape(-1, value.shape[-1])[-1:].contiguous()

        def output_hook(
            _module: torch.nn.Module,
            _hook_args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            value = output.detach().to("cpu", torch.float32)
            captured_outputs[key] = value.reshape(-1, value.shape[-1])[-1:].contiguous()

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    for module in replacements.values():
        module.set_mode("exact")
    try:
        with torch.inference_mode():
            result = model(
                **encoded,
                use_cache=False,
                return_dict=True,
            )
            logits = result.logits[0, -1].detach()
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        name
        for name in replacements
        if name not in captured_inputs or name not in captured_outputs
    ]
    if missing:
        raise RuntimeError(f"missing exact anchor captures for {missing[:5]}")
    return logits, captured_inputs, captured_outputs


def current_observations(
    *,
    fixed_inputs: Mapping[str, torch.Tensor],
    fixed_outputs: Mapping[str, torch.Tensor],
    anchor_inputs: Mapping[str, deque[torch.Tensor]],
    anchor_outputs: Mapping[str, deque[torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs: dict[str, torch.Tensor] = {}
    outputs: dict[str, torch.Tensor] = {}
    for name in fixed_inputs:
        input_parts = [fixed_inputs[name], *list(anchor_inputs[name])]
        output_parts = [fixed_outputs[name], *list(anchor_outputs[name])]
        inputs[name] = torch.cat(input_parts, dim=0)
        outputs[name] = torch.cat(output_parts, dim=0)
    return inputs, outputs


def refresh_capsules(
    *,
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    captured_inputs: Mapping[str, torch.Tensor],
    captured_outputs: Mapping[str, torch.Tensor],
    uniform_equivalent_rank: int,
    profile_max_rank: int,
    minimum_rank: int,
    capsule_bits: int,
    rank_rtol: float,
) -> dict[str, object]:
    profiles: dict[str, ModuleRankProfile] = {}
    for name, module in replacements.items():
        profiles[name] = profile_module_rank_value(
            name=name,
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            bias=module.exact.bias,
            maximum_rank=profile_max_rank,
            bits=capsule_bits,
            rank_rtol=rank_rtol,
        )
    byte_budget = uniform_equivalent_byte_budget(
        profiles,
        rank=uniform_equivalent_rank,
    )
    allocation = allocate_nonuniform_ranks(
        profiles,
        byte_budget=byte_budget,
        minimum_rank=minimum_rank,
    )
    compile_stats = compile_allocated_capsules(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        allocated_ranks=allocation.ranks,
        rank_rtol=rank_rtol,
    )
    quantization, per_module_quantization = fake_quantize_response_capsules(
        replacements,
        bits=capsule_bits,
    )
    ranks = list(allocation.ranks.values())
    return {
        "observation_vectors": {
            name: int(value.shape[0]) for name, value in captured_inputs.items()
        },
        "byte_budget": byte_budget,
        "allocation": allocation.to_dict(),
        "rank_statistics": {
            "minimum": min(ranks),
            "maximum": max(ranks),
            "mean": sum(ranks) / len(ranks),
            "total": sum(ranks),
            "suffix_summary": suffix_summary(allocation.ranks),
        },
        "compile_stats": {
            name: item.to_dict() for name, item in compile_stats.items()
        },
        "quantization": {
            "aggregate": quantization.to_dict(),
            "per_module": per_module_quantization,
        },
    }


def offset_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if bool(row["is_anchor"]):
            continue
        grouped[int(row["distance_from_anchor"])].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for offset, values in sorted(grouped.items()):
        result[str(offset)] = {
            "count": len(values),
            "top1": sum(bool(value["exact_match"]) for value in values) / len(values),
            "top32": sum(int(value["exact_token_rank"]) <= 32 for value in values)
            / len(values),
            "mean_exact_token_rank": sum(
                int(value["exact_token_rank"]) for value in values
            )
            / len(values),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.refresh_interval <= 0:
        raise SystemExit("tokens and refresh interval must be positive")
    if args.anchor_window <= 0:
        raise SystemExit("anchor window must be positive")
    if args.uniform_equivalent_rank <= 0 or args.profile_max_rank <= 0:
        raise SystemExit("rank limits must be positive")
    if args.minimum_rank <= 0:
        raise SystemExit("minimum rank must be positive")
    if not 2 <= args.capsule_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")

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
    started = time.perf_counter()
    fixed_inputs, fixed_outputs = capture_exact_prompt_io(
        model=model,
        tokenizer=tokenizer,
        replacements=replacements,
        prompts=[*build_prompts, args.eval_prompt],
        device=device,
    )
    anchor_inputs: dict[str, deque[torch.Tensor]] = {
        name: deque(maxlen=args.anchor_window) for name in replacements
    }
    anchor_outputs: dict[str, deque[torch.Tensor]] = {
        name: deque(maxlen=args.anchor_window) for name in replacements
    }

    observations_in, observations_out = current_observations(
        fixed_inputs=fixed_inputs,
        fixed_outputs=fixed_outputs,
        anchor_inputs=anchor_inputs,
        anchor_outputs=anchor_outputs,
    )
    refresh_history: list[dict[str, object]] = [
        {
            "position": 0,
            "kind": "exact_prefill",
            **refresh_capsules(
                replacements=replacements,
                captured_inputs=observations_in,
                captured_outputs=observations_out,
                uniform_equivalent_rank=args.uniform_equivalent_rank,
                profile_max_rank=args.profile_max_rank,
                minimum_rank=args.minimum_rank,
                capsule_bits=args.capsule_bits,
                rank_rtol=args.rank_rtol,
            ),
        }
    ]

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    rows: list[dict[str, object]] = []
    hot_rows: list[CandidateCoverageRow] = []
    last_anchor_position = 0
    decode_anchor_count = 0

    for position in range(args.tokens):
        is_anchor = position > 0 and position % args.refresh_interval == 0
        if is_anchor:
            exact_logits, new_inputs, new_outputs = capture_exact_anchor(
                model=model,
                encoded={"input_ids": prefix},
                replacements=replacements,
            )
            for name in replacements:
                anchor_inputs[name].append(new_inputs[name])
                anchor_outputs[name].append(new_outputs[name])
            observations_in, observations_out = current_observations(
                fixed_inputs=fixed_inputs,
                fixed_outputs=fixed_outputs,
                anchor_inputs=anchor_inputs,
                anchor_outputs=anchor_outputs,
            )
            refresh_history.append(
                {
                    "position": position,
                    "kind": "scheduled_exact_anchor",
                    **refresh_capsules(
                        replacements=replacements,
                        captured_inputs=observations_in,
                        captured_outputs=observations_out,
                        uniform_equivalent_rank=args.uniform_equivalent_rank,
                        profile_max_rank=args.profile_max_rank,
                        minimum_rank=args.minimum_rank,
                        capsule_bits=args.capsule_bits,
                        rank_rtol=args.rank_rtol,
                    ),
                }
            )
            decode_anchor_count += 1
            last_anchor_position = position
        else:
            for module in replacements.values():
                module.set_mode("exact")
            exact_logits = last_logits(model, prefix)

        exact_token = int(torch.argmax(exact_logits).item())
        for module in replacements.values():
            module.set_mode("project")
        hot_logits = last_logits(model, prefix)
        hot_token = int(torch.argmax(hot_logits).item())
        rank = token_rank(hot_logits, exact_token)
        candidate_token = exact_token if is_anchor else hot_token
        row = {
            "position": position,
            "is_anchor": is_anchor,
            "distance_from_anchor": position - last_anchor_position,
            "exact_token": exact_token,
            "hot_token": hot_token,
            "candidate_token": candidate_token,
            "exact_token_rank": rank,
            "hot_top1_margin": top1_margin(hot_logits),
            "exact_logit_gap_from_hot_top1": float(
                (hot_logits[hot_token] - hot_logits[exact_token]).item()
            ),
            "exact_match": hot_token == exact_token,
            "candidate_match": candidate_token == exact_token,
        }
        rows.append(row)
        if not is_anchor:
            hot_rows.append(
                CandidateCoverageRow(
                    position=position,
                    exact_token=exact_token,
                    hot_token=hot_token,
                    exact_token_rank=rank,
                    hot_top1_margin=float(row["hot_top1_margin"]),
                    exact_logit_gap_from_hot_top1=float(
                        row["exact_logit_gap_from_hot_top1"]
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

    if not hot_rows:
        raise RuntimeError("rolling frontier requires at least one hot position")
    hot_coverage = coverage_at_k(hot_rows, DEFAULT_WIDTHS)
    first_hot_divergence = next((row for row in hot_rows if not row.exact_match), None)
    hot_exact_ranks = [row.exact_token_rank for row in hot_rows]
    hot_budget = rank_budget_point(
        args.uniform_equivalent_rank,
        capsule_bits=args.capsule_bits,
    )
    managed_refresh = managed_o_down_refresh_cost(
        refresh_interval=args.refresh_interval,
        hot_budget=hot_budget,
    )
    full_refresh = full_model_refresh_cost(
        refresh_interval=args.refresh_interval,
        hot_budget=hot_budget,
    )
    accuracy_pass = (
        (first_hot_divergence is None or first_hot_divergence.exact_token_rank <= 32)
        and hot_coverage["32"] >= 0.95
    )
    lower_bound_qualifies = hot_budget.pass_all and managed_refresh.pass_all and accuracy_pass
    final_gate_compatible = lower_bound_qualifies and full_refresh.pass_all

    result = {
        "evidence_level": "E1 causal rolling nonuniform trajectory frontier",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "generic_build_prompt_count": len(build_prompts),
        "session_prompt_included_for_initial_prefill": True,
        "evaluated_tokens": len(rows),
        "refresh_interval": args.refresh_interval,
        "decode_anchor_count": decode_anchor_count,
        "observed_anchor_fraction": decode_anchor_count / len(rows),
        "anchor_window": args.anchor_window,
        "causal_contract": (
            "Initial capsules use exact disjoint generic prompt prefills and the "
            "exact current user-prompt prefill. Later capsule changes use only "
            "scheduled exact prefix anchors strictly before the continuation "
            "states they are evaluated on. Non-anchor continuation outputs never "
            "enter the dictionary."
        ),
        "candidate_commit_contract": (
            "Scheduled anchor positions commit the exact token and are charged as "
            "repairs. Other positions commit the hot token. All diagnostics keep "
            "the authoritative exact prefix to isolate local representation error."
        ),
        "uniform_equivalent": {
            "rank": args.uniform_equivalent_rank,
            "capsule_bits": args.capsule_bits,
            "gate0_hot_budget": hot_budget.to_dict(),
        },
        "refresh_budget": {
            "managed_o_down_lower_bound": managed_refresh.to_dict(),
            "full_model_anchor": full_refresh.to_dict(),
            "warning": (
                "The managed O/down estimate omits basis maintenance and capsule "
                "writes, so passing it is necessary but not sufficient."
            ),
        },
        "refresh_history": refresh_history,
        "overall_candidate_top1_match_rate": sum(
            bool(row["candidate_match"]) for row in rows
        )
        / len(rows),
        "hot_only_top1_match_rate": sum(row.exact_match for row in hot_rows)
        / len(hot_rows),
        "hot_only_coverage_at_k": hot_coverage,
        "hot_rank_statistics": {
            "minimum": min(hot_exact_ranks),
            "maximum": max(hot_exact_ranks),
            "mean": sum(hot_exact_ranks) / len(hot_exact_ranks),
        },
        "first_hot_divergence": (
            None
            if first_hot_divergence is None
            else {
                "position": first_hot_divergence.position,
                "exact_token": first_hot_divergence.exact_token,
                "hot_token": first_hot_divergence.hot_token,
                "exact_token_rank": first_hot_divergence.exact_token_rank,
                "hot_top1_margin": first_hot_divergence.hot_top1_margin,
                "exact_logit_gap_from_hot_top1": (
                    first_hot_divergence.exact_logit_gap_from_hot_top1
                ),
            }
        ),
        "offset_from_anchor": offset_summary(rows),
        "rows": rows,
        "decision_rule": (
            "advance only when the fixed hot budget passes, even the optimistic "
            "O/down exact-anchor lower bound fits traffic and compute, first hot "
            "divergence rank is at most 32, and hot top-32 coverage is at least "
            "0.95. Full-model compatibility additionally requires the full exact "
            "anchor budget to pass."
        ),
        "accuracy_pass": accuracy_pass,
        "lower_bound_qualifies": lower_bound_qualifies,
        "final_gate_compatible": final_gate_compatible,
        "decision": (
            "advance rolling nonuniform trajectory dictionary"
            if final_gate_compatible
            else (
                "accuracy survives but refresh cost rejects this cadence"
                if accuracy_pass
                else "reject tested rolling nonuniform cadence"
            )
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
                "refresh_interval": result["refresh_interval"],
                "decode_anchor_count": result["decode_anchor_count"],
                "managed_refresh_budget": result["refresh_budget"][
                    "managed_o_down_lower_bound"
                ],
                "full_refresh_budget": result["refresh_budget"][
                    "full_model_anchor"
                ],
                "overall_candidate_top1_match_rate": result[
                    "overall_candidate_top1_match_rate"
                ],
                "hot_only_top1_match_rate": result["hot_only_top1_match_rate"],
                "hot_only_coverage_at_k": result["hot_only_coverage_at_k"],
                "first_hot_divergence": result["first_hot_divergence"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
