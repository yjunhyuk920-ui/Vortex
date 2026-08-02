from __future__ import annotations

import argparse
import gc
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_layer_local_precision_oracle import (
    apply_group_precision,
    encode_prompt,
    error_positions,
    greedy_tokens,
    load_model,
    require_transformers,
    restore_group,
    teacher_forced_logits,
)
from vortex_runtime.candidate_coverage import top1_margin
from vortex_runtime.feasibility import default_specs
from vortex_runtime.layer_precision_oracle import (
    precision_module_groups,
    unique_weight_elements,
)
from vortex_runtime.precision_consensus import (
    PrecisionConsensusRow,
    analyze_precision_consensus,
)
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules
from vortex_runtime.token_routed_precision import token_routed_refinement_budget

DEFAULT_CANDIDATES = (
    "io",
    "layers_002_003",
    "layers_010_011",
    "layers_000_001",
    "layers_018_019",
    "layers_006_007",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use exact labels as an oracle to find the minimum residual layer "
            "route for each consensus-flagged token, while charging transfer "
            "for the block-wide union of all selected routes."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--layers-per-group", type=int, default=2)
    parser.add_argument("--margin-threshold", type=float, default=0.4)
    parser.add_argument(
        "--candidate-groups",
        default=",".join(DEFAULT_CANDIDATES),
    )
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantized_logits(
    *,
    model_name: str,
    bits: int,
    device: torch.device,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.nn.Module]:
    model = load_model(model_name, device)
    fake_quantize_full_rank_modules(
        model,
        bits=bits,
        source_bits=16,
        row_chunk=row_chunk,
    )
    logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    return logits, model


def mask_groups(mask: int, candidate_groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        group
        for index, group in enumerate(candidate_groups)
        if mask & (1 << index)
    )


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.row_chunk <= 0 or args.layers_per_group <= 0:
        raise SystemExit("tokens, row chunk, and layer grouping must be positive")
    candidate_groups = tuple(
        item.strip() for item in args.candidate_groups.split(",") if item.strip()
    )
    if not candidate_groups or len(candidate_groups) > 12:
        raise SystemExit("candidate group count must be in [1, 12]")

    _, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    started = time.perf_counter()

    exact_model = load_model(args.model, device)
    exact_tokens = greedy_tokens(
        model=exact_model,
        encoded=encoded,
        count=args.tokens,
    )
    q4_logits, q4_model = quantized_logits(
        model_name=args.model,
        bits=4,
        device=device,
        encoded=encoded,
        exact_tokens=exact_tokens,
        row_chunk=args.row_chunk,
    )
    q6_logits, q6_model = quantized_logits(
        model_name=args.model,
        bits=6,
        device=device,
        encoded=encoded,
        exact_tokens=exact_tokens,
        row_chunk=args.row_chunk,
    )

    consensus_rows: list[PrecisionConsensusRow] = []
    for position in range(args.tokens):
        exact_token = int(exact_tokens[0, position].item())
        q4_position = q4_logits[0, position]
        q6_position = q6_logits[0, position]
        consensus_rows.append(
            PrecisionConsensusRow(
                position=position,
                exact_token=exact_token,
                q4_token=int(torch.argmax(q4_position).item()),
                q6_token=int(torch.argmax(q6_position).item()),
                q6_margin=top1_margin(q6_position),
            )
        )
    consensus = analyze_precision_consensus(
        consensus_rows,
        margin_threshold=args.margin_threshold,
    )
    flagged_positions = [
        row.position
        for row in consensus_rows
        if not (row.agrees and row.q6_margin >= args.margin_threshold)
    ]
    q6_error_positions = error_positions(q6_logits, exact_tokens)
    del q4_model, q4_logits
    gc.collect()

    groups = precision_module_groups(
        q6_model,
        layers_per_group=args.layers_per_group,
    )
    missing = [group for group in candidate_groups if group not in groups]
    if missing:
        raise RuntimeError(f"candidate groups not present: {missing}")
    total_elements = unique_weight_elements(q6_model)
    group_elements = {
        group: unique_weight_elements(q6_model, groups[group])
        for group in candidate_groups
    }
    group_fractions = {
        group: elements / total_elements
        for group, elements in group_elements.items()
    }

    subset_results: dict[int, dict[str, object]] = {}
    for size in range(1, len(candidate_groups) + 1):
        for selected in combinations(range(len(candidate_groups)), size):
            mask = sum(1 << index for index in selected)
            selected_groups = mask_groups(mask, candidate_groups)
            backups_by_group: list[dict[str, torch.Tensor]] = []
            for group in selected_groups:
                backups, _ = apply_group_precision(
                    target_model=q6_model,
                    source_model=exact_model,
                    module_names=groups[group],
                    bits=8,
                    row_chunk=args.row_chunk,
                    keep_applied=False,
                )
                backups_by_group.append(backups)
            logits = teacher_forced_logits(
                model=q6_model,
                encoded=encoded,
                exact_tokens=exact_tokens,
            )
            corrected_positions = [
                position
                for position in flagged_positions
                if int(torch.argmax(logits[0, position]).item())
                == int(exact_tokens[0, position].item())
            ]
            fraction = sum(group_fractions[group] for group in selected_groups)
            subset_results[mask] = {
                "groups": list(selected_groups),
                "layer_fraction": fraction,
                "corrected_flagged_positions": corrected_positions,
            }
            for backups in reversed(backups_by_group):
                restore_group(q6_model, backups)
            del logits

    routes_by_position: dict[int, list[int]] = {}
    for position in flagged_positions:
        base_is_exact = position not in q6_error_positions
        if base_is_exact:
            routes_by_position[position] = [0]
            continue
        routes = [
            mask
            for mask, point in subset_results.items()
            if position in point["corrected_flagged_positions"]
        ]
        if not routes:
            routes_by_position[position] = []
        else:
            routes_by_position[position] = sorted(
                routes,
                key=lambda mask: (
                    subset_results[mask]["layer_fraction"],
                    mask.bit_count(),
                ),
            )

    target, baseline = default_specs()
    feasible_unions: list[dict[str, object]] = []
    all_mask = (1 << len(candidate_groups)) - 1
    for allowed_union in range(all_mask + 1):
        selected_routes: dict[int, int] = {}
        routable = True
        for position in flagged_positions:
            candidates = [
                mask
                for mask in routes_by_position[position]
                if mask & ~allowed_union == 0
            ]
            if not candidates:
                routable = False
                break
            selected_routes[position] = min(
                candidates,
                key=lambda mask: (
                    0.0
                    if mask == 0
                    else subset_results[mask]["layer_fraction"],
                    mask.bit_count(),
                ),
            )
        if not routable:
            continue
        actual_union = 0
        token_layer_sum = 0.0
        route_payload: dict[str, object] = {}
        for position, route_mask in selected_routes.items():
            actual_union |= route_mask
            route_fraction = (
                0.0
                if route_mask == 0
                else float(subset_results[route_mask]["layer_fraction"])
            )
            token_layer_sum += route_fraction
            route_payload[str(position)] = {
                "groups": []
                if route_mask == 0
                else subset_results[route_mask]["groups"],
                "layer_fraction": route_fraction,
                "q6_was_exact": position not in q6_error_positions,
            }
        union_groups = mask_groups(actual_union, candidate_groups)
        union_fraction = sum(group_fractions[group] for group in union_groups)
        mean_token_layer_fraction = token_layer_sum / args.tokens
        budget = token_routed_refinement_budget(
            target=target,
            baseline=baseline,
            block_positions=4096,
            union_layer_fraction=union_fraction,
            mean_token_layer_fraction=mean_token_layer_fraction,
            consensus_bits=6,
            residual_bits=2,
        )
        feasible_unions.append(
            {
                "union_groups": list(union_groups),
                "union_layer_fraction": union_fraction,
                "mean_token_layer_fraction": mean_token_layer_fraction,
                "routes": route_payload,
                "budget": budget.to_dict(),
            }
        )

    best = (
        min(
            feasible_unions,
            key=lambda point: (
                not point["budget"]["ideal_pass"],
                point["budget"]["required_overlap_fraction"],
                point["budget"]["ideal_seconds_per_token"],
                point["union_layer_fraction"],
                point["mean_token_layer_fraction"],
            ),
        )
        if feasible_unions
        else None
    )
    all_errors_routable = all(routes_by_position[position] for position in flagged_positions)
    qualifies = bool(
        best is not None
        and consensus.accepted_error_tokens == 0
        and all_errors_routable
        and best["budget"]["ideal_pass"]
        and best["budget"]["required_overlap_fraction"] <= 0.95
    )

    payload = {
        "evidence_level": "E1 exact-future token-routed layer precision oracle",
        "model": args.model,
        "prompt": args.eval_prompt,
        "tokens": args.tokens,
        "fixed_margin_threshold": args.margin_threshold,
        "candidate_groups": list(candidate_groups),
        "group_layer_fractions": group_fractions,
        "subsets_evaluated": len(subset_results),
        "consensus": consensus.to_dict(),
        "flagged_positions": flagged_positions,
        "q6_error_positions": q6_error_positions,
        "all_flagged_errors_routable": all_errors_routable,
        "feasible_union_count": len(feasible_unions),
        "best_token_routed_point": best,
        "qualifies": qualifies,
        "decision": (
            "advance token-routed precision into causal route prediction"
            if qualifies
            else "reject tested token-routed precision point"
        ),
        "oracle_warning": (
            "Exact future tokens select the route in this experiment. The result "
            "proves only that a resource-compatible route assignment exists; a "
            "deployable runtime must infer routes from causal Q4/Q6 state."
        ),
        "causal_generation_warning": (
            "All measurements use authoritative exact prefixes. This does not "
            "solve candidate-block generation, which remains a separate Gate."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "flagged_positions": flagged_positions,
                "q6_error_positions": q6_error_positions,
                "feasible_union_count": len(feasible_unions),
                "best_token_routed_point": best,
                "qualifies": qualifies,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
