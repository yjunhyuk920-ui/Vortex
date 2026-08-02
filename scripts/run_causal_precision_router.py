from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_layer_local_precision_oracle import (
    apply_group_precision,
    encode_prompt,
    greedy_tokens,
    load_model,
    require_transformers,
    teacher_forced_logits,
)
from vortex_runtime.candidate_coverage import top1_margin
from vortex_runtime.causal_precision_router import (
    PrecisionStageObservation,
    select_stable_precision_stage,
)
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

STAGE_GROUPS = (
    "layers_002_003",
    "layers_010_011",
    "io",
    "layers_006_007",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pre-registered label-free Q4/Q6 consensus and staged "
            "Q6-to-Q8 residual router on a second held-out prompt set."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--layers-per-group", type=int, default=2)
    parser.add_argument("--margin-threshold", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantize_model(
    *,
    model_name: str,
    bits: int,
    device: torch.device,
    row_chunk: int,
) -> torch.nn.Module:
    model = load_model(model_name, device)
    fake_quantize_full_rank_modules(
        model,
        bits=bits,
        source_bits=16,
        row_chunk=row_chunk,
    )
    return model


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit("prompt index is outside the prompt set")
    if args.tokens <= 0 or args.row_chunk <= 0 or args.layers_per_group <= 0:
        raise SystemExit("tokens, row chunk, and layer grouping must be positive")
    prompt = prompts[args.prompt_index]

    _, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    encoded = encode_prompt(tokenizer, str(prompt["text"]), device)
    started = time.perf_counter()

    exact_model = load_model(args.model, device)
    exact_tokens = greedy_tokens(
        model=exact_model,
        encoded=encoded,
        count=args.tokens,
    )

    q4_model = quantize_model(
        model_name=args.model,
        bits=4,
        device=device,
        row_chunk=args.row_chunk,
    )
    q4_logits = teacher_forced_logits(
        model=q4_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    del q4_model
    gc.collect()

    q6_model = quantize_model(
        model_name=args.model,
        bits=6,
        device=device,
        row_chunk=args.row_chunk,
    )
    q6_logits = teacher_forced_logits(
        model=q6_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )

    consensus_rows: list[PrecisionConsensusRow] = []
    for position in range(args.tokens):
        q4_position = q4_logits[0, position]
        q6_position = q6_logits[0, position]
        consensus_rows.append(
            PrecisionConsensusRow(
                position=position,
                exact_token=int(exact_tokens[0, position].item()),
                q4_token=int(torch.argmax(q4_position).item()),
                q6_token=int(torch.argmax(q6_position).item()),
                q6_margin=top1_margin(q6_position),
            )
        )
    consensus = analyze_precision_consensus(
        consensus_rows,
        margin_threshold=args.margin_threshold,
    )
    initially_accepted = {
        row.position
        for row in consensus_rows
        if row.agrees and row.q6_margin >= args.margin_threshold
    }
    flagged_positions = [
        position for position in range(args.tokens) if position not in initially_accepted
    ]

    groups = precision_module_groups(
        q6_model,
        layers_per_group=args.layers_per_group,
    )
    missing = [group for group in STAGE_GROUPS if group not in groups]
    if missing:
        raise RuntimeError(f"pre-registered stage groups are missing: {missing}")
    total_elements = unique_weight_elements(q6_model)
    group_fractions = {
        group: unique_weight_elements(q6_model, groups[group]) / total_elements
        for group in STAGE_GROUPS
    }

    stage_logits = [q6_logits]
    cumulative_fractions = [0.0]
    cumulative = 0.0
    for group in STAGE_GROUPS:
        apply_group_precision(
            target_model=q6_model,
            source_model=exact_model,
            module_names=groups[group],
            bits=8,
            row_chunk=args.row_chunk,
            keep_applied=True,
        )
        cumulative += group_fractions[group]
        cumulative_fractions.append(cumulative)
        stage_logits.append(
            teacher_forced_logits(
                model=q6_model,
                encoded=encoded,
                exact_tokens=exact_tokens,
            )
        )

    unsafe_positions: list[int] = []
    fallback_positions: list[int] = []
    selected_stage_counts = {str(stage): 0 for stage in range(len(stage_logits))}
    decisions: list[dict[str, object]] = []
    maximum_selected_stage = 0
    token_layer_sum = 0.0

    for position, row in enumerate(consensus_rows):
        exact_token = row.exact_token
        if position in initially_accepted:
            selected_stage_counts["0"] += 1
            if row.q6_token != exact_token:
                unsafe_positions.append(position)
            decisions.append(
                {
                    "position": position,
                    "mode": "initial_consensus",
                    "selected_stage": 0,
                    "selected_token": row.q6_token,
                    "exact_token": exact_token,
                    "exact": row.q6_token == exact_token,
                    "cumulative_layer_fraction": 0.0,
                }
            )
            continue

        observations = []
        for stage, logits in enumerate(stage_logits):
            position_logits = logits[0, position]
            observations.append(
                PrecisionStageObservation(
                    stage=stage,
                    token=int(torch.argmax(position_logits).item()),
                    margin=top1_margin(position_logits),
                    cumulative_layer_fraction=cumulative_fractions[stage],
                )
            )
        route = select_stable_precision_stage(
            observations,
            margin_threshold=args.margin_threshold,
        )
        if not route.accepted:
            fallback_positions.append(position)
            maximum_selected_stage = max(maximum_selected_stage, len(STAGE_GROUPS))
            token_layer_sum += cumulative_fractions[-1]
            decisions.append(
                {
                    "position": position,
                    "mode": "exact_fallback",
                    "selected_stage": None,
                    "selected_token": None,
                    "exact_token": exact_token,
                    "exact": True,
                    "cumulative_layer_fraction": cumulative_fractions[-1],
                    "observations": [item.to_dict() for item in observations],
                }
            )
            continue

        selected_stage = int(route.selected_stage)
        selected_token = int(route.selected_token)
        maximum_selected_stage = max(maximum_selected_stage, selected_stage)
        selected_stage_counts[str(selected_stage)] += 1
        token_layer_sum += route.cumulative_layer_fraction
        if selected_token != exact_token:
            unsafe_positions.append(position)
        decisions.append(
            {
                "position": position,
                "mode": "staged_residual",
                "selected_stage": selected_stage,
                "selected_token": selected_token,
                "exact_token": exact_token,
                "exact": selected_token == exact_token,
                "cumulative_layer_fraction": route.cumulative_layer_fraction,
                "observations": [item.to_dict() for item in observations],
            }
        )

    union_groups = list(STAGE_GROUPS[:maximum_selected_stage])
    union_fraction = sum(group_fractions[group] for group in union_groups)
    mean_token_layer_fraction = token_layer_sum / args.tokens
    target, baseline = default_specs()
    budget = token_routed_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        union_layer_fraction=union_fraction,
        mean_token_layer_fraction=mean_token_layer_fraction,
        consensus_bits=6,
        residual_bits=2,
    )
    qualifies = bool(
        not unsafe_positions
        and not fallback_positions
        and budget.ideal_pass
        and budget.required_overlap_fraction <= 0.95
    )

    payload = {
        "evidence_level": "E2 second-held-out causal residual route point",
        "model": args.model,
        "prompt_index": args.prompt_index,
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "evaluated_tokens": args.tokens,
        "pre_registered": {
            "margin_threshold": args.margin_threshold,
            "stage_groups": list(STAGE_GROUPS),
            "stop_rule": (
                "accept the earliest adjacent precision stages with identical "
                "top-1 and later-stage margin >= 0.4; otherwise exact fallback"
            ),
        },
        "initial_consensus": consensus.to_dict(),
        "flagged_positions": flagged_positions,
        "group_layer_fractions": group_fractions,
        "selected_stage_counts": selected_stage_counts,
        "maximum_selected_stage": maximum_selected_stage,
        "union_groups": union_groups,
        "union_layer_fraction": union_fraction,
        "mean_token_layer_fraction": mean_token_layer_fraction,
        "unsafe_positions": unsafe_positions,
        "fallback_positions": fallback_positions,
        "budget": budget.to_dict(),
        "decisions": decisions,
        "qualifies": qualifies,
        "decision": (
            "survive causal route point"
            if qualifies
            else "reject fixed causal route point"
        ),
        "exact_prefix_warning": (
            "The selector itself uses no future labels, but logits are evaluated "
            "on authoritative exact prefixes. Candidate-block generation remains "
            "a separate unsolved Gate."
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
                "prompt_id": prompt["id"],
                "unsafe_positions": unsafe_positions,
                "fallback_positions": fallback_positions,
                "selected_stage_counts": selected_stage_counts,
                "mean_token_layer_fraction": mean_token_layer_fraction,
                "required_overlap_fraction": budget.required_overlap_fraction,
                "qualifies": qualifies,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
