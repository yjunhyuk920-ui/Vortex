from __future__ import annotations

import argparse
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
from vortex_runtime.feasibility import default_specs
from vortex_runtime.layer_precision_oracle import (
    precision_module_groups,
    unique_weight_elements,
)
from vortex_runtime.precision_consensus import progressive_refinement_budget
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules

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
            "Exhaustively search all subsets of the top Q6-to-Q8 layer groups "
            "identified by the first exact-future oracle."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--layers-per-group", type=int, default=2)
    parser.add_argument("--runtime-refinement-fraction", type=float, default=0.25)
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


def main() -> None:
    args = parse_args()
    candidate_groups = tuple(
        item.strip() for item in args.candidate_groups.split(",") if item.strip()
    )
    if not candidate_groups:
        raise SystemExit("at least one candidate group is required")
    if len(candidate_groups) > 12:
        raise SystemExit("refusing an unbounded exhaustive search")

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
    base_model = load_model(args.model, device)
    fake_quantize_full_rank_modules(
        base_model,
        bits=6,
        source_bits=16,
        row_chunk=args.row_chunk,
    )
    base_logits = teacher_forced_logits(
        model=base_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    base_errors = error_positions(base_logits, exact_tokens)

    groups = precision_module_groups(
        base_model,
        layers_per_group=args.layers_per_group,
    )
    missing = [group for group in candidate_groups if group not in groups]
    if missing:
        raise RuntimeError(f"candidate groups not present: {missing}")
    total_elements = unique_weight_elements(base_model)
    group_elements = {
        group: unique_weight_elements(base_model, groups[group])
        for group in candidate_groups
    }
    target, baseline = default_specs()

    evaluated: list[dict[str, object]] = []
    exact_subsets: list[dict[str, object]] = []
    best_partial: dict[str, object] | None = None
    for size in range(1, len(candidate_groups) + 1):
        for subset in combinations(candidate_groups, size):
            backups_by_group: list[dict[str, torch.Tensor]] = []
            for group in subset:
                backups, _ = apply_group_precision(
                    target_model=base_model,
                    source_model=exact_model,
                    module_names=groups[group],
                    bits=8,
                    row_chunk=args.row_chunk,
                    keep_applied=False,
                )
                backups_by_group.append(backups)
            logits = teacher_forced_logits(
                model=base_model,
                encoded=encoded,
                exact_tokens=exact_tokens,
            )
            errors = error_positions(logits, exact_tokens)
            elements = sum(group_elements[group] for group in subset)
            layer_fraction = elements / total_elements
            budget = progressive_refinement_budget(
                target=target,
                baseline=baseline,
                block_positions=4096,
                refinement_fraction=args.runtime_refinement_fraction,
                refined_layer_fraction=layer_fraction,
                consensus_bits=6,
                residual_bits=2,
                consensus_effective_tops=120.0,
                residual_effective_tops=320.0,
            )
            point = {
                "groups": list(subset),
                "group_count": size,
                "layer_fraction": layer_fraction,
                "exact_top1_tokens": args.tokens - len(errors),
                "exact_top1_rate": (args.tokens - len(errors)) / args.tokens,
                "remaining_error_positions": errors,
                "budget": budget.to_dict(),
            }
            evaluated.append(point)
            if not errors:
                exact_subsets.append(point)
            if best_partial is None or (
                point["exact_top1_tokens"],
                -point["layer_fraction"],
            ) > (
                best_partial["exact_top1_tokens"],
                -best_partial["layer_fraction"],
            ):
                best_partial = point
            for backups in reversed(backups_by_group):
                restore_group(base_model, backups)
            del logits

    minimum_exact = (
        min(
            exact_subsets,
            key=lambda point: (
                point["layer_fraction"],
                point["group_count"],
                point["budget"]["required_overlap_fraction"],
            ),
        )
        if exact_subsets
        else None
    )
    qualifies = bool(
        minimum_exact is not None
        and minimum_exact["budget"]["ideal_pass"]
        and minimum_exact["budget"]["required_overlap_fraction"] <= 0.95
    )
    payload = {
        "evidence_level": "E1 exhaustive exact-future layer-subset oracle",
        "model": args.model,
        "prompt": args.eval_prompt,
        "tokens": args.tokens,
        "base_error_positions": base_errors,
        "candidate_groups": list(candidate_groups),
        "subsets_evaluated": len(evaluated),
        "exact_subsets": exact_subsets,
        "minimum_exact_subset": minimum_exact,
        "best_partial_subset": best_partial,
        "qualifies": qualifies,
        "decision": (
            "advance minimum layer subset to causal routing"
            if qualifies
            else "reject tested exhaustive layer-subset point"
        ),
        "oracle_warning": (
            "Candidate groups were selected using exact future labels in the "
            "previous oracle. This experiment tests existence and resource "
            "bounds only, not a deployable selector."
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
                "subsets_evaluated": len(evaluated),
                "exact_subset_count": len(exact_subsets),
                "minimum_exact_subset": minimum_exact,
                "qualifies": qualifies,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
