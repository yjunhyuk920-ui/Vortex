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
    encode_prompt,
    greedy_tokens,
    load_model,
    require_transformers,
    teacher_forced_logits,
)
from vortex_runtime.candidate_coverage import top1_margin
from vortex_runtime.feasibility import default_specs
from vortex_runtime.nested_precision import (
    fake_quantize_nested_modules,
    nested_bitplane_budget,
)
from vortex_runtime.precision_consensus import (
    PrecisionConsensusRow,
    analyze_precision_consensus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pre-registered nested Q6-Q7-Q8 full-rank bitplane "
            "ladder on a third held-out prompt set."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--margin-threshold", type=float, default=0.4)
    parser.add_argument("--bitplane-effective-tops", type=float, default=640.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested_logits(
    *,
    model_name: str,
    bits: int,
    device: torch.device,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
    row_chunk: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    model = load_model(model_name, device)
    precision, _ = fake_quantize_nested_modules(
        model,
        bits=bits,
        maximum_bits=8,
        source_bits=16,
        row_chunk=row_chunk,
    )
    logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    del model
    gc.collect()
    return logits, precision.to_dict()


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit("prompt index is outside the prompt set")
    if args.tokens <= 0 or args.row_chunk <= 0:
        raise SystemExit("tokens and row chunk must be positive")
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
    del exact_model
    gc.collect()

    logits_by_bits: dict[int, torch.Tensor] = {}
    precision_by_bits: dict[str, dict[str, int | float]] = {}
    for bits in (4, 6, 7, 8):
        logits, precision = nested_logits(
            model_name=args.model,
            bits=bits,
            device=device,
            encoded=encoded,
            exact_tokens=exact_tokens,
            row_chunk=args.row_chunk,
        )
        logits_by_bits[bits] = logits
        precision_by_bits[str(bits)] = precision

    consensus_rows: list[PrecisionConsensusRow] = []
    for position in range(args.tokens):
        exact_token = int(exact_tokens[0, position].item())
        q4 = logits_by_bits[4][0, position]
        q6 = logits_by_bits[6][0, position]
        consensus_rows.append(
            PrecisionConsensusRow(
                position=position,
                exact_token=exact_token,
                q4_token=int(torch.argmax(q4).item()),
                q6_token=int(torch.argmax(q6).item()),
                q6_margin=top1_margin(q6),
            )
        )
    consensus = analyze_precision_consensus(
        consensus_rows,
        margin_threshold=args.margin_threshold,
    )
    initial_accepts = {
        row.position
        for row in consensus_rows
        if row.agrees and row.q6_margin >= args.margin_threshold
    }

    unsafe_positions: list[int] = []
    fallback_positions: list[int] = []
    ladder_accept_positions: list[int] = []
    q8_error_positions: list[int] = []
    decisions: list[dict[str, object]] = []
    for position, row in enumerate(consensus_rows):
        exact_token = row.exact_token
        q6_logits = logits_by_bits[6][0, position]
        q7_logits = logits_by_bits[7][0, position]
        q8_logits = logits_by_bits[8][0, position]
        q6_token = int(torch.argmax(q6_logits).item())
        q7_token = int(torch.argmax(q7_logits).item())
        q8_token = int(torch.argmax(q8_logits).item())
        q8_margin = top1_margin(q8_logits)
        if q8_token != exact_token:
            q8_error_positions.append(position)

        if position in initial_accepts:
            selected_token = row.q6_token
            exact = selected_token == exact_token
            if not exact:
                unsafe_positions.append(position)
            decisions.append(
                {
                    "position": position,
                    "mode": "initial_q4_q6_consensus",
                    "selected_token": selected_token,
                    "exact_token": exact_token,
                    "exact": exact,
                    "q6_token": q6_token,
                    "q7_token": q7_token,
                    "q8_token": q8_token,
                    "q8_margin": q8_margin,
                }
            )
            continue

        ladder_accept = (
            q6_token == q7_token == q8_token
            and q8_margin >= args.margin_threshold
        )
        if ladder_accept:
            ladder_accept_positions.append(position)
            exact = q8_token == exact_token
            if not exact:
                unsafe_positions.append(position)
            decisions.append(
                {
                    "position": position,
                    "mode": "q6_q7_q8_triple_consensus",
                    "selected_token": q8_token,
                    "exact_token": exact_token,
                    "exact": exact,
                    "q6_token": q6_token,
                    "q7_token": q7_token,
                    "q8_token": q8_token,
                    "q6_margin": top1_margin(q6_logits),
                    "q7_margin": top1_margin(q7_logits),
                    "q8_margin": q8_margin,
                }
            )
        else:
            fallback_positions.append(position)
            decisions.append(
                {
                    "position": position,
                    "mode": "exact_fallback",
                    "selected_token": None,
                    "exact_token": exact_token,
                    "exact": True,
                    "q6_token": q6_token,
                    "q7_token": q7_token,
                    "q8_token": q8_token,
                    "q6_margin": top1_margin(q6_logits),
                    "q7_margin": top1_margin(q7_logits),
                    "q8_margin": q8_margin,
                }
            )

    flagged_fraction = (args.tokens - len(initial_accepts)) / args.tokens
    target, baseline = default_specs()
    budget = nested_bitplane_budget(
        target=target,
        baseline=baseline,
        base_bits=6,
        maximum_bits=8,
        block_positions=4096,
        fractions_reaching_bits={
            7: flagged_fraction,
            8: flagged_fraction,
        },
        base_effective_tops=120.0,
        bitplane_effective_tops=args.bitplane_effective_tops,
    )
    qualifies = bool(
        not unsafe_positions
        and not fallback_positions
        and not q8_error_positions
        and budget.ideal_pass
        and budget.required_overlap_fraction <= 0.95
    )

    payload = {
        "evidence_level": "E2 third-held-out nested bitplane ladder point",
        "model": args.model,
        "prompt_index": args.prompt_index,
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "evaluated_tokens": args.tokens,
        "pre_registered": {
            "initial_rule": "nested Q4 == nested Q6 and Q6 margin >= 0.4",
            "ladder_rule": (
                "for initially flagged tokens accept only when nested Q6, Q7, "
                "and Q8 top-1 all agree and Q8 margin >= 0.4"
            ),
            "maximum_nested_bits": 8,
            "bitplane_effective_tops": args.bitplane_effective_tops,
        },
        "precision": precision_by_bits,
        "initial_consensus": consensus.to_dict(),
        "initial_accept_positions": sorted(initial_accepts),
        "flagged_fraction": flagged_fraction,
        "ladder_accept_positions": ladder_accept_positions,
        "unsafe_positions": unsafe_positions,
        "fallback_positions": fallback_positions,
        "q8_error_positions": q8_error_positions,
        "budget": budget.to_dict(),
        "decisions": decisions,
        "qualifies": qualifies,
        "decision": (
            "survive nested bitplane ladder point"
            if qualifies
            else "reject tested nested bitplane ladder point"
        ),
        "exact_prefix_warning": (
            "The rules use no future labels, but all precision logits are "
            "evaluated on authoritative exact prefixes. Candidate-block "
            "generation remains a separate unsolved Gate."
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
                "flagged_fraction": flagged_fraction,
                "unsafe_positions": unsafe_positions,
                "fallback_positions": fallback_positions,
                "q8_error_positions": q8_error_positions,
                "required_overlap_fraction": budget.required_overlap_fraction,
                "ideal_ratio_to_native_4b": (
                    budget.ideal_seconds_per_token
                    / budget.baseline_seconds_per_token
                ),
                "qualifies": qualifies,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
