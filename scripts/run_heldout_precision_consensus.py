from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.candidate_coverage import token_rank, top1_margin
from vortex_runtime.feasibility import default_specs
from vortex_runtime.precision_consensus import (
    PrecisionConsensusRow,
    analyze_precision_consensus,
    progressive_refinement_budget,
    sweep_consensus_thresholds,
)
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pre-registered Q4/Q6 consensus rule on a held-out "
            "prompt using the exact source model only as the label oracle."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--margin-threshold", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    AutoModelForCausalLM, _ = require_transformers()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def greedy_tokens(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    count: int,
) -> torch.Tensor:
    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        output = model(**encoded, use_cache=True, return_dict=True)
        current = torch.argmax(output.logits[:, -1, :], dim=-1).reshape(-1, 1)
        generated.append(current.reshape(-1))
        past = output.past_key_values
        for _ in range(1, count):
            output = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            current = torch.argmax(output.logits[:, -1, :], dim=-1).reshape(-1, 1)
            generated.append(current.reshape(-1))
    return torch.stack(generated, dim=1).to("cpu")


def teacher_forced_logits(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    teacher_tokens = exact_tokens.to(input_ids.device)
    teacher_ids = (
        torch.cat((input_ids, teacher_tokens[:, :-1]), dim=1)
        if teacher_tokens.shape[1] > 1
        else input_ids
    )
    kwargs: dict[str, Any] = {
        "input_ids": teacher_ids,
        "use_cache": False,
        "return_dict": True,
    }
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        extension = torch.ones(
            (attention_mask.shape[0], teacher_tokens.shape[1] - 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        kwargs["attention_mask"] = torch.cat((attention_mask, extension), dim=1)
    with torch.inference_mode():
        output = model(**kwargs)
    start = input_ids.shape[1] - 1
    end = start + teacher_tokens.shape[1]
    return output.logits[:, start:end, :].detach().to("cpu", torch.float32)


def precision_rows(
    *,
    model_name: str,
    bits: int,
    device: torch.device,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
    row_chunk: int,
) -> tuple[list[dict[str, int | float | bool]], dict[str, int | float]]:
    model = load_model(model_name, device)
    precision, _ = fake_quantize_full_rank_modules(
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
    rows: list[dict[str, int | float | bool]] = []
    for position in range(exact_tokens.shape[1]):
        exact_token = int(exact_tokens[0, position].item())
        position_logits = logits[0, position]
        predicted = int(torch.argmax(position_logits).item())
        rows.append(
            {
                "position": position,
                "exact_token": exact_token,
                "predicted_token": predicted,
                "exact_match": predicted == exact_token,
                "exact_token_rank": token_rank(position_logits, exact_token),
                "top1_margin": top1_margin(position_logits),
                "exact_gap_from_top1": float(
                    (position_logits[predicted] - position_logits[exact_token]).item()
                ),
            }
        )
    del logits, model
    gc.collect()
    return rows, precision.to_dict()


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.row_chunk <= 0:
        raise SystemExit("tokens and row chunk must be positive")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit("prompt index is outside the prompt set")
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

    q4_rows, q4_precision = precision_rows(
        model_name=args.model,
        bits=4,
        device=device,
        encoded=encoded,
        exact_tokens=exact_tokens,
        row_chunk=args.row_chunk,
    )
    q6_rows, q6_precision = precision_rows(
        model_name=args.model,
        bits=6,
        device=device,
        encoded=encoded,
        exact_tokens=exact_tokens,
        row_chunk=args.row_chunk,
    )

    consensus_rows: list[PrecisionConsensusRow] = []
    joined_rows: list[dict[str, int | float | bool]] = []
    for q4, q6 in zip(q4_rows, q6_rows, strict=True):
        if q4["position"] != q6["position"] or q4["exact_token"] != q6["exact_token"]:
            raise RuntimeError("precision rows are not aligned")
        row = PrecisionConsensusRow(
            position=int(q4["position"]),
            exact_token=int(q4["exact_token"]),
            q4_token=int(q4["predicted_token"]),
            q6_token=int(q6["predicted_token"]),
            q6_margin=float(q6["top1_margin"]),
        )
        consensus_rows.append(row)
        accepted = row.agrees and row.q6_margin >= args.margin_threshold
        joined_rows.append(
            {
                "position": row.position,
                "exact_token": row.exact_token,
                "q4_token": row.q4_token,
                "q6_token": row.q6_token,
                "q4_exact": row.q4_exact,
                "q6_exact": row.q6_exact,
                "agrees": row.agrees,
                "q4_exact_rank": int(q4["exact_token_rank"]),
                "q6_exact_rank": int(q6["exact_token_rank"]),
                "q4_margin": float(q4["top1_margin"]),
                "q6_margin": row.q6_margin,
                "accepted_by_fixed_rule": accepted,
                "fixed_rule_exact": accepted and row.q6_exact,
            }
        )

    fixed_report = analyze_precision_consensus(
        consensus_rows,
        margin_threshold=args.margin_threshold,
    )
    threshold_reports = sweep_consensus_thresholds(
        consensus_rows,
        thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0),
    )
    target, baseline = default_specs()
    budgets = {
        str(layer_fraction): progressive_refinement_budget(
            target=target,
            baseline=baseline,
            block_positions=4096,
            refinement_fraction=fixed_report.refinement_fraction,
            refined_layer_fraction=layer_fraction,
        ).to_dict()
        for layer_fraction in (1.0, 0.5, 0.25, 0.125, 0.0625)
    }

    payload = {
        "evidence_level": "E2 held-out multi-precision consensus point",
        "model": args.model,
        "prompt_index": args.prompt_index,
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "evaluated_tokens": args.tokens,
        "fixed_margin_threshold": args.margin_threshold,
        "q4_precision": q4_precision,
        "q6_precision": q6_precision,
        "fixed_rule": fixed_report.to_dict(),
        "threshold_diagnostics": [report.to_dict() for report in threshold_reports],
        "refinement_budgets_at_observed_fraction": budgets,
        "rows": joined_rows,
        "pre_registration": (
            "The 0.4 margin threshold was fixed from the earlier development "
            "prompt before this held-out prompt set was evaluated."
        ),
        "decision": (
            "survive held-out point"
            if fixed_report.all_exact_errors_flagged
            else "reject fixed consensus rule on this held-out point"
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
                "tokens": fixed_report.tokens,
                "accepted": fixed_report.accepted_tokens,
                "accepted_errors": fixed_report.accepted_error_tokens,
                "refinement_fraction": fixed_report.refinement_fraction,
                "q4_errors": fixed_report.q4_errors,
                "q6_errors": fixed_report.q6_errors,
                "all_errors_flagged": fixed_report.all_exact_errors_flagged,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
