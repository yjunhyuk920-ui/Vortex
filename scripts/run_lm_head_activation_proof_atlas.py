from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_lm_head_decision_proof import (
    exact_prefix_hidden_states,
    exact_token_rank,
)
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.activation_proof_atlas import (
    activation_proof_atlas_budget,
    apply_activation_proof_atlas,
    certify_activation_atlas_argmax,
    compile_activation_proof_atlas,
)
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Q4 LM-head residual proof atlas from causally available prompt "
            "hidden states and certify exact continuation decisions."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--build-samples", type=int, default=32)
    parser.add_argument(
        "--eval-prompt",
        default=(
            "You are writing a rigorous technical note for software engineers. "
            "Explain stable sorting, compare merge sort with quicksort, discuss "
            "time and memory complexity, give a concrete example, and finish "
            "with practical advice for choosing an algorithm in production systems."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_head_activation_proof_atlas.json"),
    )
    return parser.parse_args()


def prompt_hidden_states(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    samples: int,
) -> torch.Tensor:
    kwargs: dict[str, object] = {
        "input_ids": encoded["input_ids"],
        "use_cache": False,
        "return_dict": True,
    }
    if "attention_mask" in encoded:
        kwargs["attention_mask"] = encoded["attention_mask"]
    with torch.inference_mode():
        output = model.model(**kwargs)
    hidden = output.last_hidden_state[0].detach().to("cpu", torch.float32)
    count = min(samples, hidden.shape[0])
    if count <= 0:
        raise RuntimeError("prompt produced no hidden states")
    return hidden[-count:]


def main() -> None:
    args = parse_args()
    if min(args.tokens, args.hot_bits, args.rank, args.build_samples) <= 0:
        raise SystemExit("tokens, precision, rank and build samples must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)

    started = time.perf_counter()
    build_hidden = prompt_hidden_states(
        model=model,
        encoded=encoded,
        samples=args.build_samples,
    )
    if args.rank > build_hidden.shape[0]:
        raise SystemExit(
            f"rank {args.rank} exceeds available prompt samples {build_hidden.shape[0]}"
        )

    exact_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    continuation_hidden = exact_prefix_hidden_states(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )[0]

    exact_weight = model.lm_head.weight.detach().to("cpu", torch.float32)
    hot_weight, quant_stats = symmetric_per_row_fake_quantize(
        exact_weight,
        bits=args.hot_bits,
        source_bits=16,
        name="lm_head.weight",
        row_chunk=128,
    )
    residual = exact_weight - hot_weight
    compile_started = time.perf_counter()
    atlas = compile_activation_proof_atlas(
        residual=residual,
        build_activations=build_hidden,
        rank=args.rank,
    )
    compile_seconds = time.perf_counter() - compile_started

    exact_logits = continuation_hidden @ exact_weight.T
    hot_logits = continuation_hidden @ hot_weight.T
    certified = 0
    unsafe = 0
    hot_top1 = 0
    refined_top1 = 0
    refined_top4 = 0
    refined_top32 = 0
    exact_reference_mismatches = 0
    perpendicular_ratios: list[float] = []
    rows: list[dict[str, int | float | bool]] = []

    for position in range(args.tokens):
        reference_token = int(exact_tokens[0, position].item())
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        hot_rank = exact_token_rank(hot_logits[position], reference_token)
        refined_logits, effects, perpendicular_norm, perpendicular_ratio = (
            apply_activation_proof_atlas(
                hot_logits=hot_logits[position],
                activation=continuation_hidden[position],
                atlas=atlas,
            )
        )
        refined_rank = exact_token_rank(refined_logits, reference_token)
        certificate = certify_activation_atlas_argmax(
            hot_logits=hot_logits[position],
            activation=continuation_hidden[position],
            atlas=atlas,
        )
        safe = (not certificate.certified) or certificate.candidate == exact_argmax
        hot_top1 += int(hot_rank == 1)
        refined_top1 += int(refined_rank == 1)
        refined_top4 += int(refined_rank <= 4)
        refined_top32 += int(refined_rank <= 32)
        certified += int(certificate.certified)
        unsafe += int(not safe)
        exact_reference_mismatches += int(reference_token != exact_argmax)
        perpendicular_ratios.append(perpendicular_ratio)
        rows.append(
            {
                "position": position,
                "reference_token": reference_token,
                "exact_argmax": exact_argmax,
                "hot_exact_token_rank": hot_rank,
                "refined_exact_token_rank": refined_rank,
                "refined_argmax": int(torch.argmax(refined_logits).item()),
                "perpendicular_norm": perpendicular_norm,
                "perpendicular_ratio": perpendicular_ratio,
                "maximum_remainder_effect_bound": float(effects.max().item()),
                "certified": certificate.certified,
                "certificate_candidate": certificate.candidate,
                "certificate_safe": safe,
                "certified_margin": certificate.certified_margin,
            }
        )

    tiny_budget = activation_proof_atlas_budget(
        rows=exact_weight.shape[0],
        columns=exact_weight.shape[1],
        rank=args.rank,
        metadata_bits=32,
    )
    target_budget = activation_proof_atlas_budget(
        rows=128_256,
        columns=16_384,
        rank=args.rank,
        metadata_bits=32,
    )
    certificate_rate = certified / args.tokens
    qualifies = bool(
        unsafe == 0
        and exact_reference_mismatches == 0
        and certificate_rate >= 0.5
        and target_budget.metadata_gib < 0.05
    )
    payload = {
        "evidence_level": "E2 causal prompt activation proof atlas",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "build_samples": int(build_hidden.shape[0]),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "rank": args.rank,
        "hot_quantization": quant_stats.to_dict(),
        "atlas": atlas.stats(),
        "continuation_perpendicular_ratio": {
            "mean": statistics.fmean(perpendicular_ratios),
            "maximum": max(perpendicular_ratios),
        },
        "hot_top1_rate": hot_top1 / args.tokens,
        "refined_top1_rate": refined_top1 / args.tokens,
        "refined_top4_rate": refined_top4 / args.tokens,
        "refined_top32_rate": refined_top32 / args.tokens,
        "certificate_rate": certificate_rate,
        "certified_tokens": certified,
        "unsafe_certificates": unsafe,
        "exact_reference_mismatches": exact_reference_mismatches,
        "tiny_metadata_budget": tiny_budget.to_dict(),
        "projected_405b_lm_head_metadata_budget": target_budget.to_dict(),
        "compile_seconds": compile_seconds,
        "positions": rows,
        "contract": (
            "The residual atlas is built only from prompt-prefill hidden states. "
            "Residual correction is exact on their span. Continuation components "
            "outside that span are bounded conservatively, and no token is committed "
            "unless the exact argmax is proven."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance causal proof atlas into internal projections"
            if qualifies
            else "expand causal activation coverage or reject tested atlas rank"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
