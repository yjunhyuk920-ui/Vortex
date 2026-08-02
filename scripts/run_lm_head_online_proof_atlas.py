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
from scripts.run_lm_head_activation_proof_atlas import prompt_hidden_states
from scripts.run_lm_head_decision_proof import exact_prefix_hidden_states
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.online_activation_proof_atlas import (
    OnlineActivationProofAtlas,
    online_atlas_traffic_budget,
)
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure proof-triggered online residual atlas expansion and charge "
            "each new exact residual image against 405B warm-decode traffic."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--initial-rank", type=int, required=True)
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
        default=Path("lm_head_online_proof_atlas.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.tokens, args.hot_bits, args.initial_rank, args.build_samples) <= 0:
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
    if args.initial_rank > build_hidden.shape[0]:
        raise SystemExit("initial rank exceeds available prompt hidden states")
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
    atlas = OnlineActivationProofAtlas.from_prompt(
        residual=residual,
        build_activations=build_hidden,
        rank=args.initial_rank,
    )
    exact_logits = continuation_hidden @ exact_weight.T
    hot_logits = continuation_hidden @ hot_weight.T

    pre_certified = 0
    post_certified = 0
    unsafe = 0
    expansions = 0
    pre_perpendicular_ratios: list[float] = []
    rows: list[dict[str, int | float | bool]] = []

    for position in range(args.tokens):
        activation = continuation_hidden[position]
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        _, _, _, pre_ratio = atlas.apply(
            hot_logits=hot_logits[position],
            activation=activation,
        )
        pre = atlas.certify(
            hot_logits=hot_logits[position],
            activation=activation,
        )
        pre_safe = (not pre.certified) or pre.candidate == exact_argmax
        pre_certified += int(pre.certified)
        unsafe += int(not pre_safe)
        pre_perpendicular_ratios.append(pre_ratio)

        expanded = False
        if not pre.certified:
            expanded = atlas.expand(
                activation=activation,
                residual=residual,
            )
            expansions += int(expanded)
        post = atlas.certify(
            hot_logits=hot_logits[position],
            activation=activation,
        )
        post_safe = (not post.certified) or post.candidate == exact_argmax
        post_certified += int(post.certified)
        unsafe += int(not post_safe)
        _, _, _, post_ratio = atlas.apply(
            hot_logits=hot_logits[position],
            activation=activation,
        )
        rows.append(
            {
                "position": position,
                "exact_argmax": exact_argmax,
                "pre_certified": pre.certified,
                "pre_candidate": pre.candidate,
                "pre_safe": pre_safe,
                "pre_perpendicular_ratio": pre_ratio,
                "expanded": expanded,
                "post_certified": post.certified,
                "post_candidate": post.candidate,
                "post_safe": post_safe,
                "post_perpendicular_ratio": post_ratio,
                "atlas_rank_after": atlas.rank,
            }
        )

    target_traffic = online_atlas_traffic_budget(
        rows=128_256,
        columns=16_384,
        expansions=expansions,
        tokens=args.tokens,
        source_bits=16,
        hot_bits=args.hot_bits,
    )
    reuse_tokens_per_expansion = (
        args.tokens / expansions if expansions else float("inf")
    )
    pre_rate = pre_certified / args.tokens
    post_rate = post_certified / args.tokens
    qualifies = bool(
        unsafe == 0
        and post_rate == 1.0
        and target_traffic.amortized_residual_gib_per_token <= 0.1
    )
    payload = {
        "evidence_level": "E2 proof-triggered online activation atlas",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "initial_rank": args.initial_rank,
        "final_rank": atlas.rank,
        "hot_quantization": quant_stats.to_dict(),
        "pre_expansion_certificate_rate": pre_rate,
        "post_expansion_certificate_rate": post_rate,
        "unsafe_certificates": unsafe,
        "expansions": expansions,
        "reuse_tokens_per_expansion": reuse_tokens_per_expansion,
        "pre_perpendicular_ratio": {
            "mean": statistics.fmean(pre_perpendicular_ratios),
            "maximum": max(pre_perpendicular_ratios),
        },
        "projected_405b_lm_head_expansion_traffic": target_traffic.to_dict(),
        "positions": rows,
        "contract": (
            "A proof miss may append only the current causally available hidden "
            "direction and its exact residual image. The current token is committed "
            "only after the expanded atlas proves the exact argmax. Every expansion "
            "is charged as one complete LM-head residual stream."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance online proof atlas under measured amortization"
            if qualifies
            else "reject tested online atlas expansion rate"
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
