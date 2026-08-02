from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize
from vortex_runtime.residual_sketch_proof import (
    apply_residual_sketch,
    certify_sketch_argmax,
    compile_orthogonal_residual_sketch,
    residual_sketch_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct a Q4 LM head with an orthogonal residual sketch and certify "
            "the remaining exact argmax uncertainty without reading residual values."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_head_residual_sketch_proof.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.tokens, args.hot_bits, args.rank) <= 0:
        raise SystemExit("tokens, hot bits and rank must be positive")

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
    exact_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    hidden = exact_prefix_hidden_states(
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
    sketch = compile_orthogonal_residual_sketch(
        residual,
        rank=args.rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
    )
    compile_seconds = time.perf_counter() - compile_started

    exact_logits = hidden @ exact_weight.T
    hot_logits = hidden @ hot_weight.T
    certified = 0
    unsafe = 0
    refined_top1 = 0
    refined_top4 = 0
    refined_top32 = 0
    hot_top1 = 0
    exact_reference_mismatches = 0
    rows: list[dict[str, int | float | bool]] = []
    for position in range(args.tokens):
        reference_token = int(exact_tokens[0, position].item())
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        hot_rank = exact_token_rank(hot_logits[position], reference_token)
        refined_logits, effects, perpendicular_norm = apply_residual_sketch(
            hot_logits=hot_logits[position],
            activation=hidden[position],
            sketch=sketch,
        )
        refined_rank = exact_token_rank(refined_logits, reference_token)
        certificate = certify_sketch_argmax(
            hot_logits=hot_logits[position],
            activation=hidden[position],
            sketch=sketch,
        )
        safe = (not certificate.certified) or certificate.candidate == exact_argmax
        hot_top1 += int(hot_rank == 1)
        refined_top1 += int(refined_rank == 1)
        refined_top4 += int(refined_rank <= 4)
        refined_top32 += int(refined_rank <= 32)
        certified += int(certificate.certified)
        unsafe += int(not safe)
        exact_reference_mismatches += int(reference_token != exact_argmax)
        rows.append(
            {
                "position": position,
                "reference_token": reference_token,
                "exact_argmax": exact_argmax,
                "hot_exact_token_rank": hot_rank,
                "refined_exact_token_rank": refined_rank,
                "refined_argmax": int(torch.argmax(refined_logits).item()),
                "maximum_remainder_effect_bound": float(effects.max().item()),
                "perpendicular_activation_norm": perpendicular_norm,
                "certified": certificate.certified,
                "certificate_candidate": certificate.candidate,
                "certificate_safe": safe,
                "certified_margin": certificate.certified_margin,
            }
        )

    tiny_budget = residual_sketch_budget(
        rows=exact_weight.shape[0],
        columns=exact_weight.shape[1],
        rank=args.rank,
        metadata_bits=32,
    )
    target_budget = residual_sketch_budget(
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
    )
    payload = {
        "evidence_level": "E2 pretrained orthogonal residual proof sketch",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "rank": args.rank,
        "oversample": args.oversample,
        "power_iterations": args.power_iterations,
        "hot_quantization": quant_stats.to_dict(),
        "sketch": sketch.stats(),
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
            "The residual is decomposed into C U^T plus an orthogonal remainder. "
            "Runtime reads U, C and remainder row norms. The exact remainder "
            "values remain unread, and a token is committed only when its lower "
            "bound exceeds every competitor upper bound."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance orthogonal proof sketches into internal projections"
            if qualifies
            else "increase proof rank or reject tested residual sketch point"
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
