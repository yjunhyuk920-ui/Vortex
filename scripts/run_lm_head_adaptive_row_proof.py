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
from scripts.run_lm_head_decision_proof import exact_prefix_hidden_states
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.adaptive_row_proof import (
    adaptive_row_proof_budget,
    certify_with_adaptive_exact_rows,
)
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize
from vortex_runtime.residual_proof import (
    residual_metadata_budget,
    rowwise_residual_block_norms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptively refine ambiguous Q4 LM-head rows until the exact global "
            "argmax is proven or a residual-row budget is exhausted."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--column-block", type=int, required=True)
    parser.add_argument("--initial-top-k", type=int, default=32)
    parser.add_argument("--refinement-batch", type=int, default=32)
    parser.add_argument("--max-refined-rows", type=int, default=4096)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_head_adaptive_row_proof.json"),
    )
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    controls = (
        args.tokens,
        args.hot_bits,
        args.column_block,
        args.initial_top_k,
        args.refinement_batch,
        args.max_refined_rows,
    )
    if min(controls) <= 0:
        raise SystemExit("all controls must be positive")

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
    residual_norms = rowwise_residual_block_norms(
        residual,
        column_block=args.column_block,
    )
    exact_logits = hidden @ exact_weight.T
    hot_logits = hidden @ hot_weight.T

    rows: list[dict[str, int | float | bool]] = []
    refined_counts: list[int] = []
    iterations: list[int] = []
    certified = 0
    unsafe = 0
    for position in range(args.tokens):
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        certificate = certify_with_adaptive_exact_rows(
            hot_logits=hot_logits[position],
            activation=hidden[position],
            residual=residual,
            residual_norms=residual_norms,
            column_block=args.column_block,
            initial_top_k=args.initial_top_k,
            refinement_batch=args.refinement_batch,
            max_refined_rows=min(args.max_refined_rows, exact_weight.shape[0]),
        )
        safe = (not certificate.certified) or certificate.candidate == exact_argmax
        certified += int(certificate.certified)
        unsafe += int(not safe)
        refined_counts.append(certificate.refined_row_count)
        iterations.append(certificate.iterations)
        rows.append(
            {
                "position": position,
                "exact_argmax": exact_argmax,
                "candidate": certificate.candidate,
                "certified": certificate.certified,
                "safe": safe,
                "refined_rows": certificate.refined_row_count,
                "iterations": certificate.iterations,
                "ambiguous_rows_remaining": certificate.ambiguous_rows_remaining,
                "certified_margin": certificate.certified_margin,
            }
        )

    mean_rows = statistics.fmean(refined_counts)
    maximum_rows = max(refined_counts)
    p95_rows = percentile(refined_counts, 0.95)
    projected_mean = adaptive_row_proof_budget(
        columns=16_384,
        refined_rows=max(1, round(mean_rows)),
        source_bits=16,
        hot_bits=args.hot_bits,
    )
    projected_p95 = adaptive_row_proof_budget(
        columns=16_384,
        refined_rows=max(1, round(p95_rows)),
        source_bits=16,
        hot_bits=args.hot_bits,
    )
    target_metadata = residual_metadata_budget(
        rows=128_256,
        columns=16_384,
        column_block=args.column_block,
        metadata_bits=16,
    )
    certificate_rate = certified / args.tokens
    qualifies = bool(
        unsafe == 0
        and certificate_rate >= 0.95
        and projected_p95.residual_gib <= 0.1
    )
    payload = {
        "evidence_level": "E2 pretrained adaptive exact-row proof",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "column_block": args.column_block,
        "initial_top_k": args.initial_top_k,
        "refinement_batch": args.refinement_batch,
        "max_refined_rows": args.max_refined_rows,
        "hot_quantization": quant_stats.to_dict(),
        "certificate_rate": certificate_rate,
        "certified_tokens": certified,
        "unsafe_certificates": unsafe,
        "refined_rows": {
            "mean": mean_rows,
            "p95": p95_rows,
            "maximum": maximum_rows,
            "mean_iterations": statistics.fmean(iterations),
        },
        "projected_405b_mean_residual_budget": projected_mean.to_dict(),
        "projected_405b_p95_residual_budget": projected_p95.to_dict(),
        "projected_405b_metadata_budget": target_metadata.to_dict(),
        "positions": rows,
        "contract": (
            "Only rows whose conservative upper bounds can still beat the current "
            "exact winner are refined. Certification occurs only when every unread "
            "row is proven below the best exact refined logit."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance adaptive row proof into internal residual tiles"
            if qualifies
            else "tighten bounds or reduce ambiguous-row count before promotion"
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
