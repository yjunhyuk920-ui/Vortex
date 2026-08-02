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
from vortex_runtime.residual_proof import (
    residual_metadata_budget,
    rowwise_residual_block_norms,
)
from vortex_runtime.topk_row_proof import (
    certify_with_exact_topk_rows,
    topk_row_proof_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine only Q4 LM-head top-K rows with exact residuals, then prove "
            "that no unread outside row can beat the exact refined winner."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--column-block", type=int, default=16)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_head_topk_row_proof.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.tokens, args.hot_bits, args.top_k, args.column_block) <= 0:
        raise SystemExit("tokens, precision, top-K and block size must be positive")

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
    certified = 0
    unsafe = 0
    hot_top1 = 0
    hot_top4 = 0
    hot_top32 = 0
    exact_reference_mismatches = 0
    exact_winner_in_hot_topk = 0
    rows: list[dict[str, int | float | bool]] = []
    for position in range(args.tokens):
        reference_token = int(exact_tokens[0, position].item())
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        hot_rank = exact_token_rank(hot_logits[position], reference_token)
        hot_indices = torch.topk(
            hot_logits[position],
            k=min(args.top_k, hot_logits.shape[-1]),
        ).indices
        winner_in_topk = bool(torch.any(hot_indices == exact_argmax).item())
        certificate = certify_with_exact_topk_rows(
            hot_logits=hot_logits[position],
            activation=hidden[position],
            residual=residual,
            residual_norms=residual_norms,
            column_block=args.column_block,
            top_k=args.top_k,
        )
        safe = (not certificate.certified) or certificate.candidate == exact_argmax
        certified += int(certificate.certified)
        unsafe += int(not safe)
        hot_top1 += int(hot_rank == 1)
        hot_top4 += int(hot_rank <= 4)
        hot_top32 += int(hot_rank <= 32)
        exact_winner_in_hot_topk += int(winner_in_topk)
        exact_reference_mismatches += int(reference_token != exact_argmax)
        rows.append(
            {
                "position": position,
                "reference_token": reference_token,
                "exact_argmax": exact_argmax,
                "hot_exact_token_rank": hot_rank,
                "exact_winner_in_hot_topk": winner_in_topk,
                "certified": certificate.certified,
                "certificate_candidate": certificate.candidate,
                "certificate_safe": safe,
                "certified_margin": certificate.certified_margin,
                "strongest_outside_competitor": (
                    certificate.strongest_outside_competitor
                ),
                "strongest_outside_upper_bound": (
                    certificate.strongest_outside_upper_bound
                ),
                "candidate_exact_logit": certificate.candidate_exact_logit,
            }
        )

    tiny_refinement = topk_row_proof_budget(
        rows=exact_weight.shape[0],
        columns=exact_weight.shape[1],
        top_k=args.top_k,
        source_bits=16,
        hot_bits=args.hot_bits,
    )
    target_refinement = topk_row_proof_budget(
        rows=128_256,
        columns=16_384,
        top_k=args.top_k,
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
        and exact_reference_mismatches == 0
        and certificate_rate >= 0.95
        and target_refinement.exact_residual_gib_per_token < 0.01
    )
    payload = {
        "evidence_level": "E2 pretrained exact top-K LM-head row proof",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "top_k": args.top_k,
        "column_block": args.column_block,
        "hot_quantization": quant_stats.to_dict(),
        "hot_top1_rate": hot_top1 / args.tokens,
        "hot_top4_rate": hot_top4 / args.tokens,
        "hot_top32_rate": hot_top32 / args.tokens,
        "exact_winner_in_hot_topk_rate": exact_winner_in_hot_topk / args.tokens,
        "certificate_rate": certificate_rate,
        "certified_tokens": certified,
        "unsafe_certificates": unsafe,
        "exact_reference_mismatches": exact_reference_mismatches,
        "tiny_exact_refinement_budget": tiny_refinement.to_dict(),
        "projected_405b_exact_refinement_budget": target_refinement.to_dict(),
        "projected_405b_outside_metadata_budget": target_metadata.to_dict(),
        "positions": rows,
        "contract": (
            "Only hot top-K LM-head residual rows are read exactly. All outside "
            "rows remain unread and are excluded by conservative row/block norm "
            "upper bounds. A missing exact winner therefore prevents certification "
            "rather than causing an unsafe commit."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance exact top-K row proofs into model-wide adjoint refinement"
            if qualifies
            else "increase top-K or tighten outside-row support bounds"
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
