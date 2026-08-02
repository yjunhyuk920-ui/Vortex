from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize
from vortex_runtime.residual_proof import (
    certify_linear_argmax,
    residual_metadata_budget,
    rowwise_residual_block_norms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Certify Q4 LM-head argmax decisions using only hierarchical residual "
            "norm metadata. Exact hidden states isolate the output-layer proof."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hot-bits", type=int, default=4)
    parser.add_argument("--column-block", type=int, default=64)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm_head_decision_proof.json"),
    )
    return parser.parse_args()


def exact_prefix_hidden_states(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> torch.Tensor:
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    continuation_input = exact_tokens[:, :-1]
    combined_ids = torch.cat((prompt_ids, continuation_input.to(prompt_ids.device)), dim=1)
    combined_mask = None
    if prompt_mask is not None:
        combined_mask = torch.cat(
            (
                prompt_mask,
                torch.ones_like(
                    continuation_input,
                    dtype=prompt_mask.dtype,
                    device=prompt_mask.device,
                ),
            ),
            dim=1,
        )
    kwargs: dict[str, object] = {
        "input_ids": combined_ids,
        "use_cache": False,
        "return_dict": True,
    }
    if combined_mask is not None:
        kwargs["attention_mask"] = combined_mask
    with torch.inference_mode():
        output = model.model(**kwargs)
    hidden = output.last_hidden_state
    start = prompt_ids.shape[1] - 1
    end = start + exact_tokens.shape[1]
    selected = hidden[:, start:end, :]
    if selected.shape[1] != exact_tokens.shape[1]:
        raise RuntimeError("failed to align hidden states with exact continuation")
    return selected.detach().to("cpu", torch.float32)


def exact_token_rank(logits: torch.Tensor, token: int) -> int:
    target = logits[token]
    return int(torch.count_nonzero(logits > target).item()) + 1


def main() -> None:
    args = parse_args()
    if min(args.tokens, args.hot_bits, args.column_block) <= 0:
        raise SystemExit("tokens, hot bits and column block must be positive")

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
    metadata_started = time.perf_counter()
    residual_norms = rowwise_residual_block_norms(
        residual,
        column_block=args.column_block,
    )
    metadata_seconds = time.perf_counter() - metadata_started

    exact_logits = hidden @ exact_weight.T
    hot_logits = hidden @ hot_weight.T
    rows: list[dict[str, int | float | bool]] = []
    certified = 0
    unsafe = 0
    hot_top1 = 0
    hot_top4 = 0
    hot_top32 = 0
    exact_reference_mismatches = 0
    for position in range(args.tokens):
        reference_token = int(exact_tokens[0, position].item())
        exact_argmax = int(torch.argmax(exact_logits[position]).item())
        hot_argmax = int(torch.argmax(hot_logits[position]).item())
        rank = exact_token_rank(hot_logits[position], reference_token)
        certificate = certify_linear_argmax(
            approximate_logits=hot_logits[position],
            activation=hidden[position],
            residual_norms=residual_norms,
            column_block=args.column_block,
        )
        is_safe = (not certificate.certified) or certificate.candidate == exact_argmax
        certified += int(certificate.certified)
        unsafe += int(not is_safe)
        hot_top1 += int(rank == 1)
        hot_top4 += int(rank <= 4)
        hot_top32 += int(rank <= 32)
        exact_reference_mismatches += int(exact_argmax != reference_token)
        rows.append(
            {
                "position": position,
                "reference_token": reference_token,
                "exact_argmax": exact_argmax,
                "hot_argmax": hot_argmax,
                "exact_token_rank_under_hot": rank,
                "certified": certificate.certified,
                "certificate_candidate": certificate.candidate,
                "certificate_safe": is_safe,
                "certified_margin": certificate.certified_margin,
                "strongest_competitor": certificate.strongest_competitor,
            }
        )

    tiny_budget = residual_metadata_budget(
        rows=exact_weight.shape[0],
        columns=exact_weight.shape[1],
        column_block=args.column_block,
        metadata_bits=16,
    )
    target_budget = residual_metadata_budget(
        rows=128_256,
        columns=16_384,
        column_block=256,
        metadata_bits=16,
    )
    certificate_rate = certified / args.tokens
    qualifies = bool(
        unsafe == 0
        and exact_reference_mismatches == 0
        and certificate_rate >= 0.5
    )
    payload = {
        "evidence_level": "E2 exact LM-head decision proof on pretrained model",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "tokens": args.tokens,
        "hot_bits": args.hot_bits,
        "column_block": args.column_block,
        "hot_quantization": quant_stats.to_dict(),
        "hot_top1_rate": hot_top1 / args.tokens,
        "hot_top4_rate": hot_top4 / args.tokens,
        "hot_top32_rate": hot_top32 / args.tokens,
        "certificate_rate": certificate_rate,
        "certified_tokens": certified,
        "unsafe_certificates": unsafe,
        "exact_reference_mismatches": exact_reference_mismatches,
        "tiny_metadata_budget": tiny_budget.to_dict(),
        "projected_405b_lm_head_metadata_budget": target_budget.to_dict(),
        "metadata_build_seconds": metadata_seconds,
        "positions": rows,
        "contract": (
            "The proof reads Q4 logits, the activation vector and precomputed "
            "row-by-column-block residual norms. It does not read residual weight "
            "values. A token is committed only when the candidate lower bound is "
            "strictly greater than every competitor upper bound."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance decision proofs into internal projections"
            if qualifies
            else "tighten metadata hierarchy or reject tested LM-head proof point"
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
