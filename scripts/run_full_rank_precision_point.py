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
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import (
    fake_quantize_full_rank_modules,
    full_rank_hot_budget,
)

CANDIDATE_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much exact continuation information survives a "
            "training-free full-rank coarse-precision execution path."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--hot-effective-tops", type=float, required=True)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_rank_precision_point.json"),
    )
    return parser.parse_args()


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
    return torch.stack(generated, dim=1)


def teacher_forced_logits(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    if exact_tokens.shape[1] > 1:
        teacher_ids = torch.cat((input_ids, exact_tokens[:, :-1]), dim=1)
    else:
        teacher_ids = input_ids
    kwargs: dict[str, Any] = {
        "input_ids": teacher_ids,
        "use_cache": False,
        "return_dict": True,
    }
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        extension = torch.ones(
            (attention_mask.shape[0], exact_tokens.shape[1] - 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        kwargs["attention_mask"] = torch.cat((attention_mask, extension), dim=1)
    with torch.inference_mode():
        output = model(**kwargs)
    start = input_ids.shape[1] - 1
    end = start + exact_tokens.shape[1]
    return output.logits[:, start:end, :].detach().to("cpu", torch.float32)


def common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    unequal = torch.nonzero(left.reshape(-1) != right.reshape(-1), as_tuple=False)
    if unequal.numel() == 0:
        return int(left.numel())
    return int(unequal[0, 0].item())


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.row_chunk <= 0:
        raise SystemExit("tokens and row chunk must be positive")
    if not 2 <= args.bits < 16:
        raise SystemExit("bits must be in [2, 16)")

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
    exact_tokens = greedy_tokens(model=model, encoded=encoded, count=args.tokens)
    exact_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    exact_teacher_top1 = torch.argmax(exact_logits, dim=-1)
    if not torch.equal(exact_teacher_top1, exact_tokens.to("cpu")):
        raise RuntimeError("teacher-forced exact logits disagree with exact greedy tokens")

    precision_stats, per_tensor = fake_quantize_full_rank_modules(
        model,
        bits=args.bits,
        source_bits=16,
        row_chunk=args.row_chunk,
    )
    gc.collect()

    coarse_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    coarse_tokens = torch.argmax(coarse_logits, dim=-1)
    rows: list[CandidateCoverageRow] = []
    row_payloads: list[dict[str, int | float | bool]] = []
    for position in range(args.tokens):
        exact_token = int(exact_tokens[0, position].item())
        logits = coarse_logits[0, position]
        hot_token = int(torch.argmax(logits).item())
        row = CandidateCoverageRow(
            position=position,
            exact_token=exact_token,
            hot_token=hot_token,
            exact_token_rank=token_rank(logits, exact_token),
            hot_top1_margin=top1_margin(logits),
            exact_logit_gap_from_hot_top1=float(
                (logits[hot_token] - logits[exact_token]).item()
            ),
        )
        rows.append(row)
        exact_margin = top1_margin(exact_logits[0, position])
        logit_relative_error = float(
            (
                torch.linalg.vector_norm(
                    coarse_logits[0, position] - exact_logits[0, position]
                )
                / torch.clamp(
                    torch.linalg.vector_norm(exact_logits[0, position]),
                    min=1e-12,
                )
            ).item()
        )
        row_payloads.append(
            {
                "position": position,
                "exact_token": exact_token,
                "coarse_token": hot_token,
                "exact_match": row.exact_match,
                "exact_token_rank": row.exact_token_rank,
                "coarse_top1_margin": row.hot_top1_margin,
                "exact_top1_margin": exact_margin,
                "exact_gap_from_coarse_top1": row.exact_logit_gap_from_hot_top1,
                "logit_relative_l2_error": logit_relative_error,
            }
        )

    coarse_autoregressive_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    exact_tokens_cpu = exact_tokens.to("cpu")
    autoregressive_prefix = common_prefix_length(
        coarse_autoregressive_tokens,
        exact_tokens_cpu,
    )
    autoregressive_matches = int(
        torch.count_nonzero(coarse_autoregressive_tokens == exact_tokens_cpu).item()
    )

    first_divergence = next(
        (payload for payload in row_payloads if not bool(payload["exact_match"])),
        None,
    )
    target, baseline = default_specs()
    budgets = {
        str(block): full_rank_hot_budget(
            target=target,
            baseline=baseline,
            hot_bits=args.bits,
            block_positions=block,
            host_to_device_gib_s=args.host_to_device_gib_s,
            hot_effective_tops=args.hot_effective_tops,
        ).to_dict()
        for block in (64, 128, 256, 512, 1024, 2048, 4096)
    }
    coverage = coverage_at_k(rows, CANDIDATE_WIDTHS)
    top1_rate = coverage["1"]
    qualifies_as_hot_candidate = bool(
        top1_rate >= 0.90
        and coverage["32"] >= 0.99
        and any(point["ideal_pass"] for point in budgets.values())
    )
    worst_tensors = sorted(
        per_tensor,
        key=lambda item: item.relative_l2_error,
        reverse=True,
    )[:12]

    payload = {
        "evidence_level": "E1 full-rank progressive precision diagnostic",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "source_bits": 16,
        "tokens": args.tokens,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "precision": precision_stats.to_dict(),
        "worst_relative_error_tensors": [item.to_dict() for item in worst_tensors],
        "teacher_forced": {
            "exact_top1_match_rate": top1_rate,
            "coverage_at_k": coverage,
            "first_divergence": first_divergence,
            "mean_exact_token_rank": sum(row.exact_token_rank for row in rows)
            / len(rows),
            "maximum_exact_token_rank": max(row.exact_token_rank for row in rows),
            "rows": row_payloads,
        },
        "autoregressive": {
            "exact_prefix_tokens": autoregressive_prefix,
            "exact_match_count": autoregressive_matches,
            "exact_match_rate": autoregressive_matches / args.tokens,
        },
        "405b_hot_roofline": {
            "host_to_device_gib_s": args.host_to_device_gib_s,
            "hot_effective_tops": args.hot_effective_tops,
            "blocks": budgets,
        },
        "contract": (
            "Every original weight direction is retained in the coarse path. "
            "The source checkpoint is unchanged and supplies the remaining "
            "exact residual precision to a future progressive repair stage."
        ),
        "qualifies_as_hot_candidate": qualifies_as_hot_candidate,
        "decision": (
            "advance this precision into progressive residual certification"
            if qualifies_as_hot_candidate
            else "reject this precision as the sole hot representation"
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
                "bits": args.bits,
                "top1": top1_rate,
                "top32": coverage["32"],
                "autoregressive_prefix": autoregressive_prefix,
                "maximum_exact_token_rank": payload["teacher_forced"][
                    "maximum_exact_token_rank"
                ],
                "qualifies": qualifies_as_hot_candidate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
