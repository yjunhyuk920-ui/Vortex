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

from scripts.run_full_rank_precision_point import greedy_tokens
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import fake_quantize_full_rank_modules
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    speculative_tree_verification_budget,
    unique_prefix_node_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a causal Q4 top-k beam tree and measure the optimistic exact "
            "target prefix that one offloaded-model verification could commit."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--node-budget", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument("--hot-effective-tops", type=float, default=160.0)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("q4_speculative_tree_gate.json"),
    )
    return parser.parse_args()


def _next_logits(
    *,
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    sequences: list[tuple[int, ...]],
    batch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        continuation = torch.tensor(
            batch,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        repeated_prompt = prompt_ids.expand(len(batch), -1)
        input_ids = torch.cat((repeated_prompt, continuation), dim=1)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": False,
            "return_dict": True,
        }
        if prompt_mask is not None:
            repeated_mask = prompt_mask.expand(len(batch), -1)
            continuation_mask = torch.ones_like(
                continuation,
                dtype=prompt_mask.dtype,
            )
            kwargs["attention_mask"] = torch.cat(
                (repeated_mask, continuation_mask),
                dim=1,
            )
        with torch.inference_mode():
            output = model(**kwargs)
        rows.append(output.logits[:, -1, :].detach().to("cpu", torch.float32))
    return torch.cat(rows, dim=0)


def _trim_to_node_budget(
    candidates: list[tuple[float, tuple[int, ...]]],
    *,
    beam_width: int,
    node_budget: int,
) -> list[tuple[float, tuple[int, ...]]]:
    retained: list[tuple[float, tuple[int, ...]]] = []
    for candidate in candidates:
        trial = retained + [candidate]
        if unique_prefix_node_count([sequence for _, sequence in trial]) > node_budget:
            continue
        retained.append(candidate)
        if len(retained) >= beam_width:
            break
    return retained


def main() -> None:
    args = parse_args()
    if not 2 <= args.bits < 16:
        raise SystemExit("bits must be in [2, 16)")
    if min(
        args.depth,
        args.top_k,
        args.beam_width,
        args.node_budget,
        args.eval_batch_size,
        args.row_chunk,
    ) <= 0:
        raise SystemExit("tree and quantization parameters must be positive")

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
        count=args.depth,
    ).to("cpu")
    reference = tuple(int(token) for token in exact_tokens[0].tolist())

    precision_stats, _ = fake_quantize_full_rank_modules(
        model,
        bits=args.bits,
        source_bits=16,
        row_chunk=args.row_chunk,
    )
    gc.collect()

    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, tuple())]
    depth_rows: list[dict[str, int | float | bool]] = []
    exact_path_last_alive_depth = 0

    for depth_index in range(args.depth):
        sequences = [sequence for _, sequence in beams]
        logits = _next_logits(
            model=model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            sequences=sequences,
            batch_size=args.eval_batch_size,
        )
        log_probabilities = torch.log_softmax(logits, dim=-1)
        width = min(args.top_k, log_probabilities.shape[-1])
        values, indices = torch.topk(log_probabilities, k=width, dim=-1)

        expanded: list[tuple[float, tuple[int, ...]]] = []
        for beam_index, (score, sequence) in enumerate(beams):
            for child_index in range(width):
                token = int(indices[beam_index, child_index].item())
                child_score = score + float(values[beam_index, child_index].item())
                expanded.append((child_score, sequence + (token,)))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = _trim_to_node_budget(
            expanded,
            beam_width=args.beam_width,
            node_budget=args.node_budget,
        )
        if not beams:
            break

        retained_sequences = [sequence for _, sequence in beams]
        current_depth = depth_index + 1
        exact_prefix = longest_reference_prefix(retained_sequences, reference)
        exact_path = reference[:current_depth]
        exact_path_alive = exact_path in retained_sequences
        if exact_path_alive:
            exact_path_last_alive_depth = current_depth
        depth_rows.append(
            {
                "depth": current_depth,
                "retained_branches": len(beams),
                "unique_tree_nodes": unique_prefix_node_count(retained_sequences),
                "longest_exact_reference_prefix": exact_prefix,
                "exact_path_alive": exact_path_alive,
                "best_log_probability": beams[0][0],
                "worst_retained_log_probability": beams[-1][0],
            }
        )

    retained_sequences = [sequence for _, sequence in beams]
    tree_nodes = unique_prefix_node_count(retained_sequences) if beams else 0
    reached_depth = max((len(sequence) for sequence in retained_sequences), default=0)
    optimistic_committed = (
        longest_reference_prefix(retained_sequences, reference) if beams else 0
    )

    target, baseline = default_specs()
    budget = speculative_tree_verification_budget(
        target=target,
        baseline=baseline,
        hot_bits=args.bits,
        tree_nodes=max(1, tree_nodes),
        tree_depth=max(1, reached_depth),
        committed_tokens=optimistic_committed,
        host_to_device_gib_s=args.host_to_device_gib_s,
        hot_effective_tops=args.hot_effective_tops,
    )
    qualifies = bool(
        optimistic_committed == args.depth
        and budget.observed_serialized_pass
    )

    payload = {
        "evidence_level": "E2 causal Q4 tree plus optimistic free-draft resource gate",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "bits": args.bits,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "requested_depth": args.depth,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "node_budget": args.node_budget,
        "retained_tree_nodes": tree_nodes,
        "reached_depth": reached_depth,
        "optimistic_committed_exact_prefix": optimistic_committed,
        "exact_path_last_alive_depth": exact_path_last_alive_depth,
        "precision": precision_stats.to_dict(),
        "depth_rows": depth_rows,
        "405b_free_draft_upper_bound": budget.to_dict(),
        "contract": (
            "The Q4 drafter is charged zero time and zero VRAM. The exact target "
            "verification streams one Q4 full-rank target representation and "
            "evaluates every retained tree node. This is strictly more favorable "
            "than any deployable 8 GiB substitute draft."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance Q4 speculative tree into exact verifier implementation"
            if qualifies
            else "reject tested Q4 speculative tree point"
        ),
        "next_candidate_if_rejected": (
            "retain tree verification only if a much smaller target-derived "
            "substitute preserves a path thousands of tokens deep; otherwise "
            "the 405B PCIe stream cannot be amortized to native-4B latency"
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
