from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import (
    DEFAULT_BUILD_PROMPTS,
    configure_top_tiles,
    encode_prompt,
    generate,
    longest_common_prefix,
    model_parameter_bytes,
    require_transformers,
)
from vortex_runtime.adjoint_profiler import profile_exact_target_margin_tiles
from vortex_runtime.block_gate import (
    BlockSharedGate,
    maximum_selected_bytes_for_combined_gate,
    maximum_selected_bytes_for_compute,
)
from vortex_runtime.decision_tile_repair import (
    replace_with_decision_tile_modules,
)
from vortex_runtime.feasibility import default_gate0_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test one exact-target residual tile set shared across a long token "
            "block while enforcing both the 405B traffic and compute gates."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--col-tile", type=int, default=128)
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oracle_block_shared_gate.json"),
    )
    return parser.parse_args()


def count_within_bytes(cumulative: list[int], maximum: float) -> int:
    count = 0
    for index, value in enumerate(cumulative, start=1):
        if value > maximum:
            break
        count = index
    return count


def probe_counts(
    *,
    candidate_count: int,
    combined_count: int,
    compute_count: int,
    traffic_count: int,
) -> list[int]:
    counts = {
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        combined_count,
        compute_count,
        traffic_count,
        768,
        1024,
        2048,
        4096,
        8192,
        candidate_count,
    }
    for center in (combined_count, compute_count):
        for offset in (-128, -64, -32, 32, 64, 128):
            counts.add(center + offset)
    return sorted(count for count in counts if 0 <= count <= candidate_count)


def _is_better_combined(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
) -> bool:
    if current is None:
        return True
    candidate_key = (
        int(candidate["incremental_committed_tokens"]),
        int(candidate["committed_prefix_tokens"]),
        -int(candidate["selected_weight_bytes"]),
    )
    current_key = (
        int(current["incremental_committed_tokens"]),
        int(current["committed_prefix_tokens"]),
        -int(current["selected_weight_bytes"]),
    )
    return candidate_key > current_key


def main() -> None:
    args = parse_args()
    if args.block_tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("block and build token counts must be positive")
    if args.row_tile <= 0 or args.col_tile <= 0:
        raise SystemExit("tile dimensions must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    suffixes = tuple(args.suffixes or ("self_attn.o_proj", "mlp.down_proj"))
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    for module in replacements.values():
        module.set_mode("exact")
    exact_sequence = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.block_tokens,
    )
    prompt_length = int(
        encode_prompt(tokenizer, args.eval_prompt, device)["input_ids"].shape[-1]
    )
    exact_generated = exact_sequence[prompt_length:]
    actual_block_tokens = len(exact_generated)
    if actual_block_tokens <= 0:
        raise RuntimeError("model generated no evaluation tokens")

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    for module in replacements.values():
        module.set_mode("learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.build_new_tokens,
        )

    ranks = {name: module.atlas.rank for name, module in replacements.items()}
    capsule_bytes = sum(
        module.atlas.capsule_bytes for module in replacements.values()
    )
    test_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )

    target_report = default_gate0_report(1.0)
    target_hot_gflop = float(target_report["compute"]["hot_total_gflop_per_token"])
    target_full_repair_gflop = float(
        target_report["compute"]["cold_full_repair_gflop"]
    )
    target_compute_limit = float(
        target_report["compute"]["limit_gflop_per_token"]
    )
    minimum_traffic_efficiency = float(
        target_report["traffic"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )

    started = time.perf_counter()
    profile = profile_exact_target_margin_tiles(
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_sequence=exact_sequence,
        replacements=replacements,
        device=device,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        encode_prompt=encode_prompt,
    )
    candidates = sorted(
        profile.candidates,
        key=lambda item: (
            float(item["positive_contribution_per_byte"]),
            float(item["signed_margin_contribution"]),
        ),
        reverse=True,
    )

    cumulative: list[int] = []
    total = 0
    for item in candidates:
        total += int(item["weight_bytes"])
        cumulative.append(total)

    traffic_max_bytes = (
        actual_block_tokens
        * test_model_bytes
        / minimum_traffic_efficiency
    )
    compute_max_bytes = maximum_selected_bytes_for_compute(
        full_model_weight_bytes=test_model_bytes,
        hot_gflop_per_token=target_hot_gflop,
        full_exact_repair_gflop_per_token=target_full_repair_gflop,
        compute_limit_gflop_per_token=target_compute_limit,
    )
    combined_max_bytes = maximum_selected_bytes_for_combined_gate(
        committed_tokens=actual_block_tokens,
        full_model_weight_bytes=test_model_bytes,
        minimum_traffic_efficiency=minimum_traffic_efficiency,
        hot_gflop_per_token=target_hot_gflop,
        full_exact_repair_gflop_per_token=target_full_repair_gflop,
        compute_limit_gflop_per_token=target_compute_limit,
    )
    traffic_count = count_within_bytes(cumulative, traffic_max_bytes)
    compute_count = count_within_bytes(cumulative, compute_max_bytes)
    combined_count = count_within_bytes(cumulative, combined_max_bytes)

    tested: list[dict[str, Any]] = []
    zero_repair_prefix: int | None = None
    best_combined: dict[str, Any] | None = None
    best_efficiency: dict[str, Any] | None = None

    for count in probe_counts(
        candidate_count=len(candidates),
        combined_count=combined_count,
        compute_count=compute_count,
        traffic_count=traffic_count,
    ):
        selected_bytes = configure_top_tiles(
            replacements,
            candidates,
            count,
            args.row_tile,
            args.col_tile,
        )
        candidate_sequence = generate(
            model,
            tokenizer,
            args.eval_prompt,
            device,
            args.block_tokens,
        )
        committed_prefix = longest_common_prefix(
            exact_generated,
            candidate_sequence[prompt_length:],
        )
        if count == 0:
            zero_repair_prefix = committed_prefix
        if zero_repair_prefix is None:
            raise RuntimeError("zero-repair baseline must be evaluated first")

        incremental = max(0, committed_prefix - zero_repair_prefix)
        gate = BlockSharedGate(
            committed_tokens=committed_prefix,
            selected_weight_bytes=selected_bytes,
            full_model_weight_bytes=test_model_bytes,
            minimum_traffic_efficiency=minimum_traffic_efficiency,
            hot_gflop_per_token=target_hot_gflop,
            full_exact_repair_gflop_per_token=target_full_repair_gflop,
            compute_limit_gflop_per_token=target_compute_limit,
        )
        repair_gain_pass = count > 0 and incremental > 0
        logical_oracle_pass = gate.pass_all and repair_gain_pass
        item = {
            "selected_tiles": count,
            "selected_weight_bytes": selected_bytes,
            "committed_prefix_tokens": committed_prefix,
            "zero_repair_prefix_tokens": zero_repair_prefix,
            "incremental_committed_tokens": incremental,
            "repair_gain_pass": repair_gain_pass,
            "logical_oracle_pass": logical_oracle_pass,
            "full_block_match": committed_prefix == actual_block_tokens,
            **gate.to_dict(),
        }
        tested.append(item)

        efficiency = gate.traffic_efficiency
        if logical_oracle_pass and _is_better_combined(item, best_combined):
            best_combined = item
        if repair_gain_pass and efficiency is not None and (
            best_efficiency is None
            or float(efficiency) > float(best_efficiency["traffic_efficiency"])
        ):
            best_efficiency = item

    if best_combined is not None:
        decision = (
            "block-shared repair survives the E1 combined oracle; "
            "selector and certificate remain unproven"
        )
    elif best_efficiency is not None and bool(best_efficiency["traffic_pass"]):
        decision = "repair increases the prefix but fails the compute gate"
    else:
        decision = "block-shared repair fails the combined Gate 0 oracle"

    result = {
        "evidence_level": "E1 exact-target block-shared combined oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "row_tile": args.row_tile,
        "col_tile": args.col_tile,
        "requested_block_tokens": args.block_tokens,
        "actual_generated_tokens": actual_block_tokens,
        "build_new_tokens": args.build_new_tokens,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "rank_min": min(ranks.values()),
        "rank_max": max(ranks.values()),
        "rank_mean": sum(ranks.values()) / len(ranks),
        "capsule_bytes": capsule_bytes,
        "test_model_weight_bytes": test_model_bytes,
        "managed_weight_bytes": managed_weight_bytes,
        "candidate_tiles": len(candidates),
        "zero_repair_prefix_tokens": zero_repair_prefix,
        "adjoint": profile.metadata(),
        "target_compute": {
            "hot_gflop_per_token": target_hot_gflop,
            "full_exact_repair_gflop_per_token": target_full_repair_gflop,
            "limit_gflop_per_token": target_compute_limit,
        },
        "combined_budget": {
            "minimum_traffic_efficiency": minimum_traffic_efficiency,
            "traffic_max_shared_bytes": traffic_max_bytes,
            "compute_max_selected_bytes": compute_max_bytes,
            "combined_max_selected_bytes": combined_max_bytes,
            "traffic_budget_tile_count": traffic_count,
            "compute_budget_tile_count": compute_count,
            "combined_budget_tile_count": combined_count,
        },
        "tested_shared_tile_sets": tested,
        "best_combined_candidate": best_combined,
        "best_efficiency_candidate": best_efficiency,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Extremely optimistic oracle: exact target tokens and gradients rank "
            "one shared tile set. A pass only means the logical byte/compute "
            "envelope survives and the repair increases the exact prefix beyond "
            "the zero-repair baseline. No deployable selector or certificate is "
            "proved."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "actual_generated_tokens": actual_block_tokens,
                "zero_repair_prefix_tokens": zero_repair_prefix,
                "combined_budget_tile_count": combined_count,
                "compute_max_selected_bytes": compute_max_bytes,
                "best_combined_candidate": best_combined,
                "best_efficiency_candidate": best_efficiency,
                "decision": decision,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
