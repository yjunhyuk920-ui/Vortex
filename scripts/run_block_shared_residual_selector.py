from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_residual_tile_repair import (
    DEFAULT_BUILD_PROMPTS,
    configure_top_tiles,
    generate,
    model_parameter_bytes,
    require_transformers,
)
from vortex_runtime.block_gate import (
    BlockSharedGate,
    maximum_selected_bytes_for_combined_gate,
    maximum_selected_bytes_for_compute,
)
from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.residual_tile_repair import (
    replace_with_residual_tile_modules,
    set_residual_replacement_modes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one block-shared repair set using only projected activation "
            "residual energy and precomputed weight-tile norms. Exact target "
            "tokens are used only for offline evaluation."
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
        default=Path("block_shared_residual_selector.json"),
    )
    return parser.parse_args()


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def count_within_bytes(cumulative: list[int], maximum: float) -> int:
    count = 0
    for index, value in enumerate(cumulative, start=1):
        if value > maximum:
            break
        count = index
    return count


def selector_probe_counts(
    *,
    candidate_count: int,
    combined_count: int,
    cumulative_bytes: list[int],
) -> list[int]:
    counts = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, combined_count}
    for budget_mib in (1, 2, 4, 8, 12, 16, 24, 32, 40):
        counts.add(
            count_within_bytes(cumulative_bytes, budget_mib * 1024 * 1024)
        )
    for offset in (-128, -64, -32, 32, 64, 128):
        counts.add(combined_count + offset)
    return sorted(count for count in counts if 0 <= count <= candidate_count)


def is_better(
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
    replacements = replace_with_residual_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    encoded = tokenizer(args.eval_prompt, return_tensors="pt")
    prompt_length = int(encoded["input_ids"].shape[-1])

    set_residual_replacement_modes(replacements, "exact")
    exact_sequence = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.block_tokens,
    )
    exact_generated = exact_sequence[prompt_length:]
    actual_block_tokens = len(exact_generated)
    if actual_block_tokens <= 0:
        raise RuntimeError("model generated no evaluation tokens")

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    set_residual_replacement_modes(replacements, "learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.build_new_tokens,
        )

    for module in replacements.values():
        module.reset_residual_tile_profile(
            row_tile=args.row_tile,
            col_tile=args.col_tile,
        )
        module.set_mode("profile_residual")

    started = time.perf_counter()
    zero_repair_sequence = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.block_tokens,
    )
    zero_repair_generated = zero_repair_sequence[prompt_length:]
    zero_repair_prefix = longest_common_prefix(
        exact_generated,
        zero_repair_generated,
    )

    candidates: list[dict[str, Any]] = []
    for name, module in replacements.items():
        for tile in module.profiled_residual_tiles():
            candidates.append({"module": name, **tile})
    candidates.sort(
        key=lambda item: (
            float(item["score_per_byte"]),
            float(item["score"]),
        ),
        reverse=True,
    )

    cumulative: list[int] = []
    total = 0
    for item in candidates:
        total += int(item["weight_bytes"])
        cumulative.append(total)

    full_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )
    ranks = {name: module.atlas.rank for name, module in replacements.items()}
    capsule_bytes = sum(
        module.atlas.capsule_bytes for module in replacements.values()
    )

    target_report = default_gate0_report(1.0)
    minimum_traffic_efficiency = float(
        target_report["traffic"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )
    hot_gflop = float(target_report["compute"]["hot_total_gflop_per_token"])
    full_repair_gflop = float(
        target_report["compute"]["cold_full_repair_gflop"]
    )
    compute_limit_gflop = float(
        target_report["compute"]["limit_gflop_per_token"]
    )

    compute_max_bytes = maximum_selected_bytes_for_compute(
        full_model_weight_bytes=full_model_bytes,
        hot_gflop_per_token=hot_gflop,
        full_exact_repair_gflop_per_token=full_repair_gflop,
        compute_limit_gflop_per_token=compute_limit_gflop,
    )
    combined_max_bytes = maximum_selected_bytes_for_combined_gate(
        committed_tokens=actual_block_tokens,
        full_model_weight_bytes=full_model_bytes,
        minimum_traffic_efficiency=minimum_traffic_efficiency,
        hot_gflop_per_token=hot_gflop,
        full_exact_repair_gflop_per_token=full_repair_gflop,
        compute_limit_gflop_per_token=compute_limit_gflop,
    )
    combined_count = count_within_bytes(cumulative, combined_max_bytes)

    tested: list[dict[str, Any]] = []
    best_selector_candidate: dict[str, Any] | None = None
    for count in selector_probe_counts(
        candidate_count=len(candidates),
        combined_count=combined_count,
        cumulative_bytes=cumulative,
    ):
        if count == 0:
            selected_bytes = 0
            candidate_sequence = zero_repair_sequence
        else:
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
        incremental = max(0, committed_prefix - zero_repair_prefix)
        gate = BlockSharedGate(
            committed_tokens=committed_prefix,
            selected_weight_bytes=selected_bytes,
            full_model_weight_bytes=full_model_bytes,
            minimum_traffic_efficiency=minimum_traffic_efficiency,
            hot_gflop_per_token=hot_gflop,
            full_exact_repair_gflop_per_token=full_repair_gflop,
            compute_limit_gflop_per_token=compute_limit_gflop,
        )
        selector_pass = gate.pass_all and incremental > 0
        item = {
            "selected_tiles": count,
            "selected_weight_bytes": selected_bytes,
            "committed_prefix_tokens": committed_prefix,
            "zero_repair_prefix_tokens": zero_repair_prefix,
            "incremental_committed_tokens": incremental,
            "selector_pass": selector_pass,
            "full_block_match": committed_prefix == actual_block_tokens,
            **gate.to_dict(),
        }
        tested.append(item)
        if selector_pass and is_better(item, best_selector_candidate):
            best_selector_candidate = item

    decision = (
        "target-independent residual-energy selector survives the E1 combined gate"
        if best_selector_candidate is not None
        else "target-independent residual-energy selector fails the E1 combined gate"
    )
    result = {
        "evidence_level": "E1 target-independent block selector falsification",
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
        "full_model_weight_bytes": full_model_bytes,
        "managed_weight_bytes": managed_weight_bytes,
        "candidate_tiles": len(candidates),
        "zero_repair_prefix_tokens": zero_repair_prefix,
        "selector_inputs": [
            "approximate autoregressive activation residuals",
            "precomputed exact weight-tile Frobenius norms",
        ],
        "selector_forbidden_inputs": [
            "exact target continuation tokens",
            "teacher-forced gradients",
            "exact target logit margins",
        ],
        "combined_budget": {
            "minimum_traffic_efficiency": minimum_traffic_efficiency,
            "compute_max_selected_bytes": compute_max_bytes,
            "combined_max_selected_bytes": combined_max_bytes,
            "combined_budget_tile_count": combined_count,
        },
        "tested_selector_prefixes": tested,
        "best_selector_candidate": best_selector_candidate,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Exact target tokens are generated only to measure the causal-prefix "
            "agreement after selection. They do not influence tile scores or the "
            "selected set. A sound online commit certificate is still absent."
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
                "zero_repair_prefix_tokens": zero_repair_prefix,
                "combined_budget_tile_count": combined_count,
                "best_selector_candidate": best_selector_candidate,
                "decision": decision,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
