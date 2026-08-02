from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.adjoint_profiler import profile_exact_target_margin_tiles
from vortex_runtime.block_repair import (
    BlockRepairBudget,
    maximum_shared_repair_bytes,
)
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    replace_with_decision_tile_modules,
)


DEFAULT_BUILD_PROMPTS = [
    "Explain why the sky appears blue and why sunsets appear red.",
    "Write Python code for merge sort and explain its complexity.",
    "한국어로 PLM BOM 변경 검증 절차와 오류 처리 방법을 설명해줘.",
    "Solve a probability problem using conditional probability step by step.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use exact-target adjoints to select one residual tile set, stream "
            "that set once for a long proposed block, and measure the exact "
            "causal prefix committed per full-model-equivalent repair."
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
        default=Path("oracle_block_shared_adjoint.json"),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("pip install transformers sentencepiece") from exc
    return AutoModelForCausalLM, AutoTokenizer


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> list[int]:
    encoded = encode_prompt(tokenizer, prompt, device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return output[0].detach().cpu().tolist()


def model_parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        pointer = parameter.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += parameter.numel() * parameter.element_size()
    return total


def configure_top_tiles(
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    candidates: list[dict[str, Any]],
    count: int,
    row_tile: int,
    col_tile: int,
) -> int:
    selected: dict[str, list[tuple[int, int]]] = {
        name: [] for name in replacements
    }
    for item in candidates[:count]:
        selected[str(item["module"])].append(
            (int(item["row_tile"]), int(item["col_tile"]))
        )
    for name, module in replacements.items():
        module.configure_residual_tile_repair(
            row_tile=row_tile,
            col_tile=col_tile,
            tile_indices=selected[name],
        )
        module.set_mode("project_residual_repair")
    return sum(
        module.selected_residual_repair_bytes
        for module in replacements.values()
    )


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


def main() -> None:
    args = parse_args()
    if args.block_tokens <= 0:
        raise SystemExit("--block-tokens must be positive")
    if args.build_new_tokens <= 0:
        raise SystemExit("--build-new-tokens must be positive")
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
    full_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
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

    gate_minimum = 491.29915997929805
    promotion_threshold = 600.0
    rejection_threshold = 300.0
    promotion_bytes = maximum_shared_repair_bytes(
        committed_tokens=args.block_tokens,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=promotion_threshold,
    )
    gate_bytes = maximum_shared_repair_bytes(
        committed_tokens=args.block_tokens,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=gate_minimum,
    )
    rejection_bytes = maximum_shared_repair_bytes(
        committed_tokens=args.block_tokens,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=rejection_threshold,
    )
    promotion_count = count_within_bytes(cumulative, promotion_bytes)
    gate_count = count_within_bytes(cumulative, gate_bytes)
    rejection_count = count_within_bytes(cumulative, rejection_bytes)

    probe_counts = {
        0,
        2048,
        4096,
        6144,
        7168,
        8192,
        10240,
        12288,
        14336,
        16384,
        len(candidates),
        promotion_count,
        gate_count,
        rejection_count,
    }
    ordered_counts = sorted(
        count for count in probe_counts if 0 <= count <= len(candidates)
    )

    tested: list[dict[str, Any]] = []
    best_gate_candidate: dict[str, Any] | None = None
    best_efficiency_candidate: dict[str, Any] | None = None
    for count in ordered_counts:
        exact_bytes = configure_top_tiles(
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
        candidate_generated = candidate_sequence[prompt_length:]
        committed_prefix = longest_common_prefix(
            exact_generated,
            candidate_generated,
        )
        budget = BlockRepairBudget(
            committed_tokens=committed_prefix,
            selected_weight_bytes=exact_bytes,
            full_model_weight_bytes=full_model_bytes,
        )
        efficiency = budget.tokens_per_full_repair_equivalent
        item = {
            "selected_tiles": count,
            "selected_weight_bytes": exact_bytes,
            "committed_prefix_tokens": committed_prefix,
            "full_block_match": committed_prefix == len(exact_generated),
            "repair_fraction": budget.repair_fraction,
            "tokens_per_full_repair_equivalent": efficiency,
            "gate_pass": (
                exact_bytes > 0
                and efficiency is not None
                and efficiency >= gate_minimum
            ),
            "promotion_pass": (
                exact_bytes > 0
                and efficiency is not None
                and efficiency >= promotion_threshold
            ),
        }
        tested.append(item)
        if item["gate_pass"] and (
            best_gate_candidate is None
            or float(efficiency) > float(
                best_gate_candidate["tokens_per_full_repair_equivalent"]
            )
        ):
            best_gate_candidate = item
        if efficiency is not None and (
            best_efficiency_candidate is None
            or float(efficiency) > float(
                best_efficiency_candidate[
                    "tokens_per_full_repair_equivalent"
                ]
            )
        ):
            best_efficiency_candidate = item

    if best_gate_candidate is not None:
        decision = "block-shared adjoint repair reaches Gate 0 logical traffic"
    elif best_efficiency_candidate is not None and float(
        best_efficiency_candidate["tokens_per_full_repair_equivalent"]
    ) >= rejection_threshold:
        decision = "block-shared repair remains below Gate 0 but above rejection floor"
    else:
        decision = "reject 64-token block-shared rank-32 adjoint repair"

    result = {
        "evidence_level": "E1 exact-target block-shared adjoint oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "row_tile": args.row_tile,
        "col_tile": args.col_tile,
        "block_tokens": args.block_tokens,
        "actual_generated_tokens": len(exact_generated),
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
        "adjoint": profile.metadata(),
        "gate": {
            "minimum_efficiency": gate_minimum,
            "promotion_threshold": promotion_threshold,
            "rejection_threshold": rejection_threshold,
            "promotion_max_shared_bytes": promotion_bytes,
            "gate_max_shared_bytes": gate_bytes,
            "rejection_max_shared_bytes": rejection_bytes,
            "promotion_budget_tile_count": promotion_count,
            "gate_budget_tile_count": gate_count,
            "rejection_budget_tile_count": rejection_count,
        },
        "tested_shared_tile_sets": tested,
        "best_gate_candidate": best_gate_candidate,
        "best_efficiency_candidate": best_efficiency_candidate,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Extremely optimistic oracle: exact target tokens and their full "
            "teacher-forced gradients choose one shared tile set. Selected "
            "weight bytes are charged once for the entire block, while exact "
            "tile compute is still applied at each generated position."
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
                "block_tokens": result["block_tokens"],
                "candidate_tiles": result["candidate_tiles"],
                "gate_budget_tile_count": result["gate"][
                    "gate_budget_tile_count"
                ],
                "best_gate_candidate": best_gate_candidate,
                "best_efficiency_candidate": best_efficiency_candidate,
                "decision": decision,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
