from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.residual_tile_repair import (
    ResidualTileAtlasLinearModule,
    replace_with_residual_tile_modules,
    set_residual_replacement_modes,
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
            "Profile W(I-UU^T)x residual contribution bounds by two-dimensional "
            "weight tile and test top-ranked exact tile repairs."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--col-tile", type=int, default=128)
    parser.add_argument(
        "--build-prompt",
        action="append",
        dest="build_prompts",
    )
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oracle_residual_tile_repair.json"),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("pip install transformers sentencepiece") from exc
    return AutoModelForCausalLM, AutoTokenizer


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> list[int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return output[0].detach().to("cpu").tolist()


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
    replacements: dict[str, ResidualTileAtlasLinearModule],
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


def prefix_metrics(
    *,
    count: int,
    exact_bytes: int,
    full_model_bytes: int,
    exact_match: bool,
) -> dict[str, int | float | bool | None]:
    return {
        "selected_tiles": count,
        "exact_weight_bytes_per_token": exact_bytes,
        "full_model_repair_fraction_per_token": exact_bytes / full_model_bytes,
        "zero_exact_repair": exact_bytes == 0,
        "tokens_per_full_repair_equivalent": (
            None if exact_bytes == 0 else full_model_bytes / exact_bytes
        ),
        "exact_sequence_match": exact_match,
    }


def count_within_bytes(
    cumulative_bytes: list[int],
    maximum: float,
) -> int:
    result = 0
    for index, value in enumerate(cumulative_bytes, start=1):
        if value > maximum:
            break
        result = index
    return result


def main() -> None:
    args = parse_args()
    if args.row_tile <= 0 or args.col_tile <= 0:
        raise SystemExit("tile dimensions must be positive")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    suffixes = tuple(
        args.suffixes or ("self_attn.o_proj", "mlp.down_proj")
    )
    replacements = replace_with_residual_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    set_residual_replacement_modes(replacements, "exact")
    exact_tokens = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.max_new_tokens,
    )

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    set_residual_replacement_modes(replacements, "learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
        )

    ranks = {name: module.atlas.rank for name, module in replacements.items()}
    capsule_bytes = sum(
        module.atlas.capsule_bytes for module in replacements.values()
    )
    full_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )

    for module in replacements.values():
        module.reset_residual_tile_profile(
            row_tile=args.row_tile,
            col_tile=args.col_tile,
        )
        module.set_mode("profile_residual")
    profiled_tokens = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.max_new_tokens,
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

    cumulative_bytes: list[int] = []
    total = 0
    for item in candidates:
        total += int(item["weight_bytes"])
        cumulative_bytes.append(total)

    gate_minimum = 491.29915997929805
    promotion_threshold = 600.0
    rejection_threshold = 300.0
    max_gate_bytes = full_model_bytes / gate_minimum
    max_promotion_bytes = full_model_bytes / promotion_threshold
    max_rejection_bytes = full_model_bytes / rejection_threshold
    promotion_count = count_within_bytes(cumulative_bytes, max_promotion_bytes)
    gate_count = count_within_bytes(cumulative_bytes, max_gate_bytes)
    rejection_count = count_within_bytes(cumulative_bytes, max_rejection_bytes)

    probe_counts = {0, 1, 2, 4, 8, 16, 32, 64}
    probe_counts.update(
        count
        for count in range(4, rejection_count + 1, 4)
    )
    probe_counts.update(
        {promotion_count, gate_count, rejection_count}
    )
    value = 128
    while value < len(candidates):
        probe_counts.add(value)
        value *= 2
    probe_counts.add(len(candidates))
    ordered_counts = sorted(
        count
        for count in probe_counts
        if 0 <= count <= len(candidates)
    )

    started = time.perf_counter()
    tested: list[dict[str, int | float | bool | None]] = []
    all_project_match = profiled_tokens == exact_tokens
    first_match: dict[str, int | float | bool | None] | None = None
    gate_budget_match: dict[str, int | float | bool | None] | None = None

    for count in ordered_counts:
        if count == 0:
            exact_bytes = 0
            tokens = profiled_tokens
        else:
            exact_bytes = configure_top_tiles(
                replacements,
                candidates,
                count,
                args.row_tile,
                args.col_tile,
            )
            tokens = generate(
                model,
                tokenizer,
                args.eval_prompt,
                device,
                args.max_new_tokens,
            )
        item = prefix_metrics(
            count=count,
            exact_bytes=exact_bytes,
            full_model_bytes=full_model_bytes,
            exact_match=tokens == exact_tokens,
        )
        tested.append(item)
        if item["exact_sequence_match"] and first_match is None:
            first_match = item
        if (
            item["exact_sequence_match"]
            and exact_bytes <= max_gate_bytes
            and gate_budget_match is None
        ):
            gate_budget_match = item

    elapsed = time.perf_counter() - started
    if all_project_match:
        decision = "all-project residual capsule matched this sequence"
    elif gate_budget_match is not None:
        decision = "2D residual tile oracle meets Gate 0 byte envelope"
    elif first_match is not None:
        decision = "2D residual tile oracle restores output only below Gate 0 efficiency"
    else:
        decision = "2D residual tile top-prefix oracle did not restore the sequence"

    result = {
        "evidence_level": "E1 optimistic 2D residual tile oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "row_tile": args.row_tile,
        "col_tile": args.col_tile,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "max_new_tokens": args.max_new_tokens,
        "rank_min": min(ranks.values()),
        "rank_max": max(ranks.values()),
        "rank_mean": sum(ranks.values()) / len(ranks),
        "capsule_bytes": capsule_bytes,
        "full_model_weight_bytes": full_model_bytes,
        "managed_weight_bytes": managed_weight_bytes,
        "candidate_tiles": len(candidates),
        "all_project_sequence_match": all_project_match,
        "gate_minimum": gate_minimum,
        "promotion_threshold": promotion_threshold,
        "rejection_threshold": rejection_threshold,
        "max_exact_bytes_for_gate": max_gate_bytes,
        "max_exact_bytes_for_promotion": max_promotion_bytes,
        "max_exact_bytes_for_rejection": max_rejection_bytes,
        "promotion_budget_tile_count": promotion_count,
        "gate_budget_tile_count": gate_count,
        "rejection_budget_tile_count": rejection_count,
        "gate_budget_match": gate_budget_match,
        "first_repair_match": first_match,
        "tested_prefixes": tested,
        "top_profiled_tiles": candidates[:64],
        "decision": decision,
        "elapsed_seconds": elapsed,
        "scope_note": (
            "Optimistic oracle: the exact evaluation residual ranks tiles, "
            "prefill repair is not separately charged, and only nested top-"
            "score prefixes are tested rather than every tile subset."
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
                "rank_min": result["rank_min"],
                "rank_max": result["rank_max"],
                "candidate_tiles": len(candidates),
                "gate_budget_tile_count": gate_count,
                "all_project_sequence_match": all_project_match,
                "gate_budget_match": gate_budget_match,
                "first_repair_match": first_match,
                "decision": decision,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
