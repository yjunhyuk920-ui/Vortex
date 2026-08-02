from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.adjoint_profiler import profile_exact_target_margin_tiles
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
            "Rank exact residual weight tiles by first-order influence on the "
            "exact-target versus approximate-competitor logit margin."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=2)
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
        default=Path("oracle_adjoint_tile_repair.json"),
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


def count_within_bytes(cumulative: list[int], maximum: float) -> int:
    count = 0
    for index, value in enumerate(cumulative, start=1):
        if value > maximum:
            break
        count = index
    return count


def repair_result(
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
        "tokens_per_full_repair_equivalent": (
            None if exact_bytes == 0 else full_model_bytes / exact_bytes
        ),
        "exact_sequence_match": exact_match,
    }


def evaluate_ranking(
    *,
    ranking_name: str,
    candidates: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    exact_tokens: list[int],
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    max_new_tokens: int,
    row_tile: int,
    col_tile: int,
    full_model_bytes: int,
) -> dict[str, Any]:
    cumulative: list[int] = []
    total = 0
    for item in candidates:
        total += int(item["weight_bytes"])
        cumulative.append(total)

    gate_minimum = 491.29915997929805
    promotion = 600.0
    rejection = 300.0
    promotion_count = count_within_bytes(cumulative, full_model_bytes / promotion)
    gate_count = count_within_bytes(cumulative, full_model_bytes / gate_minimum)
    rejection_count = count_within_bytes(cumulative, full_model_bytes / rejection)

    probes = {0, 1, 2, 4, 8, 16, 32, 64}
    probes.update(range(4, rejection_count + 1, 4))
    probes.update({promotion_count, gate_count, rejection_count})
    value = 128
    while value < len(candidates):
        probes.add(value)
        value *= 2
    probes.add(len(candidates))

    tested: list[dict[str, int | float | bool | None]] = []
    first_match = None
    gate_match = None
    for count in sorted(value for value in probes if 0 <= value <= len(candidates)):
        exact_bytes = configure_top_tiles(
            replacements,
            candidates,
            count,
            row_tile,
            col_tile,
        )
        generated = generate(
            model,
            tokenizer,
            eval_prompt,
            device,
            max_new_tokens,
        )
        item = repair_result(
            count=count,
            exact_bytes=exact_bytes,
            full_model_bytes=full_model_bytes,
            exact_match=generated == exact_tokens,
        )
        tested.append(item)
        if item["exact_sequence_match"] and first_match is None:
            first_match = item
        if (
            item["exact_sequence_match"]
            and exact_bytes <= full_model_bytes / gate_minimum
            and gate_match is None
        ):
            gate_match = item

    return {
        "ranking": ranking_name,
        "candidate_tiles": len(candidates),
        "promotion_budget_tile_count": promotion_count,
        "gate_budget_tile_count": gate_count,
        "rejection_budget_tile_count": rejection_count,
        "gate_budget_match": gate_match,
        "first_repair_match": first_match,
        "tested_prefixes": tested,
        "top_tiles": candidates[:64],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.row_tile <= 0 or args.col_tile <= 0:
        raise ValueError("tile dimensions must be positive")

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
    exact_tokens = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.max_new_tokens,
    )

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    for module in replacements.values():
        module.set_mode("learn_exact")
    for prompt in build_prompts:
        generate(model, tokenizer, prompt, device, args.max_new_tokens)

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
        exact_sequence=exact_tokens,
        replacements=replacements,
        device=device,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        encode_prompt=encode_prompt,
    )

    positive = sorted(
        profile.candidates,
        key=lambda item: (
            float(item["positive_contribution_per_byte"]),
            float(item["signed_margin_contribution"]),
        ),
        reverse=True,
    )
    absolute = sorted(
        profile.candidates,
        key=lambda item: (
            float(item["absolute_contribution_per_byte"]),
            float(item["absolute_margin_contribution"]),
        ),
        reverse=True,
    )

    positive_result = evaluate_ranking(
        ranking_name="positive-signed-margin",
        candidates=positive,
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_tokens=exact_tokens,
        replacements=replacements,
        device=device,
        max_new_tokens=args.max_new_tokens,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        full_model_bytes=full_model_bytes,
    )
    absolute_result = evaluate_ranking(
        ranking_name="absolute-margin-influence",
        candidates=absolute,
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_tokens=exact_tokens,
        replacements=replacements,
        device=device,
        max_new_tokens=args.max_new_tokens,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        full_model_bytes=full_model_bytes,
    )

    gate_match = (
        positive_result["gate_budget_match"]
        or absolute_result["gate_budget_match"]
    )
    first_matches = [
        item
        for item in (
            positive_result["first_repair_match"],
            absolute_result["first_repair_match"],
        )
        if item is not None
    ]
    best_first_match = max(
        first_matches,
        key=lambda item: float(item["tokens_per_full_repair_equivalent"]),
        default=None,
    )
    if gate_match is not None:
        decision = "adjoint tile oracle meets Gate 0 byte envelope"
    elif best_first_match is not None:
        decision = "adjoint tile oracle restores output only below Gate 0 efficiency"
    else:
        decision = "adjoint tile oracle did not restore the sequence"

    return {
        "status": "ok",
        "evidence_level": "E1 exact-target adjoint tile oracle",
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
        "adjoint": profile.metadata(),
        "positive_ranking": positive_result,
        "absolute_ranking": absolute_result,
        "gate_budget_match": gate_match,
        "best_first_repair_match": best_first_match,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Extremely optimistic oracle: exact generated targets define the "
            "margin and exact evaluation gradients rank tiles. A deployable "
            "runtime would have to infer this information without the answer."
        ),
    }


def output_path_from_argv(default: Path) -> Path:
    try:
        index = sys.argv.index("--output")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return default


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rank_min": result["rank_min"],
                "rank_max": result["rank_max"],
                "missing_gradient_modules": result["adjoint"][
                    "missing_gradient_modules"
                ],
                "non_differentiable_modules": result["adjoint"][
                    "non_differentiable_modules"
                ],
                "gate_budget_match": result["gate_budget_match"],
                "best_first_repair_match": result["best_first_repair_match"],
                "decision": result["decision"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        output = output_path_from_argv(Path("oracle_adjoint_tile_repair.json"))
        diagnostic = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        output.write_text(
            json.dumps(diagnostic, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(diagnostic, indent=2, allow_nan=False))
        raise
