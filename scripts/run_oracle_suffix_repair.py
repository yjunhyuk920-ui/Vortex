from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.falsification import (
    AtlasLinearModule,
    replace_linear_modules,
    set_replacement_modes,
)


LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
DEFAULT_BUILD_PROMPTS = [
    "Explain why the sky appears blue and why sunsets appear red.",
    "Write Python code for merge sort and explain its complexity.",
    "한국어로 PLM BOM 변경 검증 절차와 오류 처리 방법을 설명해줘.",
    "Solve a probability problem using conditional probability step by step.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build approximate Atlas capsules, then use an oracle search to "
            "find the smallest exact layer suffix that restores the original "
            "greedy token sequence."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-rank", type=int, default=32)
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
        default=Path("oracle_suffix_repair.json"),
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


def layer_index(name: str) -> int:
    match = LAYER_PATTERN.search(name)
    if match is None:
        raise ValueError(f"replacement has no layer index: {name}")
    return int(match.group(1))


def configure_exact_suffix(
    replacements: dict[str, AtlasLinearModule],
    *,
    total_layers: int,
    suffix_layers: int,
) -> int:
    if not 0 <= suffix_layers <= total_layers:
        raise ValueError("suffix_layers out of range")
    first_exact = total_layers - suffix_layers
    exact_bytes = 0
    for name, module in replacements.items():
        exact = layer_index(name) >= first_exact
        module.set_mode("exact" if exact else "project")
        if exact:
            exact_bytes += module.logical_weight_bytes
    return exact_bytes


def candidate_suffix_sizes(total_layers: int) -> list[int]:
    sizes = {0, total_layers}
    value = 1
    while value < total_layers:
        sizes.add(value)
        value *= 2
    return sorted(sizes)


def repair_metrics(
    *,
    exact_bytes: int,
    full_model_bytes: int,
    matches: bool,
) -> dict[str, int | float | bool | None]:
    zero_repair = exact_bytes == 0
    return {
        "exact_bytes_per_token": exact_bytes,
        "full_model_repair_fraction_per_token": (
            exact_bytes / full_model_bytes
        ),
        "zero_exact_repair": zero_repair,
        "tokens_per_full_repair_equivalent": (
            None if zero_repair else full_model_bytes / exact_bytes
        ),
        "exact_sequence_match": matches,
    }


def main() -> None:
    args = parse_args()
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
    replacements = replace_linear_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    layer_ids = sorted({layer_index(name) for name in replacements})
    total_layers = max(layer_ids) + 1
    if layer_ids != list(range(total_layers)):
        raise RuntimeError("matched replacement layers are not contiguous")

    set_replacement_modes(replacements, "exact")
    exact_tokens = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.max_new_tokens,
    )

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    set_replacement_modes(replacements, "learn_exact")
    for prompt in build_prompts:
        generate(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
        )

    full_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )
    capsule_bytes = sum(
        module.atlas.capsule_bytes for module in replacements.values()
    )
    ranks = {name: module.atlas.rank for name, module in replacements.items()}

    tested: list[dict[str, int | float | bool | None]] = []
    first_passing_size: int | None = None
    previous_size = 0
    started = time.perf_counter()

    for size in candidate_suffix_sizes(total_layers):
        exact_bytes = configure_exact_suffix(
            replacements,
            total_layers=total_layers,
            suffix_layers=size,
        )
        tokens = generate(
            model,
            tokenizer,
            args.eval_prompt,
            device,
            args.max_new_tokens,
        )
        matches = tokens == exact_tokens
        tested.append(
            {
                "exact_suffix_layers": size,
                **repair_metrics(
                    exact_bytes=exact_bytes,
                    full_model_bytes=full_model_bytes,
                    matches=matches,
                ),
            }
        )
        if matches:
            first_passing_size = size
            break
        previous_size = size

    if first_passing_size is None:
        raise RuntimeError("even all managed exact modules did not restore output")

    minimal_size = first_passing_size
    if first_passing_size - previous_size > 1:
        for size in range(previous_size + 1, first_passing_size):
            exact_bytes = configure_exact_suffix(
                replacements,
                total_layers=total_layers,
                suffix_layers=size,
            )
            tokens = generate(
                model,
                tokenizer,
                args.eval_prompt,
                device,
                args.max_new_tokens,
            )
            matches = tokens == exact_tokens
            tested.append(
                {
                    "exact_suffix_layers": size,
                    **repair_metrics(
                        exact_bytes=exact_bytes,
                        full_model_bytes=full_model_bytes,
                        matches=matches,
                    ),
                }
            )
            if matches:
                minimal_size = size
                break

    minimal = next(
        item
        for item in tested
        if item["exact_suffix_layers"] == minimal_size
    )
    all_project_match = bool(
        next(
            item["exact_sequence_match"]
            for item in tested
            if item["exact_suffix_layers"] == 0
        )
    )
    efficiency_value = minimal["tokens_per_full_repair_equivalent"]
    gate_pass = all_project_match or (
        efficiency_value is not None and float(efficiency_value) >= 600.0
    )
    elapsed = time.perf_counter() - started
    result = {
        "evidence_level": "E1 oracle repair falsification",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "total_layers": total_layers,
        "max_rank": args.max_rank,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "max_new_tokens": args.max_new_tokens,
        "capsule_bytes": capsule_bytes,
        "rank_min": min(ranks.values()),
        "rank_max": max(ranks.values()),
        "rank_mean": sum(ranks.values()) / len(ranks),
        "full_model_weight_bytes": full_model_bytes,
        "managed_weight_bytes": managed_weight_bytes,
        "all_project_sequence_match": all_project_match,
        "minimal_exact_suffix_layers": minimal_size,
        "minimal_exact_bytes_per_token": minimal["exact_bytes_per_token"],
        "minimal_full_model_repair_fraction_per_token": minimal[
            "full_model_repair_fraction_per_token"
        ],
        "zero_exact_repair": minimal["zero_exact_repair"],
        "oracle_tokens_per_full_repair_equivalent": efficiency_value,
        "gate_minimum": 491.29915997929805,
        "promotion_threshold": 600.0,
        "gate_pass": gate_pass,
        "tested_suffixes": sorted(
            tested,
            key=lambda item: int(item["exact_suffix_layers"]),
        ),
        "elapsed_seconds": elapsed,
        "scope_note": (
            "This is an optimistic oracle. It assumes the runtime already "
            "knows which layer suffix will restore the exact sequence."
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
                "all_project_sequence_match": all_project_match,
                "minimal_exact_suffix_layers": minimal_size,
                "zero_exact_repair": minimal["zero_exact_repair"],
                "oracle_tokens_per_full_repair_equivalent": efficiency_value,
                "gate_pass": gate_pass,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
