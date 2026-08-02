from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.falsification import (
    compute_repair_efficiency,
    replace_linear_modules,
    replacement_delta,
    snapshot_replacements,
)


DEFAULT_BUILD_PROMPTS = [
    "Explain in English why the sky appears blue.",
    "Write a Python function that merges two sorted lists.",
    "한국어로 제품 BOM 변경 검증 절차를 설명해줘.",
]

DEFAULT_EVAL_PROMPTS = [
    "Explain in English why sunsets often appear red.",
    "Write a Python function that finds duplicate values in a list.",
    "한국어로 JSON 스키마 검증 실패 원인을 분석해줘.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace real model linear operations with AtlasLinearModule and "
            "measure disjoint-prompt logical repair efficiency."
        )
    )
    parser.add_argument("--model", required=True, help="HF repo ID or local path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-rank", type=int, default=256)
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument("--eval-prompt", action="append", dest="eval_prompts")
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        default=None,
        help=(
            "Linear module suffix to replace. Defaults to self_attn.o_proj "
            "and mlp.down_proj."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_operation_falsification.json"),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "This optional runner requires transformers: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def generate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[int], int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    sequence = output[0].detach().to("cpu").tolist()
    return sequence, max(0, len(sequence) - prompt_tokens)


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

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    eval_prompts = args.eval_prompts or DEFAULT_EVAL_PROMPTS
    suffixes = tuple(
        args.suffixes or ("self_attn.o_proj", "mlp.down_proj")
    )

    exact_eval: list[list[int]] = []
    for prompt in eval_prompts:
        tokens, _ = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        exact_eval.append(tokens)

    full_model_bytes = model_parameter_bytes(model)
    replacements = replace_linear_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no linear modules matched suffixes: {suffixes}")
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )

    build_generated_tokens = 0
    for prompt in build_prompts:
        _, generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        build_generated_tokens += generated

    before = snapshot_replacements(replacements)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    eval_generated_tokens = 0
    eval_outputs: list[list[int]] = []
    for prompt in eval_prompts:
        tokens, generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        eval_outputs.append(tokens)
        eval_generated_tokens += generated
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    after = snapshot_replacements(replacements)
    modules = replacement_delta(after, before)
    logical_cold_bytes = sum(
        int(metrics["logical_cold_bytes"]) for metrics in modules.values()
    )
    efficiency = compute_repair_efficiency(
        generated_tokens=eval_generated_tokens,
        logical_cold_bytes=logical_cold_bytes,
        managed_weight_bytes=managed_weight_bytes,
        full_model_weight_bytes=full_model_bytes,
    )

    exact_match = eval_outputs == exact_eval
    peak_device_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    aggregate_vectors = sum(int(item["vectors"]) for item in modules.values())
    aggregate_fast = sum(
        int(item["fast_vectors"]) for item in modules.values()
    )
    report = {
        "evidence_level": "E1 real-operation falsification",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "max_new_tokens": args.max_new_tokens,
        "build_prompts": build_prompts,
        "eval_prompts": eval_prompts,
        "build_generated_tokens": build_generated_tokens,
        "eval_generated_tokens": eval_generated_tokens,
        "exact_token_match": exact_match,
        "elapsed_seconds": elapsed,
        "seconds_per_generated_token": (
            elapsed / max(1, eval_generated_tokens)
        ),
        "peak_device_bytes": peak_device_bytes,
        "physical_weight_residency_note": (
            "The Transformers model remains resident. Cold bytes are logical "
            "operator reads, not yet measured storage or PCIe traffic."
        ),
        "aggregate": {
            "vectors": aggregate_vectors,
            "fast_vectors": aggregate_fast,
            "fast_fraction": aggregate_fast / max(1, aggregate_vectors),
            "capsule_bytes": sum(
                int(item["capsule_bytes"]) for item in modules.values()
            ),
            "rank_growth": sum(
                int(item["rank_growth"]) for item in modules.values()
            ),
        },
        "repair_efficiency": efficiency.to_dict(),
        "modules": modules,
        "promotion_threshold": {
            "minimum_tokens_per_full_repair_equivalent": 600.0,
            "pass": (
                exact_match
                and efficiency.tokens_per_full_repair_equivalent >= 600.0
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(
        json.dumps(
            {
                "exact_token_match": exact_match,
                "fast_fraction": report["aggregate"]["fast_fraction"],
                "tokens_per_full_repair_equivalent": (
                    efficiency.tokens_per_full_repair_equivalent
                ),
                "promotion_pass": report["promotion_threshold"]["pass"],
            },
            indent=2,
        )
    )

    if not exact_match:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
