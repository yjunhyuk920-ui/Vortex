from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.atlas_linear import OnlineAtlasLinear


DEFAULT_BUILD_PROMPTS = [
    "Explain why the sky appears blue in clear weather.",
    "Write a Python function that merges two sorted lists.",
]
DEFAULT_EVAL_PROMPTS = [
    "Explain why sunsets often appear red.",
    "Write a Python function that finds duplicates in a list.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure OnlineAtlasLinear rank and cold-read behavior on real "
            "Hugging Face model activation traces."
        )
    )
    parser.add_argument("--model", required=True, help="HF repo ID or local model path")
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
        help="Module suffix to trace; defaults to self_attn.o_proj and mlp.down_proj",
    )
    parser.add_argument("--output", type=Path, default=Path("real_model_atlas_trace.json"))
    parser.add_argument(
        "--verify-every",
        type=int,
        default=0,
        help="Compare every Nth atlas result with dense module output; 0 disables",
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


def matching_modules(model: torch.nn.Module, suffixes: tuple[str, ...]) -> dict[str, torch.nn.Linear]:
    result: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            name.endswith(suffix) for suffix in suffixes
        ):
            result[name] = module
    if not result:
        raise RuntimeError(f"no linear modules matched suffixes: {suffixes}")
    return result


def capture_prompt(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    modules: dict[str, torch.nn.Linear],
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, torch.Tensor]:
    traces: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []

    for name, module in modules.items():
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], *, key: str = name) -> None:
            value = args[0].detach().to("cpu", dtype=torch.float32)
            traces[key].append(value.reshape(-1, value.shape[-1]))

        handles.append(module.register_forward_pre_hook(hook))

    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()

    return {name: torch.cat(parts, dim=0) for name, parts in traces.items() if parts}


def process_trace(
    *,
    traces: dict[str, torch.Tensor],
    modules: dict[str, torch.nn.Linear],
    atlases: dict[str, OnlineAtlasLinear],
    max_rank: int,
    verify_every: int,
) -> dict[str, int]:
    vectors = 0
    verified = 0
    for name, matrix in traces.items():
        module = modules[name]
        atlas = atlases.get(name)
        if atlas is None:
            out_features, in_features = module.weight.shape
            atlas = OnlineAtlasLinear(
                in_features=in_features,
                out_features=out_features,
                weight_loader=lambda linear=module: linear.weight.detach().to(
                    "cpu", dtype=torch.float32
                ),
                max_rank=max_rank,
                atol=1e-6,
                rtol=1e-5,
            )
            atlases[name] = atlas
        for row_index, row in enumerate(matrix):
            actual = atlas(row)
            vectors += 1
            if verify_every > 0 and row_index % verify_every == 0:
                expected = module.weight.detach().to("cpu", dtype=torch.float32) @ row
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                verified += 1
    return {"vectors": vectors, "verified": verified}


def snapshot(atlases: dict[str, OnlineAtlasLinear]) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            **atlas.stats.to_dict(),
            "rank": atlas.rank,
            "capsule_bytes": atlas.capsule_bytes,
            "in_features": atlas.in_features,
            "out_features": atlas.out_features,
        }
        for name, atlas in atlases.items()
    }


def delta(after: dict[str, dict[str, int | float]], before: dict[str, dict[str, int | float]]) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, current in after.items():
        previous = before.get(name, {})
        modules[name] = {
            "rank": current["rank"],
            "capsule_bytes": current["capsule_bytes"],
            "eval_vectors": int(current["vectors"]) - int(previous.get("vectors", 0)),
            "eval_fast_vectors": int(current["fast_vectors"]) - int(previous.get("fast_vectors", 0)),
            "eval_cold_vectors": int(current["cold_vectors"]) - int(previous.get("cold_vectors", 0)),
            "eval_cold_weight_reads": int(current["cold_weight_reads"]) - int(previous.get("cold_weight_reads", 0)),
            "eval_weight_bytes_read": int(current["weight_bytes_read"]) - int(previous.get("weight_bytes_read", 0)),
        }
        modules[name]["eval_fast_fraction"] = modules[name]["eval_fast_vectors"] / max(
            1, modules[name]["eval_vectors"]
        )
    return modules


def main() -> None:
    args = parse_args()
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
    modules = matching_modules(model, suffixes)
    atlases: dict[str, OnlineAtlasLinear] = {}
    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    eval_prompts = args.eval_prompts or DEFAULT_EVAL_PROMPTS

    build_counts = {"vectors": 0, "verified": 0}
    for prompt in build_prompts:
        traces = capture_prompt(
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        counts = process_trace(
            traces=traces,
            modules=modules,
            atlases=atlases,
            max_rank=args.max_rank,
            verify_every=args.verify_every,
        )
        build_counts = {key: build_counts[key] + counts[key] for key in build_counts}

    build_snapshot = snapshot(atlases)
    eval_counts = {"vectors": 0, "verified": 0}
    for prompt in eval_prompts:
        traces = capture_prompt(
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        counts = process_trace(
            traces=traces,
            modules=modules,
            atlases=atlases,
            max_rank=args.max_rank,
            verify_every=args.verify_every,
        )
        eval_counts = {key: eval_counts[key] + counts[key] for key in eval_counts}

    final_snapshot = snapshot(atlases)
    eval_delta = delta(final_snapshot, build_snapshot)
    total_eval_vectors = sum(item["eval_vectors"] for item in eval_delta.values())
    total_eval_fast = sum(item["eval_fast_vectors"] for item in eval_delta.values())
    total_eval_cold_reads = sum(item["eval_cold_weight_reads"] for item in eval_delta.values())

    report = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "suffixes": list(suffixes),
        "matched_modules": len(modules),
        "max_rank": args.max_rank,
        "max_new_tokens": args.max_new_tokens,
        "build_prompts": build_prompts,
        "eval_prompts": eval_prompts,
        "build_counts": build_counts,
        "eval_counts": eval_counts,
        "aggregate_eval": {
            "vectors": total_eval_vectors,
            "fast_vectors": total_eval_fast,
            "fast_fraction": total_eval_fast / max(1, total_eval_vectors),
            "cold_weight_reads": total_eval_cold_reads,
            "cold_streams_per_vector": total_eval_cold_reads / max(1, total_eval_vectors),
            "capsule_bytes": sum(atlas.capsule_bytes for atlas in atlases.values()),
        },
        "modules": eval_delta,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report["aggregate_eval"], indent=2))


if __name__ == "__main__":
    main()
