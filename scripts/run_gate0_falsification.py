from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.gated_projected_linear import GatedProjectedLinear, activation_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace real HF linear operations and measure Gate 0 repair rate."
    )
    parser.add_argument("model", help="Hugging Face model id or local path")
    parser.add_argument("--build-prompts", type=Path, required=True)
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("gate0_falsification.json"))
    parser.add_argument("--gate0-budget", type=Path, default=Path("gate0_budget.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-samples-per-module", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--include-regex",
        default=r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    parser.add_argument("--exclude-regex", default=r"(?!)")
    parser.add_argument("--minimum-token-agreement", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {path}")
    if text.startswith("["):
        payload = json.loads(text)
        prompts = [str(x["prompt"] if isinstance(x, dict) else x) for x in payload]
    else:
        prompts = []
        for line in text.splitlines():
            item = json.loads(line)
            prompts.append(str(item["prompt"] if isinstance(item, dict) else item))
    if not prompts or any(not p.strip() for p in prompts):
        raise ValueError(f"invalid prompts in {path}")
    return prompts


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def selected_linears(
    model: nn.Module, include: re.Pattern[str], exclude: re.Pattern[str]
) -> dict[str, nn.Linear]:
    result = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and include.search(name)
        and not exclude.search(name)
    }
    if not result:
        raise ValueError("no linear modules matched the include/exclude expressions")
    return result


def collect_samples(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    modules: dict[str, nn.Linear],
    device: torch.device,
    max_samples: int,
) -> dict[str, torch.Tensor]:
    buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    handles = []

    for name, module in modules.items():
        def capture(_module: nn.Module, inputs: tuple[Any, ...], key: str = name) -> None:
            if counts[key] >= max_samples or not inputs:
                return
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                return
            rows = value.detach().reshape(-1, value.shape[-1]).to("cpu", torch.float32)
            remaining = max_samples - counts[key]
            if rows.shape[0] > remaining:
                index = torch.linspace(0, rows.shape[0] - 1, remaining).long()
                rows = rows[index]
            buffers[key].append(rows)
            counts[key] += rows.shape[0]

        handles.append(module.register_forward_pre_hook(capture))

    try:
        with torch.inference_mode():
            for prompt in prompts:
                batch = tokenizer(prompt, return_tensors="pt")
                kwargs = {key: value.to(device) for key, value in batch.items()}
                model(**kwargs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in modules if not buffers[name]]
    if missing:
        raise RuntimeError(f"no activations captured for {missing[:5]}")
    return {name: torch.cat(parts, dim=0) for name, parts in buffers.items()}


def replace_linears(
    model: nn.Module,
    modules: dict[str, nn.Linear],
    samples: dict[str, torch.Tensor],
    rank: int,
    epsilon: float,
) -> dict[str, GatedProjectedLinear]:
    replacements: dict[str, GatedProjectedLinear] = {}
    for name in sorted(modules, key=lambda x: x.count("."), reverse=True):
        linear = modules[name]
        basis = activation_basis(samples[name], min(rank, linear.in_features))
        replacement = GatedProjectedLinear.from_linear(
            linear, basis, epsilon=epsilon, offload_exact=True
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
        replacements[name] = replacement
    return replacements


def reset_stats(modules: dict[str, GatedProjectedLinear]) -> None:
    for module in modules.values():
        module.reset_stats()


def aggregate_stats(modules: dict[str, GatedProjectedLinear]) -> dict[str, Any]:
    return {
        "rows": sum(m.stats.rows for m in modules.values()),
        "fast_rows": sum(m.stats.fast_rows for m in modules.values()),
        "slow_rows": sum(m.stats.slow_rows for m in modules.values()),
        "cold_weight_reads": sum(m.stats.cold_weight_reads for m in modules.values()),
        "cold_weight_bytes": sum(m.stats.cold_weight_bytes for m in modules.values()),
        "capsule_bytes_read": sum(m.stats.capsule_bytes_read for m in modules.values()),
        "capsule_bytes": sum(m.capsule_bytes for m in modules.values()),
        "managed_weight_bytes": sum(m.exact_weight_bytes for m in modules.values()),
        "per_module": {
            name: {
                **module.stats.to_dict(),
                "rank": module.rank,
                "capsule_bytes": module.capsule_bytes,
                "exact_weight_bytes": module.exact_weight_bytes,
            }
            for name, module in modules.items()
        },
    }


def manual_greedy(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    projections: dict[str, GatedProjectedLinear] | None = None,
) -> dict[str, Any]:
    batch = tokenizer(prompt, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)

    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    sync(device)
    prefill_ms = (time.perf_counter() - started) * 1000

    current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [int(current.item())]
    past = output.past_key_values
    measured_tokens = max(0, max_new_tokens - 1)
    if projections is not None:
        reset_stats(projections)

    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(measured_tokens):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(current.item()))
            past = output.past_key_values
    sync(device)
    decode_ms = (time.perf_counter() - started) * 1000

    return {
        "tokens": generated[:max_new_tokens],
        "text": tokenizer.decode(generated[:max_new_tokens], skip_special_tokens=False),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "measured_decode_tokens": measured_tokens,
        "ms_per_measured_decode_token": decode_ms / measured_tokens if measured_tokens else None,
        "projection_stats": None if projections is None else aggregate_stats(projections),
    }


def main() -> None:
    args = parse_args()
    if args.rank <= 0 or args.max_samples_per_module <= 0 or args.max_new_tokens <= 0:
        raise ValueError("rank, sample count, and max_new_tokens must be positive")
    if args.epsilon < 0 or not 0 <= args.minimum_token_agreement <= 1:
        raise ValueError("invalid epsilon or agreement threshold")

    build_prompts = load_prompts(args.build_prompts)
    eval_prompts = load_prompts(args.eval_prompts)
    if set(build_prompts) & set(eval_prompts):
        raise ValueError("build and evaluation prompts must be disjoint")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install optional dependency: pip install transformers") from exc

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    total_weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    selected = selected_linears(
        model, re.compile(args.include_regex), re.compile(args.exclude_regex)
    )
    samples = collect_samples(
        model,
        tokenizer,
        build_prompts,
        selected,
        device,
        args.max_samples_per_module,
    )
    baseline_runs = [
        manual_greedy(model, tokenizer, prompt, device, args.max_new_tokens)
        for prompt in eval_prompts
    ]

    replacements = replace_linears(model, selected, samples, args.rank, args.epsilon)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    projected_runs = [
        manual_greedy(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            replacements,
        )
        for prompt in eval_prompts
    ]

    token_total = token_matches = measured_tokens = cold_bytes = capsule_reads = 0
    for baseline, projected in zip(baseline_runs, projected_runs, strict=True):
        width = min(len(baseline["tokens"]), len(projected["tokens"]))
        token_total += width
        token_matches += sum(
            int(a == b)
            for a, b in zip(
                baseline["tokens"][:width], projected["tokens"][:width], strict=True
            )
        )
        measured_tokens += projected["measured_decode_tokens"]
        cold_bytes += projected["projection_stats"]["cold_weight_bytes"]
        capsule_reads += projected["projection_stats"]["capsule_bytes_read"]

    managed_weight_bytes = sum(m.exact_weight_bytes for m in replacements.values())
    stream_equivalents = cold_bytes / max(1, total_weight_bytes)
    stream_equivalents_per_token = stream_equivalents / max(1, measured_tokens)
    observed_a = math.inf if stream_equivalents_per_token == 0 else 1 / stream_equivalents_per_token
    token_agreement = token_matches / max(1, token_total)

    budget = json.loads(args.gate0_budget.read_text(encoding="utf-8"))
    required_a = float(
        budget["falsification_thresholds"]["minimum_amortized_tokens_per_full_stream"]
    )
    repair_pass = observed_a >= required_a
    quality_pass = token_agreement >= args.minimum_token_agreement

    result = {
        "evidence_level": "E2-candidate-harness" if quality_pass else "E1-failed-E2-attempt",
        "status": "promote_to_next_gate" if repair_pass and quality_pass else "reject_or_revise",
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "configuration": {
            "rank": args.rank,
            "epsilon": args.epsilon,
            "max_samples_per_module": args.max_samples_per_module,
            "max_new_tokens": args.max_new_tokens,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "minimum_token_agreement": args.minimum_token_agreement,
        },
        "prompt_split": {
            "build_count": len(build_prompts),
            "eval_count": len(eval_prompts),
            "disjoint": True,
        },
        "coverage": {
            "replaced_linear_modules": len(replacements),
            "managed_weight_bytes": managed_weight_bytes,
            "total_model_weight_bytes": total_weight_bytes,
            "managed_weight_fraction": managed_weight_bytes / max(1, total_weight_bytes),
            "capsule_bytes_fp_runtime": sum(m.capsule_bytes for m in replacements.values()),
        },
        "quality": {
            "token_matches": token_matches,
            "token_total": token_total,
            "token_agreement": token_agreement,
            "required_token_agreement": args.minimum_token_agreement,
            "passes": quality_pass,
        },
        "repair": {
            "measured_decode_tokens": measured_tokens,
            "cold_weight_bytes": cold_bytes,
            "hot_capsule_bytes_read": capsule_reads,
            "full_model_stream_equivalents": stream_equivalents,
            "full_stream_equivalents_per_token": stream_equivalents_per_token,
            "observed_amortized_tokens_per_full_stream": observed_a,
            "required_amortized_tokens_per_full_stream": required_a,
            "passes": repair_pass,
        },
        "baseline_runs": baseline_runs,
        "projected_runs": projected_runs,
        "limitations": [
            "The real-operation harness precedes 3-bit image and 8-bit basis kernels.",
            "Attention token-axis compression is budgeted but not implemented here.",
            "Passing is necessary but not sufficient for CUDA, VRAM, and 405B gates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
