from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_full_rank_precision_point import (
    common_prefix_length,
    greedy_tokens,
    teacher_forced_logits,
)
from scripts.run_kronecker_operator_point import teacher_summary
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.mlp_functional_skeleton import (
    compile_swiglu_functional_skeleton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one TinyLlama MLP with a functional skeleton and measure "
            "local layer sensitivity without a causal tree."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--prototypes", type=int, default=32)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--heldout-probe-count", type=int, default=64)
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7079)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlp_skeleton_layer_diagnostic.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.prototypes,
        args.probe_count,
        args.heldout_probe_count,
        args.factor_bits,
        args.tokens,
    ) <= 0:
        raise SystemExit("positive controls are required")

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    layers = model.model.layers
    if not 0 <= args.layer_index < len(layers):
        raise SystemExit("layer index out of range")
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)

    started = time.perf_counter()
    exact_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    exact_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )

    original = layers[args.layer_index].mlp
    compiled, skeleton_stats, gauge_stats = (
        compile_swiglu_functional_skeleton(
            gate_proj=original.gate_proj,
            up_proj=original.up_proj,
            down_proj=original.down_proj,
            prototypes=args.prototypes,
            probe_count=args.probe_count,
            heldout_probe_count=args.heldout_probe_count,
            factor_bits=args.factor_bits,
            ridge=1e-5,
            seed=args.seed + args.layer_index * 1009,
        )
    )
    layers[args.layer_index].mlp = compiled

    compiled_logits = teacher_forced_logits(
        model=model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    compiled_tokens = greedy_tokens(
        model=model,
        encoded=encoded,
        count=args.tokens,
    ).to("cpu")
    payload = {
        "evidence_level": "E2 single-layer functional skeleton sensitivity",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "layer_index": args.layer_index,
        "total_layers": len(layers),
        "prototypes": args.prototypes,
        "factor_bits": args.factor_bits,
        "functional_skeleton": skeleton_stats.to_dict(),
        "gauge_normalization": gauge_stats.to_dict(),
        "teacher_forced_exact_reference": teacher_summary(
            logits=exact_logits,
            exact_tokens=exact_tokens,
        ),
        "teacher_forced_compiled": teacher_summary(
            logits=compiled_logits,
            exact_tokens=exact_tokens,
        ),
        "autoregressive_exact_prefix": common_prefix_length(
            compiled_tokens,
            exact_tokens,
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
