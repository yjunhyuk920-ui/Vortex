from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import require_transformers
from vortex_runtime.mlp_functional_skeleton import (
    compile_swiglu_functional_skeleton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one checkpoint MLP into a functional skeleton and report "
            "function-space errors without running autoregressive generation."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-index", type=int, default=10)
    parser.add_argument("--prototypes", type=int, required=True)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--heldout-probe-count", type=int, default=128)
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8081)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlp_function_only_diagnostic.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.prototypes,
        args.probe_count,
        args.heldout_probe_count,
        args.factor_bits,
    ) <= 0:
        raise SystemExit("positive controls are required")

    AutoModelForCausalLM, _ = require_transformers()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    model.eval()
    layers = model.model.layers
    if not 0 <= args.layer_index < len(layers):
        raise SystemExit("layer index out of range")
    mlp = layers[args.layer_index].mlp

    _, skeleton_stats, gauge_stats = compile_swiglu_functional_skeleton(
        gate_proj=mlp.gate_proj,
        up_proj=mlp.up_proj,
        down_proj=mlp.down_proj,
        prototypes=args.prototypes,
        probe_count=args.probe_count,
        heldout_probe_count=args.heldout_probe_count,
        factor_bits=args.factor_bits,
        ridge=1e-5,
        seed=args.seed,
    )
    payload = {
        "evidence_level": "E1 single-layer SwiGLU function-space diagnostic",
        "model": args.model,
        "layer_index": args.layer_index,
        "prototypes": args.prototypes,
        "factor_bits": args.factor_bits,
        "functional_skeleton": skeleton_stats.to_dict(),
        "gauge_normalization": gauge_stats.to_dict(),
        "decision": (
            "function space is promising"
            if skeleton_stats.heldout_output_relative_l2_error <= 0.2
            else "tested function space is not low-rank enough"
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
