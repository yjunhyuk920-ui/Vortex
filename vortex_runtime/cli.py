from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch

from .hf_loader import HuggingFaceLayout
from .llama import StreamingLlama
from .progressive import ProgressiveLinear
from .toy_model import create_tiny_llama


def command_inspect(args: argparse.Namespace) -> None:
    layout = HuggingFaceLayout(args.model_dir)
    print(json.dumps(layout.summary(), indent=2, ensure_ascii=False))


def command_demo(args: argparse.Namespace) -> None:
    model_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp()) / "tiny-llama"
    create_tiny_llama(model_dir)
    model = StreamingLlama(
        model_dir,
        tensor_budget_bytes=args.budget_mb * 1024 * 1024,
        lm_head_base_bits=args.base_bits,
    )
    prompt = torch.tensor([[1, 5, 9, 12]], dtype=torch.long)
    tokens = model.generate(prompt, max_new_tokens=args.tokens)
    print(json.dumps({
        "model_dir": str(model_dir),
        "generated_tokens": tokens,
        "cache": model.cache.stats.__dict__,
        "layout": model.layout.summary(),
    }, indent=2))


def command_certify(args: argparse.Namespace) -> None:
    gen = torch.Generator().manual_seed(args.seed)
    weight = torch.randn(args.vocab, args.hidden, generator=gen)
    operator = ProgressiveLinear(
        weight,
        base_bits=args.base_bits,
        tile_cols=args.tile_cols,
    )
    fractions = []
    coarse_matches = 0
    for _ in range(args.trials):
        x = torch.randn(args.hidden, generator=gen)
        result = operator.certify_argmax(x)
        if not result.certified:
            raise RuntimeError("certification failed")
        fractions.append(result.residual_fraction_read)
        coarse_matches += int(result.coarse_token_id == result.exact_token_id)
    values = torch.tensor(fractions)
    report = {
        "trials": args.trials,
        "exact_certification_rate": 1.0,
        "coarse_top1_match_rate": coarse_matches / args.trials,
        "mean_residual_fraction_read": float(values.mean()),
        "p50_residual_fraction_read": float(values.quantile(0.50)),
        "p95_residual_fraction_read": float(values.quantile(0.95)),
        "operator_storage_bytes": operator.storage_bytes,
    }
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vortex-proto")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect an HF safetensors model")
    inspect.add_argument("model_dir")
    inspect.set_defaults(func=command_inspect)

    demo = sub.add_parser("demo", help="run a tiny streamed Llama checkpoint")
    demo.add_argument("--output-dir")
    demo.add_argument("--tokens", type=int, default=8)
    demo.add_argument("--budget-mb", type=int, default=2)
    demo.add_argument("--base-bits", type=int, default=4)
    demo.set_defaults(func=command_demo)

    certify = sub.add_parser("certify", help="benchmark exact progressive LM-head proof")
    certify.add_argument("--vocab", type=int, default=4096)
    certify.add_argument("--hidden", type=int, default=1024)
    certify.add_argument("--trials", type=int, default=32)
    certify.add_argument("--base-bits", type=int, default=4)
    certify.add_argument("--tile-cols", type=int, default=128)
    certify.add_argument("--seed", type=int, default=11)
    certify.set_defaults(func=command_certify)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
