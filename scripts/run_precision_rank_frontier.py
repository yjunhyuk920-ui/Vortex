from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.rank_frontier import (
    maximum_feasible_rank,
    rank_budget_point,
)


DEFAULT_POINTS = (
    (72, 8),
    (88, 6),
    (96, 6),
    (112, 4),
    (128, 4),
    (136, 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact-token coverage after injecting symmetric packed-"
            "capsule quantization error at the 405B-feasible 8/6/4-bit rank "
            "frontier."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument(
        "--point",
        action="append",
        help="rank:bits, repeatable; defaults to the fixed frontier points",
    )
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("precision_rank_frontier.json"),
    )
    return parser.parse_args()


def parse_points(values: list[str] | None) -> list[tuple[int, int]]:
    if not values:
        return list(DEFAULT_POINTS)
    points: list[tuple[int, int]] = []
    for value in values:
        try:
            rank_text, bits_text = value.split(":", maxsplit=1)
            rank = int(rank_text)
            bits = int(bits_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid precision-rank point: {value!r}") from exc
        if rank <= 0 or not 2 <= bits <= 16:
            raise ValueError(f"invalid precision-rank point: {value!r}")
        points.append((rank, bits))
    return sorted(set(points), key=lambda item: (item[1], item[0]), reverse=True)


def _run_point(
    *,
    rank: int,
    bits: int,
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    budget = rank_budget_point(rank, capsule_bits=bits)
    if not budget.pass_all:
        raise RuntimeError(
            f"rank {rank} at {bits} bits violates the fixed 405B envelope: "
            f"{budget.to_dict()}"
        )

    output = directory / f"candidate-rank-{rank}-bits-{bits}.json"
    command = [
        sys.executable,
        "scripts/run_hot_candidate_coverage.py",
        "--model",
        args.model,
        "--device",
        args.device,
        "--tokens",
        str(args.tokens),
        "--build-new-tokens",
        str(args.build_new_tokens),
        "--max-rank",
        str(rank),
        "--capsule-bits",
        str(bits),
        "--eval-prompt",
        args.eval_prompt,
        "--output",
        str(output),
    ]
    for prompt in args.build_prompts or ():
        command.extend(("--build-prompt", prompt))
    for suffix in args.suffixes or ():
        command.extend(("--suffix", suffix))

    started = time.perf_counter()
    subprocess.run(command, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    first = result.get("first_divergence")
    first_rank = None if first is None else int(first["exact_token_rank"])
    top32 = float(result["coverage_at_k"]["32"])
    qualifies = (
        (first_rank is None or first_rank <= 32)
        and top32 >= 0.95
    )
    quantization = result["capsule_quantization"]["aggregate"]
    return {
        "rank": rank,
        "capsule_bits": bits,
        "budget": budget.to_dict(),
        "built_rank_statistics": result["built_rank_statistics"],
        "quantization": quantization,
        "evaluated_tokens": int(result["evaluated_tokens"]),
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": result["coverage_at_k"],
        "rank_statistics": result["rank_statistics"],
        "first_divergence": first,
        "qualifies_for_multi_hypothesis": qualifies,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.build_new_tokens <= 0:
        raise SystemExit("token counts must be positive")
    try:
        requested_points = parse_points(args.point)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    precision_maxima = {
        str(bits): maximum_feasible_rank(
            capsule_bits=bits,
            step=8,
            maximum_rank=256,
        )
        for bits in sorted({bits for _, bits in requested_points}, reverse=True)
    }
    for rank, bits in requested_points:
        maximum = precision_maxima[str(bits)]
        if rank > maximum:
            raise SystemExit(
                f"rank {rank} exceeds the aligned {bits}-bit maximum {maximum}"
            )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vortex-precision-frontier-") as temp:
        directory = Path(temp)
        points = [
            _run_point(
                rank=rank,
                bits=bits,
                args=args,
                directory=directory,
            )
            for rank, bits in requested_points
        ]

    survivors = [
        point for point in points if point["qualifies_for_multi_hypothesis"]
    ]
    best = max(
        points,
        key=lambda point: (
            float(point["coverage_at_k"]["32"]),
            float(point["exact_top1_match_rate"]),
            -float(point["rank_statistics"]["mean"]),
        ),
    )
    result = {
        "evidence_level": "E1 quantized exact-prefix precision-rank frontier",
        "model": args.model,
        "device": args.device,
        "tokens_per_point": args.tokens,
        "precision_aligned_maximum_ranks": precision_maxima,
        "tested_points": [
            {"rank": rank, "capsule_bits": bits}
            for rank, bits in requested_points
        ],
        "quantization_contract": (
            "U and WU are symmetrically quantized per response-basis column. "
            "The packed payload and fp16 column scales are charged to storage "
            "and traffic; values are dequantized before the existing matmul to "
            "isolate representation error from packed-kernel performance."
        ),
        "same_context_contract": (
            "Every point is evaluated with exact and quantized-hot logits on "
            "the same authoritative exact prefix at every token position."
        ),
        "decision_rule": (
            "advance only when the point fits memory/traffic/compute, first-"
            "divergence exact-token rank is at most 32, and top-32 coverage is "
            "at least 0.95"
        ),
        "points": points,
        "best_observed_point": best,
        "surviving_points": survivors,
        "decision": (
            "advance quantized multi-hypothesis capsule"
            if survivors
            else "reject tested quantized global capsule frontier"
        ),
        "next_candidate": (
            "combine the lowest-cost surviving capsule with the exact-prompt "
            "session response basis"
            if survivors
            else (
                "combine the best quantized global prior with prompt-specific "
                "session directions, then test block-local adaptive growth"
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "precision_aligned_maximum_ranks": precision_maxima,
                "points": [
                    {
                        "rank": point["rank"],
                        "bits": point["capsule_bits"],
                        "top1": point["exact_top1_match_rate"],
                        "top32": point["coverage_at_k"]["32"],
                        "mean_exact_rank": point["rank_statistics"]["mean"],
                        "qualifies": point["qualifies_for_multi_hypothesis"],
                    }
                    for point in points
                ],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
