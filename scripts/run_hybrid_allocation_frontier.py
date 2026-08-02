from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_POINTS = (
    (56, 96, 6),
    (64, 136, 4),
    (88, 136, 4),
    (96, 136, 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep causal global-plus-exact-prompt hybrid response-basis rank "
            "allocations inside the fixed 405B precision-rank envelope."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument(
        "--point",
        action="append",
        help="global_rank:total_rank:bits, repeatable",
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
        default=Path("hybrid_allocation_frontier.json"),
    )
    return parser.parse_args()


def parse_points(values: list[str] | None) -> list[tuple[int, int, int]]:
    if not values:
        return list(DEFAULT_POINTS)
    points: set[tuple[int, int, int]] = set()
    for value in values:
        try:
            global_text, total_text, bits_text = value.split(":", maxsplit=2)
            global_rank = int(global_text)
            total_rank = int(total_text)
            bits = int(bits_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid hybrid point: {value!r}") from exc
        if (
            global_rank <= 0
            or total_rank < global_rank
            or not 2 <= bits <= 16
        ):
            raise ValueError(f"invalid hybrid point: {value!r}")
        points.add((global_rank, total_rank, bits))
    return sorted(points, key=lambda item: (item[2], item[1], item[0]), reverse=True)


def _run_point(
    *,
    global_rank: int,
    total_rank: int,
    bits: int,
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    output = directory / (
        f"hybrid-global-{global_rank}-total-{total_rank}-bits-{bits}.json"
    )
    command = [
        sys.executable,
        "scripts/run_hybrid_session_candidate_coverage.py",
        "--model",
        args.model,
        "--device",
        args.device,
        "--tokens",
        str(args.tokens),
        "--build-new-tokens",
        str(args.build_new_tokens),
        "--global-rank",
        str(global_rank),
        "--total-rank",
        str(total_rank),
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
    return {
        "global_rank_limit": global_rank,
        "total_rank_limit": total_rank,
        "capsule_bits": bits,
        "global_rank_statistics": result["global_rank_statistics"],
        "added_session_rank_statistics": result[
            "added_session_rank_statistics"
        ],
        "final_rank_statistics": result["final_rank_statistics"],
        "prompt_reconstruction": result["prompt_reconstruction"],
        "quantization": result["quantization"]["aggregate"],
        "budget": result["budget"],
        "exact_top1_match_rate": result["exact_top1_match_rate"],
        "coverage_at_k": result["coverage_at_k"],
        "rank_statistics": result["rank_statistics"],
        "first_divergence": result["first_divergence"],
        "qualifies_for_multi_hypothesis": (
            result["decision"] == "advance hybrid multi-hypothesis certificate"
        ),
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

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vortex-hybrid-frontier-") as temp:
        directory = Path(temp)
        points = [
            _run_point(
                global_rank=global_rank,
                total_rank=total_rank,
                bits=bits,
                args=args,
                directory=directory,
            )
            for global_rank, total_rank, bits in requested_points
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
        "evidence_level": "E1 causal quantized hybrid allocation frontier",
        "model": args.model,
        "device": args.device,
        "tokens_per_point": args.tokens,
        "tested_points": [
            {
                "global_rank": global_rank,
                "total_rank": total_rank,
                "capsule_bits": bits,
            }
            for global_rank, total_rank, bits in requested_points
        ],
        "causal_contract": (
            "The generic global prior uses only fixed disjoint build prompts. "
            "Session residual directions use only exact user-prompt prefill "
            "inputs and outputs. Unseen continuation targets are evaluation-only."
        ),
        "decision_rule": (
            "advance only when the fixed 405B budget passes, first-divergence "
            "exact-token rank is at most 32, and top-32 coverage is at least 0.95"
        ),
        "points": points,
        "best_observed_point": best,
        "surviving_points": survivors,
        "decision": (
            "advance hybrid allocation"
            if survivors
            else "reject tested global-plus-session allocations"
        ),
        "next_candidate": (
            "build the causal certificate at the lowest-cost surviving allocation"
            if survivors
            else (
                "allocate prompt residual directions non-uniformly by module "
                "benefit per byte, then test online state-conditional capsules"
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
                "points": [
                    {
                        "global": point["global_rank_limit"],
                        "total": point["total_rank_limit"],
                        "bits": point["capsule_bits"],
                        "actual_final_rank": point[
                            "final_rank_statistics"
                        ]["mean"],
                        "top1": point["exact_top1_match_rate"],
                        "top32": point["coverage_at_k"]["32"],
                        "mean_exact_rank": point["rank_statistics"]["mean"],
                        "qualifies": point[
                            "qualifies_for_multi_hypothesis"
                        ],
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
