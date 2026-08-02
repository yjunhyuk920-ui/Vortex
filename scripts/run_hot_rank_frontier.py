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


DEFAULT_RANKS = (32, 40, 48, 56, 64, 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact-prefix top-K coverage diagnostic across capsule "
            "ranks that fit the fixed 405B Gate 0 memory, hot-traffic, and "
            "hot-compute envelope."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--build-new-tokens", type=int, default=1)
    parser.add_argument("--rank", type=int, action="append", dest="ranks")
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hot_rank_frontier.json"),
    )
    return parser.parse_args()


def _run_rank(
    *,
    rank: int,
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    output = directory / f"candidate-coverage-rank-{rank}.json"
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
    budget = rank_budget_point(rank)
    first = result.get("first_divergence")
    first_rank = None if first is None else int(first["exact_token_rank"])
    top32 = float(result["coverage_at_k"]["32"])
    qualifies = (
        budget.pass_all
        and (first_rank is None or first_rank <= 32)
        and top32 >= 0.95
    )
    return {
        "rank": rank,
        "budget": budget.to_dict(),
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
    ranks = sorted(set(args.ranks or DEFAULT_RANKS))
    if any(rank <= 0 for rank in ranks):
        raise SystemExit("ranks must be positive")

    fixed_maximum = maximum_feasible_rank(step=8, maximum_rank=256)
    if max(ranks) > fixed_maximum:
        raise SystemExit(
            f"rank {max(ranks)} exceeds the fixed aligned Gate 0 maximum "
            f"of {fixed_maximum}"
        )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vortex-rank-frontier-") as temp:
        directory = Path(temp)
        points = [
            _run_rank(rank=rank, args=args, directory=directory)
            for rank in ranks
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
    top32_values = [float(point["coverage_at_k"]["32"]) for point in points]
    monotonic_top32 = all(
        right >= left
        for left, right in zip(top32_values, top32_values[1:])
    )

    result = {
        "evidence_level": "E1 exact-prefix feasible-rank frontier diagnostic",
        "model": args.model,
        "device": args.device,
        "tokens_per_rank": args.tokens,
        "tested_ranks": ranks,
        "fixed_405b_aligned_maximum_rank": fixed_maximum,
        "binding_budget": "hot capsule traffic per token",
        "same_context_contract": (
            "Every rank is measured with exact and hot logits on the same "
            "authoritative exact prefix at every token position."
        ),
        "decision_rule": {
            "advance": (
                "rank fits memory/traffic/compute, first divergence exact-token "
                "rank is at most 32, and top-32 coverage is at least 0.95"
            ),
            "reject_family": (
                "no tested rank through the maximum 405B-feasible aligned rank "
                "satisfies the advance rule"
            ),
        },
        "points": points,
        "top32_coverage_monotonic": monotonic_top32,
        "best_observed_point": best,
        "surviving_points": survivors,
        "decision": (
            "advance feasible-rank multi-hypothesis certificate"
            if survivors
            else "reject global low-rank O/down capsule through rank 72"
        ),
        "next_candidate": (
            "build a multi-hypothesis certificate at the lowest surviving rank"
            if survivors
            else (
                "replace the global activation-subspace capsule with a "
                "block-local trajectory or operator-structured representation"
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
                "fixed_405b_aligned_maximum_rank": fixed_maximum,
                "tested_ranks": ranks,
                "best_observed_point": best,
                "surviving_ranks": [point["rank"] for point in survivors],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
