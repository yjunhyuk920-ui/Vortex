from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate nonuniform-rank point measurements into one frontier."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points.append(payload)
    if not points:
        raise SystemExit("at least one point result is required")

    points.sort(
        key=lambda point: (
            int(point["uniform_equivalent"]["capsule_bits"]),
            int(point["uniform_equivalent"]["rank"]),
        ),
        reverse=True,
    )
    survivors = [point for point in points if point["qualifies"]]
    best = max(
        points,
        key=lambda point: (
            float(point["coverage_at_k"]["32"]),
            float(point["exact_top1_match_rate"]),
            -float(point["allocation"]["used_bytes"]),
        ),
    )
    output = {
        "evidence_level": "E1 causal prompt-valued nonuniform rank frontier",
        "model": points[0]["model"],
        "causal_contract": points[0]["causal_contract"],
        "allocation_contract": points[0]["allocation_contract"],
        "decision_rule": points[0]["decision_rule"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": [
            {
                "rank": point["uniform_equivalent"]["rank"],
                "capsule_bits": point["uniform_equivalent"]["capsule_bits"],
                "top1": point["exact_top1_match_rate"],
                "top32": point["coverage_at_k"]["32"],
            }
            for point in survivors
        ],
        "decision": (
            "advance prompt-valued nonuniform rank allocation"
            if survivors
            else "reject tested prompt-valued nonuniform rank frontier"
        ),
        "next_candidate": (
            "validate surviving allocation across disjoint prompt families"
            if survivors
            else (
                "use nonuniform module ranks inside an online rolling local "
                "trajectory dictionary refreshed only from certified prefixes"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
