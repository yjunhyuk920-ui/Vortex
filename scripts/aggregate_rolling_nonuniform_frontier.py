from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate causal rolling nonuniform trajectory measurements."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    if not points:
        raise SystemExit("at least one rolling point is required")
    points.sort(key=lambda point: int(point["refresh_interval"]))

    final_survivors = [point for point in points if point["final_gate_compatible"]]
    lower_bound_survivors = [point for point in points if point["lower_bound_qualifies"]]
    accuracy_survivors = [point for point in points if point["accuracy_pass"]]
    best = max(
        points,
        key=lambda point: (
            float(point["hot_only_coverage_at_k"]["32"]),
            float(point["hot_only_top1_match_rate"]),
            int(point["refresh_interval"]),
        ),
    )
    minimum_managed_interval = max(
        int(point["refresh_budget"]["managed_o_down_lower_bound"]["minimum_integer_interval"])
        for point in points
    )
    minimum_full_interval = max(
        int(point["refresh_budget"]["full_model_anchor"]["minimum_integer_interval"])
        for point in points
    )
    output = {
        "evidence_level": "E1 causal rolling nonuniform trajectory frontier",
        "model": points[0]["model"],
        "causal_contract": points[0]["causal_contract"],
        "candidate_commit_contract": points[0]["candidate_commit_contract"],
        "decision_rule": points[0]["decision_rule"],
        "minimum_refresh_intervals": {
            "managed_o_down_optimistic_lower_bound": minimum_managed_interval,
            "full_model_anchor": minimum_full_interval,
        },
        "points": points,
        "best_observed_point": best,
        "accuracy_surviving_intervals": [
            point["refresh_interval"] for point in accuracy_survivors
        ],
        "lower_bound_surviving_intervals": [
            point["refresh_interval"] for point in lower_bound_survivors
        ],
        "final_surviving_intervals": [
            point["refresh_interval"] for point in final_survivors
        ],
        "decision": (
            "advance rolling nonuniform trajectory dictionary"
            if final_survivors
            else "reject tested all-module rolling refresh frontier"
        ),
        "next_candidate": (
            "validate surviving cadence across prompt families and larger models"
            if final_survivors
            else (
                "replace synchronized all-module refresh with sparse event-driven "
                "module-local updates, charging exact bytes and arithmetic only "
                "for modules whose causal novelty certificate fires"
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
