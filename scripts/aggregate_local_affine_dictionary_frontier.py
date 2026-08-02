from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independently measured local affine dictionary points."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize_point(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "clusters": int(result["clusters"]),
        "local_rank": int(result["local_rank"]),
        "capsule_bits": int(result["capsule_bits"]),
        "captured_vectors_per_module": int(
            result["captured_vectors_per_module"]
        ),
        "stored_equivalent_rank": int(result["stored_equivalent_rank"]),
        "active_equivalent_rank": int(result["active_equivalent_rank"]),
        "stored_budget": result["stored_budget"],
        "active_budget": result["active_budget"],
        "actual_stored_response_columns": result[
            "actual_stored_response_columns"
        ],
        "actual_active_equivalent_columns": result[
            "actual_active_equivalent_columns"
        ],
        "post_quantization_training_reconstruction": result[
            "post_quantization_training_reconstruction"
        ],
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": result["coverage_at_k"],
        "rank_statistics": result["rank_statistics"],
        "first_divergence": result["first_divergence"],
        "qualifies": result["decision"] == "advance routed local affine dictionary",
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    loaded = list(results)
    if not loaded:
        raise ValueError("at least one local dictionary result is required")
    points = [summarize_point(result) for result in loaded]
    identities = {
        (point["clusters"], point["local_rank"], point["capsule_bits"])
        for point in points
    }
    if len(identities) != len(points):
        raise ValueError("duplicate local dictionary points")

    survivors = [point for point in points if point["qualifies"]]
    best = max(
        points,
        key=lambda point: (
            float(point["coverage_at_k"]["32"]),
            float(point["exact_top1_match_rate"]),
            -float(point["rank_statistics"]["mean"]),
            -int(point["active_equivalent_rank"]),
        ),
    )
    return {
        "evidence_level": "E1 causal routed local affine dictionary frontier",
        "model": loaded[0]["model"],
        "device": loaded[0]["device"],
        "evaluated_unseen_tokens_per_point": int(
            loaded[0]["evaluated_unseen_tokens"]
        ),
        "causal_contract": loaded[0]["causal_contract"],
        "budget_contract": loaded[0]["budget_contract"],
        "decision_rule": loaded[0]["decision_rule"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": survivors,
        "decision": (
            "advance routed local affine dictionary"
            if survivors
            else "reject tested routed local affine dictionary frontier"
        ),
        "next_candidate": (
            "build a sound route-and-token certificate for the lowest active-cost survivor"
            if survivors
            else (
                "make routing state-conditional across layers and allocate local "
                "ranks non-uniformly by causal prompt reconstruction benefit per byte"
            )
        ),
        "total_measurement_seconds": sum(
            float(point["elapsed_seconds"]) for point in points
        ),
    }


def main() -> None:
    args = parse_args()
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    aggregate = aggregate_results(results)
    args.output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "points": [
                    {
                        "clusters": point["clusters"],
                        "local_rank": point["local_rank"],
                        "bits": point["capsule_bits"],
                        "stored_rank": point["stored_equivalent_rank"],
                        "active_rank": point["active_equivalent_rank"],
                        "training_error": point[
                            "post_quantization_training_reconstruction"
                        ]["maximum_module_output_relative_error"],
                        "top1": point["exact_top1_match_rate"],
                        "top32": point["coverage_at_k"]["32"],
                        "mean_exact_rank": point["rank_statistics"]["mean"],
                        "qualifies": point["qualifies"],
                    }
                    for point in aggregate["points"]
                ],
                "decision": aggregate["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
