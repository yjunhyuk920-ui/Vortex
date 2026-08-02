from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independently measured mixed-precision hybrid points."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize_point(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_rank_limit": int(result["global_rank_limit"]),
        "session_rank_limit": int(result["session_rank_limit"]),
        "total_rank_limit": int(result["total_rank_limit"]),
        "global_bits": int(result["global_bits"]),
        "session_bits": int(result["session_bits"]),
        "global_rank_statistics": result["global_rank_statistics"],
        "added_session_rank_statistics": result[
            "added_session_rank_statistics"
        ],
        "final_rank_statistics": result["final_rank_statistics"],
        "prompt_reconstruction_before_quantization": result[
            "prompt_reconstruction_before_quantization"
        ],
        "prompt_reconstruction_after_quantization": result[
            "prompt_reconstruction_after_quantization"
        ],
        "quantization": result["quantization"]["aggregate"],
        "budget": result["budget"],
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": result["coverage_at_k"],
        "rank_statistics": result["rank_statistics"],
        "first_divergence": result["first_divergence"],
        "qualifies_for_multi_hypothesis": (
            result["decision"] == "advance mixed-precision hybrid certificate"
        ),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    loaded = list(results)
    if not loaded:
        raise ValueError("at least one mixed-precision result is required")
    points = [summarize_point(result) for result in loaded]
    identities = {
        (
            point["global_rank_limit"],
            point["session_rank_limit"],
            point["global_bits"],
            point["session_bits"],
        )
        for point in points
    }
    if len(identities) != len(points):
        raise ValueError("duplicate mixed-precision points")

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
    return {
        "evidence_level": "E1 causal mixed-precision hybrid frontier",
        "model": loaded[0]["model"],
        "device": loaded[0]["device"],
        "tokens_per_point": int(loaded[0]["evaluated_continuation_tokens"]),
        "tested_points": [
            {
                "global_rank": point["global_rank_limit"],
                "session_rank": point["session_rank_limit"],
                "global_bits": point["global_bits"],
                "session_bits": point["session_bits"],
            }
            for point in points
        ],
        "causal_contract": loaded[0]["compiler_contract"],
        "decision_rule": loaded[0]["decision_rule"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": survivors,
        "decision": (
            "advance mixed-precision hybrid allocation"
            if survivors
            else "reject tested mixed-precision hybrid allocations"
        ),
        "next_candidate": (
            "build the sound causal certificate at the lowest-traffic survivor"
            if survivors
            else (
                "allocate session directions non-uniformly by module benefit per "
                "packed byte, then test a shared-global state-conditional capsule "
                "dictionary"
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
                        "global": point["global_rank_limit"],
                        "session": point["session_rank_limit"],
                        "global_bits": point["global_bits"],
                        "session_bits": point["session_bits"],
                        "top1": point["exact_top1_match_rate"],
                        "top32": point["coverage_at_k"]["32"],
                        "mean_exact_rank": point["rank_statistics"]["mean"],
                        "post_quant_prompt_error": point[
                            "prompt_reconstruction_after_quantization"
                        ]["maximum_module_output_relative_error"],
                        "qualifies": point[
                            "qualifies_for_multi_hypothesis"
                        ],
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
