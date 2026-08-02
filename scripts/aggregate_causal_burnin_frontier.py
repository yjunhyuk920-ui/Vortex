from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independently measured causal burn-in points."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize_point(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_burnin_tokens": int(result["exact_burnin_tokens"]),
        "burnin_token_ids": result["burnin_token_ids"],
        "evaluated_unseen_tokens": int(result["evaluated_unseen_tokens"]),
        "compiled_rank_statistics": result["compiled_rank_statistics"],
        "capsule_bits": int(result["capsule_bits"]),
        "prompt_burnin_reconstruction_after_quantization": result[
            "prompt_burnin_reconstruction_after_quantization"
        ],
        "quantization": result["quantization"]["aggregate"],
        "hot_budget": result["hot_budget"],
        "startup_exact_cost": result["startup_exact_cost"],
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": result["coverage_at_k"],
        "rank_statistics": result["rank_statistics"],
        "first_divergence": result["first_divergence"],
        "warm_decode_candidate_pass": bool(
            result["warm_decode_candidate_pass"]
        ),
        "full_session_4096_pass": bool(result["full_session_4096_pass"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    loaded = list(results)
    if not loaded:
        raise ValueError("at least one causal burn-in result is required")
    points = sorted(
        (summarize_point(result) for result in loaded),
        key=lambda point: point["exact_burnin_tokens"],
    )
    burnins = [point["exact_burnin_tokens"] for point in points]
    if len(set(burnins)) != len(burnins):
        raise ValueError("duplicate causal burn-in points")

    warm_survivors = [
        point for point in points if point["warm_decode_candidate_pass"]
    ]
    full_session_survivors = [
        point for point in points if point["full_session_4096_pass"]
    ]
    best = max(
        points,
        key=lambda point: (
            float(point["coverage_at_k"]["32"]),
            float(point["exact_top1_match_rate"]),
            -int(point["exact_burnin_tokens"]),
            -float(point["rank_statistics"]["mean"]),
        ),
    )
    return {
        "evidence_level": "E1 causal exact-burnin local trajectory frontier",
        "model": loaded[0]["model"],
        "device": loaded[0]["device"],
        "evaluated_unseen_tokens_per_point": int(
            loaded[0]["evaluated_unseen_tokens"]
        ),
        "causal_contract": loaded[0]["causal_contract"],
        "decision_rule": loaded[0]["decision_rule"],
        "tested_burnin_tokens": burnins,
        "points": points,
        "best_observed_point": best,
        "warm_decode_survivors": warm_survivors,
        "full_session_4096_survivors": full_session_survivors,
        "decision": (
            "advance causal burnin local trajectory capsule"
            if warm_survivors
            else "reject tested causal burnin local trajectory frontier"
        ),
        "next_candidate": (
            "use the smallest passing burn-in and construct a causal verifier"
            if warm_survivors
            else (
                "replace one frozen session capsule with an online rolling "
                "dictionary of local trajectory capsules built only from "
                "certified prefixes"
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
                        "burnin": point["exact_burnin_tokens"],
                        "rank": point["compiled_rank_statistics"]["maximum"],
                        "top1": point["exact_top1_match_rate"],
                        "top32": point["coverage_at_k"]["32"],
                        "mean_exact_rank": point["rank_statistics"]["mean"],
                        "minimum_traffic_horizon": point[
                            "startup_exact_cost"
                        ]["minimum_traffic_amortization_horizon"],
                        "horizon_4096_pass": point[
                            "startup_exact_cost"
                        ]["horizon_4096_pass"],
                        "warm_survives": point["warm_decode_candidate_pass"],
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
