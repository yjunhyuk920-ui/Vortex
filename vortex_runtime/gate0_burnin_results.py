from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_BURNIN_FRONTIER = Path(
    "results/tinyllama_1_1b_causal_burnin_frontier.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def apply_burnin_frontier(
    report: dict[str, Any],
    burnin_frontier_path: str | Path = DEFAULT_BURNIN_FRONTIER,
) -> dict[str, Any]:
    source = Path(burnin_frontier_path)
    if not source.exists():
        return report

    result = json.loads(source.read_text(encoding="utf-8"))
    warm_survivors = result.get("warm_decode_survivors", [])
    session_survivors = result.get("full_session_4096_survivors", [])
    passed = bool(warm_survivors)
    points = [
        {
            "exact_burnin_tokens": int(point["exact_burnin_tokens"]),
            "compiled_rank_statistics": point["compiled_rank_statistics"],
            "exact_top1_match_rate": float(point["exact_top1_match_rate"]),
            "top32_coverage": float(point["coverage_at_k"]["32"]),
            "mean_exact_token_rank": float(point["rank_statistics"]["mean"]),
            "maximum_exact_token_rank": int(point["rank_statistics"]["maximum"]),
            "minimum_traffic_amortization_horizon": point[
                "startup_exact_cost"
            ]["minimum_traffic_amortization_horizon"],
            "minimum_compute_amortization_horizon": point[
                "startup_exact_cost"
            ]["minimum_compute_amortization_horizon"],
            "horizon_4096_pass": bool(
                point["startup_exact_cost"]["horizon_4096_pass"]
            ),
            "warm_decode_candidate_pass": bool(
                point["warm_decode_candidate_pass"]
            ),
        }
        for point in result["points"]
    ]
    best = max(
        points,
        key=lambda point: (
            float(point["top32_coverage"]),
            -int(point["exact_burnin_tokens"]),
            float(point["exact_top1_match_rate"]),
        ),
    )
    report["causal_burnin_frontier"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "causal_contract": result["causal_contract"],
        "points": points,
        "best_observed_point": best,
        "warm_decode_surviving_burnins": [
            int(point["exact_burnin_tokens"]) for point in warm_survivors
        ],
        "full_session_4096_surviving_burnins": [
            int(point["exact_burnin_tokens"]) for point in session_survivors
        ],
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["causal_burnin_local_trajectory"] = passed

    if passed:
        smallest = min(
            int(point["exact_burnin_tokens"])
            for point in warm_survivors
        )
        report["status"] = "causal-burnin-local-capsule-candidate"
        report["observed_component_decision"] = (
            f"A causal exact burn-in of {smallest} tokens produced a frozen "
            "local trajectory capsule that passes the unseen continuation gate."
        )
        report["next_candidate"] = (
            "construct a causal verifier and measure physical startup plus warm "
            "decode wall-clock for the smallest passing burn-in"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = (
            "single frozen 8-bit prompt-plus-exact-burnin local trajectory capsule"
        )
        if name not in rejected:
            rejected.append(name)
        report["status"] = "single-burnin-local-capsule-insufficient"
        report["observed_component_decision"] = (
            "No tested causal burn-in reached the 95-percent top-32 gate. The "
            f"best point used {int(best['exact_burnin_tokens'])} exact startup "
            f"tokens and reached {float(best['top32_coverage']):.6f}; its minimum "
            "traffic amortization horizon also exceeded the fixed context when "
            "applicable."
        )
        report["next_candidate"] = (
            "replace the single linear response capsule with a routed local "
            "affine capsule dictionary whose stored rank and active rank are "
            "separately budgeted"
        )
    return report
