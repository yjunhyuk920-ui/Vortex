from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PRECISION_FRONTIER = Path(
    "results/tinyllama_1_1b_precision_rank_frontier.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def apply_precision_frontier(
    report: dict[str, Any],
    precision_frontier_path: str | Path = DEFAULT_PRECISION_FRONTIER,
) -> dict[str, Any]:
    source = Path(precision_frontier_path)
    if not source.exists():
        return report

    result = json.loads(source.read_text(encoding="utf-8"))
    survivors = result.get("surviving_points", [])
    passed = bool(survivors)
    points = []
    for point in result["points"]:
        points.append(
            {
                "requested_rank": int(point["rank"]),
                "capsule_bits": int(point["capsule_bits"]),
                "built_rank_statistics": point["built_rank_statistics"],
                "memory_gib": float(point["budget"]["memory_gib"]),
                "hot_traffic_gib_per_token": float(
                    point["budget"]["hot_traffic_gib_per_token"]
                ),
                "hot_compute_gflop_per_token": float(
                    point["budget"]["hot_compute_gflop_per_token"]
                ),
                "maximum_quantized_relative_l2_error": float(
                    point["quantization"]["maximum_relative_l2_error"]
                ),
                "exact_top1_match_rate": float(
                    point["exact_top1_match_rate"]
                ),
                "top32_coverage": float(point["coverage_at_k"]["32"]),
                "mean_exact_token_rank": float(
                    point["rank_statistics"]["mean"]
                ),
                "maximum_exact_token_rank": int(
                    point["rank_statistics"]["maximum"]
                ),
                "first_divergence_exact_token_rank": (
                    None
                    if point["first_divergence"] is None
                    else int(point["first_divergence"]["exact_token_rank"])
                ),
                "qualifies": bool(
                    point["qualifies_for_multi_hypothesis"]
                ),
            }
        )

    maximum_built_rank = max(
        int(point["built_rank_statistics"]["maximum"])
        for point in result["points"]
    )
    best = max(
        points,
        key=lambda point: (
            float(point["top32_coverage"]),
            float(point["exact_top1_match_rate"]),
            -float(point["mean_exact_token_rank"]),
        ),
    )
    report["precision_rank_frontier"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "precision_aligned_maximum_ranks": result[
            "precision_aligned_maximum_ranks"
        ],
        "maximum_built_rank": maximum_built_rank,
        "quantization_contract": result["quantization_contract"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": [
            {
                "rank": int(point["rank"]),
                "capsule_bits": int(point["capsule_bits"]),
            }
            for point in survivors
        ],
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["precision_rank_frontier"] = passed

    if passed:
        lowest_cost = min(
            survivors,
            key=lambda point: (
                int(point["capsule_bits"]) * int(point["rank"]),
                int(point["rank"]),
            ),
        )
        report["status"] = "quantized-global-capsule-candidate"
        report["observed_component_decision"] = (
            "A quantized global capsule satisfies the causal top-32 coverage "
            "gate at rank "
            f"{int(lowest_cost['rank'])} and "
            f"{int(lowest_cost['capsule_bits'])} bits."
        )
        report["next_candidate"] = (
            "combine the surviving quantized global capsule with exact-prompt "
            "session residual directions and build the causal certificate"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = (
            "independently quantized generic global O/down capsules at the "
            "tested 8/6/4-bit precision-rank frontier"
        )
        if name not in rejected:
            rejected.append(name)
        report["status"] = "quantized-global-frontier-insufficient"
        report["observed_component_decision"] = (
            "No tested quantized global capsule reached the 95-percent top-32 "
            f"coverage gate. The largest actually constructed rank was "
            f"{maximum_built_rank}, so requested ranks above it are not treated "
            "as independent falsification points."
        )
        report["next_candidate"] = (
            "augment a 4-bit global prior with exact-prompt residual directions "
            "inside the rank-136 Gate, then measure unseen continuation coverage"
        )
    return report
