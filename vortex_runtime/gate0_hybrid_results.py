from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_HYBRID_FRONTIER = Path(
    "results/tinyllama_1_1b_hybrid_allocation_frontier.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def apply_hybrid_frontier(
    report: dict[str, Any],
    hybrid_frontier_path: str | Path = DEFAULT_HYBRID_FRONTIER,
) -> dict[str, Any]:
    source = Path(hybrid_frontier_path)
    if not source.exists():
        return report

    result = json.loads(source.read_text(encoding="utf-8"))
    survivors = result.get("surviving_points", [])
    passed = bool(survivors)
    points: list[dict[str, Any]] = []
    for point in result["points"]:
        points.append(
            {
                "global_rank_limit": int(point["global_rank_limit"]),
                "total_rank_limit": int(point["total_rank_limit"]),
                "capsule_bits": int(point["capsule_bits"]),
                "global_rank_statistics": point["global_rank_statistics"],
                "added_session_rank_statistics": point[
                    "added_session_rank_statistics"
                ],
                "final_rank_statistics": point["final_rank_statistics"],
                "maximum_final_prompt_output_relative_error": float(
                    point["prompt_reconstruction"][
                        "maximum_final_output_relative_error"
                    ]
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

    best = max(
        points,
        key=lambda point: (
            float(point["top32_coverage"]),
            float(point["exact_top1_match_rate"]),
            -float(point["mean_exact_token_rank"]),
        ),
    )
    report["hybrid_allocation_frontier"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "causal_contract": result["causal_contract"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": [
            {
                "global_rank_limit": int(point["global_rank_limit"]),
                "total_rank_limit": int(point["total_rank_limit"]),
                "capsule_bits": int(point["capsule_bits"]),
            }
            for point in survivors
        ],
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["hybrid_allocation_frontier"] = passed

    if passed:
        lowest_cost = min(
            survivors,
            key=lambda point: (
                int(point["capsule_bits"])
                * float(point["final_rank_statistics"]["mean"]),
                float(point["final_rank_statistics"]["mean"]),
            ),
        )
        report["status"] = "causal-hybrid-capsule-candidate"
        report["observed_component_decision"] = (
            "A causal global-plus-exact-prompt response capsule passes the "
            "top-32 continuation coverage gate."
        )
        report["next_candidate"] = (
            "build a sound causal token certificate and physical packed-kernel "
            "measurement for the lowest-cost surviving hybrid allocation"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = (
            "uniform-precision global plus exact-prompt residual hybrid at the "
            "tested 6/4-bit allocations"
        )
        if name not in rejected:
            rejected.append(name)
        report["status"] = "uniform-precision-hybrid-insufficient"
        report["observed_component_decision"] = (
            "The causal hybrid improved the best top-32 coverage to "
            f"{float(best['top32_coverage']):.6f}, but remained below 0.95."
        )
        report["next_candidate"] = (
            "preserve session residual columns at higher precision than the "
            "generic prior, then allocate session directions non-uniformly by "
            "module benefit per packed byte"
        )
    return report
