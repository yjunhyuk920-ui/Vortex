from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_DICTIONARY_FRONTIER = Path(
    "results/tinyllama_1_1b_local_affine_dictionary_frontier.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def apply_local_dictionary_frontier(
    report: dict[str, Any],
    frontier_path: str | Path = DEFAULT_LOCAL_DICTIONARY_FRONTIER,
) -> dict[str, Any]:
    source = Path(frontier_path)
    if not source.exists():
        return report

    result = json.loads(source.read_text(encoding="utf-8"))
    survivors = result.get("surviving_points", [])
    passed = bool(survivors)
    points = [
        {
            "clusters": int(point["clusters"]),
            "local_rank": int(point["local_rank"]),
            "capsule_bits": int(point["capsule_bits"]),
            "captured_vectors_per_module": int(
                point["captured_vectors_per_module"]
            ),
            "stored_equivalent_rank": int(point["stored_equivalent_rank"]),
            "active_equivalent_rank": int(point["active_equivalent_rank"]),
            "stored_memory_pass": bool(
                point["stored_budget"]["memory_pass"]
            ),
            "active_traffic_pass": bool(
                point["active_budget"]["traffic_pass"]
            ),
            "active_compute_pass": bool(
                point["active_budget"]["compute_pass"]
            ),
            "maximum_post_quant_training_output_error": float(
                point["post_quantization_training_reconstruction"][
                    "maximum_module_output_relative_error"
                ]
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
            "qualifies": bool(point["qualifies"]),
        }
        for point in result["points"]
    ]
    best = max(
        points,
        key=lambda point: (
            float(point["top32_coverage"]),
            float(point["exact_top1_match_rate"]),
            -float(point["mean_exact_token_rank"]),
            -int(point["active_equivalent_rank"]),
        ),
    )

    report["local_affine_dictionary_frontier"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "causal_contract": result["causal_contract"],
        "budget_contract": result["budget_contract"],
        "points": points,
        "best_observed_point": best,
        "surviving_points": [
            {
                "clusters": int(point["clusters"]),
                "local_rank": int(point["local_rank"]),
                "capsule_bits": int(point["capsule_bits"]),
            }
            for point in survivors
        ],
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["local_affine_dictionary_frontier"] = passed

    if passed:
        lowest_active = min(
            survivors,
            key=lambda point: (
                int(point["active_equivalent_rank"]),
                int(point["stored_equivalent_rank"]),
                int(point["capsule_bits"]),
            ),
        )
        report["status"] = "routed-local-affine-dictionary-candidate"
        report["observed_component_decision"] = (
            "A causal routed local affine response dictionary passes stored "
            "memory, active traffic/compute, and unseen top-32 coverage gates."
        )
        report["next_candidate"] = (
            "construct a sound route-and-token certificate and measure a packed "
            "physical runtime for the lowest-active-cost surviving dictionary"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = (
            "uniform per-module routed local affine dictionaries at the tested "
            "cluster/rank/precision frontier"
        )
        if name not in rejected:
            rejected.append(name)
        report["status"] = "uniform-local-affine-dictionary-insufficient"
        report["observed_component_decision"] = (
            "No tested routed local affine dictionary reached the 95-percent "
            f"top-32 gate; the best point reached "
            f"{float(best['top32_coverage']):.6f}."
        )
        report["next_candidate"] = (
            "allocate local ranks and precisions non-uniformly by causal prompt "
            "output-error reduction per packed byte, with layer-shared routing"
        )
    return report
