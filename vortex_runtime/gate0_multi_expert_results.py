from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MULTI_EXPERT = Path(
    "results/tinyllama_1_1b_multi_expert_candidate_union.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def apply_multi_expert_result(
    report: dict[str, Any],
    multi_expert_path: str | Path = DEFAULT_MULTI_EXPERT,
) -> dict[str, Any]:
    source = Path(multi_expert_path)
    if not source.exists():
        return report

    result = json.loads(source.read_text(encoding="utf-8"))
    best_router = result.get("best_budget_compatible_router")
    passed = result["decision"] == "advance independent-capsule causal certificate"
    fixed = result["fixed_total_candidate_allocations"]
    equal_width = result["equal_per_expert_width_diagnostics"]
    best_fixed = max(
        fixed,
        key=lambda item: (
            float(item["coverage"]),
            -float(item["mean_candidate_count"]),
        ),
    )
    best_equal = max(
        equal_width,
        key=lambda item: (
            float(item["coverage"]),
            -float(item["mean_candidate_count"]),
        ),
    )

    report["multi_expert_candidate_union"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "causal_contract": result["causal_contract"],
        "generic_expert": result["generic_expert"],
        "session_expert": result["session_expert"],
        "dictionary_budget": result["dictionary_budget"],
        "best_fixed_32_candidate_allocation": best_fixed,
        "best_equal_width_diagnostic": best_equal,
        "best_budget_compatible_router": best_router,
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["multi_expert_margin_router"] = passed

    if passed:
        report["status"] = "independent-capsule-router-candidate"
        report["observed_component_decision"] = (
            "A causal margin-routed independent-capsule candidate union passes "
            "memory, average traffic, compute, candidate-count, and coverage gates."
        )
        report["next_candidate"] = (
            "construct a sound causal token certificate and physical packed "
            "multi-capsule runtime for the passing router"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = (
            "two independent 8-bit generic/session capsules with a scalar "
            "top-1-margin fallback router"
        )
        if name not in rejected:
            rejected.append(name)
        router_coverage = (
            None if best_router is None else float(best_router["coverage"])
        )
        report["status"] = "two-capsule-margin-router-insufficient"
        report["observed_component_decision"] = (
            "Two independent capsules fit stored memory but fail the causal "
            "candidate gate: best fixed 32-candidate coverage was "
            f"{float(best_fixed['coverage']):.6f} and best budget-compatible "
            f"margin-router coverage was {router_coverage}."
        )
        report["next_candidate"] = (
            "compile a local trajectory capsule from the exact prompt plus a "
            "small causal exact-token burn-in, then evaluate only unseen future "
            "continuation while charging startup amortization separately"
        )
    return report
