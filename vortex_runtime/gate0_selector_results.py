from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PROPOSAL_ADJOINT = Path(
    "results/tinyllama_1_1b_block_shared_proposal_adjoint_oracle.json"
)
DEFAULT_MARGIN_BOUND = Path(
    "results/tinyllama_1_1b_block_shared_margin_bound_selector.json"
)
DEFAULT_PREFILL_COMPILED = Path(
    "results/tinyllama_1_1b_prefill_compiled_adjoint_oracle.json"
)
DEFAULT_CANDIDATE_COVERAGE = Path(
    "results/tinyllama_1_1b_hot_candidate_coverage.json"
)
DEFAULT_RANK_FRONTIER = Path(
    "results/tinyllama_1_1b_hot_rank_frontier.json"
)
DEFAULT_SESSION_PREFILL = Path(
    "results/tinyllama_1_1b_session_prefill_candidate_coverage.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _load_failure(path: Path, selector_name: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    best = result.get("best_selector_candidate")
    return {
        "selector": selector_name,
        "evidence_level": result["evidence_level"],
        "source": _portable_source(path),
        "zero_repair_prefix_tokens": int(result["zero_repair_prefix_tokens"]),
        "combined_budget_tile_count": int(
            result["combined_budget"]["combined_budget_tile_count"]
        ),
        "pass": best is not None,
        "best_selector_candidate": best,
        "decision": result["decision"],
        "rejection_scope": result["rejection_scope"],
    }


def _attach_candidate_coverage(
    report: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    if not source.exists():
        return report
    result = json.loads(source.read_text(encoding="utf-8"))
    coverage = result["coverage_at_k"]
    first = result.get("first_divergence")
    first_rank = None if first is None else int(first["exact_token_rank"])
    top32 = float(coverage["32"])
    advance = (first_rank is None or first_rank <= 32) and top32 >= 0.95

    report["hot_representation_coverage"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "max_rank": int(result["max_rank"]),
        "evaluated_tokens": int(result["evaluated_tokens"]),
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": coverage,
        "rank_statistics": result["rank_statistics"],
        "first_divergence": first,
        "advance_multi_hypothesis": advance,
        "decision": result["decision"],
        "rejection_scope": result["rejection_scope"],
    }
    report["gates"]["rank32_topk_coverage"] = advance

    if not advance:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = "rank-32 O/down hot representation for small top-K uncertainty"
        if name not in rejected:
            rejected.append(name)
        report["status"] = "rank32-hot-representation-rejected"
        report["observed_component_decision"] = (
            "Reject the rank-32 O/down hot representation as a small top-K "
            "certificate basis: first-divergence rank was "
            f"{first_rank}, but top-32 exact-token coverage was {top32:.6f}."
        )
        report["next_candidate"] = (
            "sweep the 8-bit hot-traffic-feasible capsule-rank frontier through "
            "rank 72"
        )
    return report


def _attach_rank_frontier(
    report: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    if not source.exists():
        return report
    result = json.loads(source.read_text(encoding="utf-8"))
    survivors = result.get("surviving_points", [])
    passed = bool(survivors)
    points = [
        {
            "rank": int(point["rank"]),
            "memory_gib": float(point["budget"]["memory_gib"]),
            "hot_traffic_gib_per_token": float(
                point["budget"]["hot_traffic_gib_per_token"]
            ),
            "hot_compute_gflop_per_token": float(
                point["budget"]["hot_compute_gflop_per_token"]
            ),
            "exact_top1_match_rate": float(point["exact_top1_match_rate"]),
            "top32_coverage": float(point["coverage_at_k"]["32"]),
            "mean_exact_token_rank": float(point["rank_statistics"]["mean"]),
            "qualifies": bool(point["qualifies_for_multi_hypothesis"]),
        }
        for point in result["points"]
    ]
    report["hot_rank_frontier"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "capsule_precision_bits": 8,
        "binding_budget": result.get("binding_budget"),
        "fixed_405b_aligned_maximum_rank": int(
            result["fixed_405b_aligned_maximum_rank"]
        ),
        "tested_ranks": result["tested_ranks"],
        "points": points,
        "surviving_ranks": [int(point["rank"]) for point in survivors],
        "pass": passed,
        "decision": result["decision"],
        "next_candidate": result["next_candidate"],
    }
    report["gates"]["feasible_rank_frontier_8bit"] = passed

    if passed:
        lowest = min(int(point["rank"]) for point in survivors)
        report["status"] = "feasible-rank-multihypothesis-candidate"
        report["observed_component_decision"] = (
            f"An 8-bit 405B-budget-compatible rank {lowest} capsule preserves "
            "the exact token inside the required top-32 candidate set."
        )
        report["next_candidate"] = (
            f"build and falsify a causal multi-hypothesis certificate at rank {lowest}"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = "generic-build-prompt 8-bit O/down capsule through rank 72"
        if name not in rejected:
            rejected.append(name)
        report["status"] = "generic-8bit-rank-frontier-rejected"
        report["observed_component_decision"] = (
            "Reject the generic-build-prompt 8-bit O/down capsule through the "
            "hot-traffic-bound rank-72 frontier. This does not reject lower-"
            "precision higher-rank capsules or prompt-specific session bases."
        )
        report["next_candidate"] = (
            "test the exact-prompt session response basis and the 4/6-bit "
            "precision-rank frontier through the compute-bound rank 136"
        )
    return report


def _attach_session_prefill(
    report: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    if not source.exists():
        return report
    result = json.loads(source.read_text(encoding="utf-8"))
    coverage = result["coverage_at_k"]
    first = result.get("first_divergence")
    first_rank = None if first is None else int(first["exact_token_rank"])
    top32 = float(coverage["32"])
    passed = (first_rank is None or first_rank <= 32) and top32 >= 0.95

    report["session_prefill_response_basis"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "prompt_tokens": int(result["prompt_tokens"]),
        "evaluated_continuation_tokens": int(
            result["evaluated_continuation_tokens"]
        ),
        "requested_max_rank": int(result["requested_max_rank"]),
        "compiled_rank_statistics": result["compiled_rank_statistics"],
        "capsule_bytes": int(result["capsule_bytes"]),
        "prompt_reconstruction": result["prompt_reconstruction"],
        "exact_top1_match_rate": float(result["exact_top1_match_rate"]),
        "coverage_at_k": coverage,
        "rank_statistics": result["rank_statistics"],
        "first_divergence": first,
        "pass": passed,
        "decision": result["decision"],
        "compiler_contract": result["compiler_contract"],
    }
    report["gates"]["session_prefill_response_basis"] = passed

    if passed:
        report["status"] = "session-prefill-multihypothesis-candidate"
        report["observed_component_decision"] = (
            "The exact-prompt response basis causally preserves the unseen "
            "continuation token inside the required top-32 candidate set."
        )
        report["next_candidate"] = (
            "build a causal multi-hypothesis certificate around the prompt-"
            "compiled session basis"
        )
    else:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = "prompt-only rank-45 session response basis without adaptive growth"
        if name not in rejected:
            rejected.append(name)
        report["status"] = "session-prefill-basis-insufficient"
        report["observed_component_decision"] = (
            "The exact-prompt session basis materially improved top-1 coverage "
            f"but remained below the top-32 gate ({top32:.6f} < 0.95)."
        )
        report["next_candidate"] = (
            "test quantized 6-bit rank-96 and 4-bit rank-128/136 capsules, then "
            "combine the best feasible global prior with the prompt-specific "
            "session basis"
        )
    return report


def apply_selector_falsifications(
    report: dict[str, Any],
    *,
    proposal_adjoint_path: str | Path = DEFAULT_PROPOSAL_ADJOINT,
    margin_bound_path: str | Path = DEFAULT_MARGIN_BOUND,
    prefill_compiled_path: str | Path = DEFAULT_PREFILL_COMPILED,
    candidate_coverage_path: str | Path = DEFAULT_CANDIDATE_COVERAGE,
    rank_frontier_path: str | Path = DEFAULT_RANK_FRONTIER,
    session_prefill_path: str | Path = DEFAULT_SESSION_PREFILL,
) -> dict[str, Any]:
    """Attach causal selector and hot-representation falsification evidence."""

    selectors = report.setdefault("selector_falsification", {})
    specifications = (
        (
            "proposal_adjoint_full_weight_scan",
            Path(proposal_adjoint_path),
            "block-shared proposal-token signed-adjoint selector",
        ),
        (
            "proposal_margin_metadata_bound",
            Path(margin_bound_path),
            "block-shared proposal-margin metadata-bound selector",
        ),
        (
            "causal_prefill_compiler",
            Path(prefill_compiled_path),
            "causal exact-prompt-prefill adjoint compiler",
        ),
    )

    loaded: list[dict[str, Any]] = []
    for key, path, rejected_name in specifications:
        evidence = _load_failure(path, key)
        if evidence is None:
            continue
        selectors[key] = evidence
        report["gates"][key] = bool(evidence["pass"])
        loaded.append(evidence)
        if not evidence["pass"]:
            rejected = report.setdefault("rejected_mechanisms", [])
            if rejected_name not in rejected:
                rejected.append(rejected_name)

    residual = selectors.get("residual_energy")
    causal_results = ([residual] if residual is not None else []) + loaded
    complete = len(causal_results) == 4
    all_failed = complete and all(not bool(item["pass"]) for item in causal_results)

    report["family_decision"] = {
        "family": "rank-32 O/down capsule plus fixed block-shared exact-tile repair",
        "exact_target_logical_oracle_pass": bool(
            report.get("gates", {}).get("logical_combined_oracle")
        ),
        "causal_selector_results_complete": complete,
        "causal_selectors_tested": len(causal_results),
        "causal_selectors_passing": sum(
            1 for item in causal_results if bool(item["pass"])
        ),
        "future_information_dependency": (
            "The only passing tile set was selected with exact future target "
            "tokens and teacher-forced continuation gradients."
        ),
        "rejected": all_failed,
    }

    if all_failed:
        report["status"] = "rank32-block-repair-family-rejected"
        report["gates"]["target_independent_selector"] = False
        report["observed_component_decision"] = (
            "Reject the rank-32 O/down fixed block-repair family: one exact-"
            "future oracle passed, but residual-energy, proposal-adjoint, "
            "proposal-margin metadata, and causal prefill selectors all failed "
            "to extend the exact prefix."
        )
        report["next_candidate"] = (
            "measure exact-token candidate coverage before attempting a "
            "multi-hypothesis uncertainty certificate"
        )

    report = _attach_candidate_coverage(report, Path(candidate_coverage_path))
    report = _attach_rank_frontier(report, Path(rank_frontier_path))
    return _attach_session_prefill(report, Path(session_prefill_path))
