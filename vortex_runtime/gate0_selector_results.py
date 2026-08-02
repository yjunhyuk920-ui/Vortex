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


def apply_selector_falsifications(
    report: dict[str, Any],
    *,
    proposal_adjoint_path: str | Path = DEFAULT_PROPOSAL_ADJOINT,
    margin_bound_path: str | Path = DEFAULT_MARGIN_BOUND,
    prefill_compiled_path: str | Path = DEFAULT_PREFILL_COMPILED,
) -> dict[str, Any]:
    """Attach causal selector evidence and reject the family when all fail.

    The exact-target block oracle is retained as a logical upper bound. It does
    not keep the family active when every tested selector available before the
    future continuation fails to recover any additional exact token.
    """

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
            "multi-hypothesis uncertainty certificate with a redesigned hot "
            "representation; do not add more single-proposal selector heuristics"
        )
    return report
