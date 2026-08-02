from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EXACT_SPAN_RESULT = Path(
    "results/tinyllama_1_1b_exact_span_warm_decode.json"
)
DEFAULT_ORACLE_RESULT = Path(
    "results/tinyllama_1_1b_rank32_repair_oracles.json"
)
DEFAULT_RESIDUAL_RESULT = Path(
    "results/tinyllama_1_1b_residual_tile_oracle.json"
)
DEFAULT_ADJOINT_RESULT = Path(
    "results/tinyllama_1_1b_adjoint_tile_oracle.json"
)
DEFAULT_COMBINED_RESULT = Path(
    "results/tinyllama_1_1b_block_shared_combined_gate.json"
)
DEFAULT_RESIDUAL_SELECTOR_RESULT = Path(
    "results/tinyllama_1_1b_block_shared_residual_selector.json"
)


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _attach_residual_selector_falsification(
    report: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    result = json.loads(source.read_text(encoding="utf-8"))
    boundary = result["tested_boundary"]
    passed = result.get("best_selector_candidate") is not None
    report.setdefault("selector_falsification", {})["residual_energy"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "zero_repair_prefix_tokens": int(result["zero_repair_prefix_tokens"]),
        "combined_budget_tile_count": int(
            result["combined_budget"]["combined_budget_tile_count"]
        ),
        "boundary_selected_tiles": int(boundary["selected_tiles"]),
        "boundary_selected_weight_bytes": int(
            boundary["selected_weight_bytes"]
        ),
        "boundary_committed_prefix_tokens": int(
            boundary["committed_prefix_tokens"]
        ),
        "boundary_incremental_committed_tokens": int(
            boundary["incremental_committed_tokens"]
        ),
        "pass": passed,
        "decision": result["decision"],
        "rejection_scope": result["rejection_scope"],
    }
    report["gates"]["residual_energy_selector"] = passed
    if not passed:
        rejected = report.setdefault("rejected_mechanisms", [])
        name = "block-shared residual-energy target-independent selector"
        if name not in rejected:
            rejected.append(name)
        report["next_candidate"] = (
            "proposal-adjoint diagnostic oracle and proposal-margin "
            "metadata-bound selector"
        )
    return report


def _apply_combined_oracle(
    report: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    result = json.loads(source.read_text(encoding="utf-8"))
    best = result["best_combined_candidate"]
    observed = float(best["traffic_efficiency"])
    repair_fraction = float(best["repair_fraction"])
    committed_tokens = int(best["committed_prefix_tokens"])
    incremental_tokens = int(
        best["incremental_committed_tokens_over_zero_repair"]
    )

    required = float(
        report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )
    maximum_compute_fraction = float(
        report["mechanism"]["maximum_compute_repair_fraction"]
    )
    traffic_pass = observed >= required
    compute_pass = repair_fraction <= maximum_compute_fraction
    logical_pass = (
        traffic_pass
        and compute_pass
        and incremental_tokens > 0
        and committed_tokens > 0
    )

    hot_traffic = float(report["traffic"]["hot_gib_per_token"])
    cold_traffic = float(report["traffic"]["cold_full_repair_gib"])
    traffic_limit = float(report["traffic"]["limit_gib_per_token"])
    hot_compute = float(report["compute"]["hot_total_gflop_per_token"])
    cold_compute = float(report["compute"]["cold_full_repair_gflop"])
    compute_limit = float(report["compute"]["limit_gflop_per_token"])

    projected_traffic = hot_traffic + cold_traffic / observed
    projected_compute = hot_compute + cold_compute * repair_fraction

    report["mechanism"].update(
        {
            "observed": observed,
            "observed_repair_fraction": repair_fraction,
            "observed_committed_tokens": committed_tokens,
            "observed_incremental_committed_tokens": incremental_tokens,
            "observed_source": _portable_source(source),
            "traffic_pass": traffic_pass,
            "compute_pass": compute_pass,
            "pass": logical_pass,
            "traffic_shortfall_factor": (
                required / observed if observed > 0 else float("inf")
            ),
            "compute_excess_factor": (
                repair_fraction / maximum_compute_fraction
                if maximum_compute_fraction > 0
                else float("inf")
            ),
            "selector_proven": False,
            "certificate_proven": False,
            "managed_model_wide": False,
        }
    )
    report["logical_oracle"] = {
        "evidence_level": result["evidence_level"],
        "source": _portable_source(source),
        "zero_repair_prefix_tokens": int(
            result["zero_repair_baseline"]["committed_prefix_tokens"]
        ),
        "selected_tiles": int(best["selected_tiles"]),
        "selected_weight_bytes": int(best["selected_weight_bytes"]),
        "repair_fraction": repair_fraction,
        "committed_prefix_tokens": committed_tokens,
        "incremental_committed_tokens": incremental_tokens,
        "traffic_efficiency": observed,
        "projected_traffic_gib_per_token": projected_traffic,
        "projected_compute_gflop_per_token": projected_compute,
        "traffic_pass": traffic_pass,
        "compute_pass": compute_pass,
        "logical_budget_pass": logical_pass,
        "selector_uses_exact_target_tokens": True,
        "selector_uses_teacher_gradients": True,
        "sound_commit_certificate": False,
        "scope": "TinyLlama 1.1B O/down projections on one disjoint evaluation prompt",
    }
    report["gates"]["observed_repair_traffic"] = traffic_pass
    report["gates"]["observed_repair_compute"] = compute_pass
    report["gates"]["observed_repair_efficiency"] = logical_pass
    report["gates"]["logical_combined_oracle"] = logical_pass
    report["gates"]["target_independent_selector"] = False
    report["gates"]["sound_commit_certificate"] = False

    report["revised_oracle_envelope"] = {
        "repair_fraction": repair_fraction,
        "committed_tokens_per_shared_repair": committed_tokens,
        "projected_traffic_gib_per_token": projected_traffic,
        "traffic_limit_gib_per_token": traffic_limit,
        "traffic_pass": projected_traffic <= traffic_limit,
        "projected_compute_gflop_per_token": projected_compute,
        "compute_limit_gflop_per_token": compute_limit,
        "compute_pass": projected_compute <= compute_limit,
        "memory_pass": bool(report["memory"]["pass"]),
    }
    report["observed_component_decision"] = result["decision"]
    report["rejected_mechanisms"] = [
        "exact-span Atlas warm-decode fast path",
        "rank-32 approximate capsule plus exact layer-suffix repair",
        "rank-32 approximate capsule plus output-row tile repair",
        "rank-32 residual-energy 2D tile repair",
        "rank-32 exact-target adjoint 2D tile repair per token",
        "the original VORTEX-WAVE-1 25-percent repair design point",
    ]
    report["next_candidate"] = (
        "target-independent block repair selector plus sound causal-prefix "
        "certificate"
    )
    report["status"] = (
        "original-wave1-rejected-logical-block-oracle-survives"
        if logical_pass
        else "block-shared-combined-oracle-rejected"
    )
    return report


def _apply_observed_efficiency(
    report: dict[str, Any],
    *,
    observed: float,
    repair_fraction_per_token: float,
    source: Path,
    decision: str,
    exact_token_match: bool,
    warm_decode_fast_fraction: float | None,
    status: str,
    rejected_mechanisms: list[str],
    next_candidate: str | None = None,
) -> dict[str, Any]:
    required = float(
        report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )
    maximum_compute_fraction = float(
        report["mechanism"]["maximum_compute_repair_fraction"]
    )
    traffic_pass = observed >= required
    compute_pass = repair_fraction_per_token <= maximum_compute_fraction
    mechanism_pass = traffic_pass and compute_pass

    report["mechanism"].update(
        {
            "observed": observed,
            "observed_repair_fraction": repair_fraction_per_token,
            "observed_source": _portable_source(source),
            "traffic_pass": traffic_pass,
            "compute_pass": compute_pass,
            "pass": mechanism_pass,
            "traffic_shortfall_factor": (
                required / observed if observed > 0 else float("inf")
            ),
            "compute_excess_factor": (
                repair_fraction_per_token / maximum_compute_fraction
                if maximum_compute_fraction > 0
                else float("inf")
            ),
            "observed_exact_token_match": exact_token_match,
            "observed_warm_decode_fast_fraction": warm_decode_fast_fraction,
        }
    )
    report["gates"]["observed_repair_traffic"] = traffic_pass
    report["gates"]["observed_repair_compute"] = compute_pass
    report["gates"]["observed_repair_efficiency"] = mechanism_pass
    report["observed_component_decision"] = decision
    report["rejected_mechanisms"] = rejected_mechanisms
    if next_candidate is not None:
        report["next_candidate"] = next_candidate

    analytic_pass = (
        bool(report["gates"]["memory"])
        and bool(report["gates"]["analytic_traffic"])
        and bool(report["gates"]["analytic_compute"])
    )
    if not analytic_pass:
        report["status"] = "rejected-vortex-wave-1-analytic-envelope"
    elif not mechanism_pass:
        report["status"] = status
    else:
        report["status"] = "gate0-candidate-ready-for-e2-falsification"
    return report


def apply_real_model_observation(
    report: dict[str, Any],
    exact_span_path: str | Path = DEFAULT_EXACT_SPAN_RESULT,
    oracle_path: str | Path = DEFAULT_ORACLE_RESULT,
    residual_path: str | Path = DEFAULT_RESIDUAL_RESULT,
    adjoint_path: str | Path = DEFAULT_ADJOINT_RESULT,
    combined_path: str | Path = DEFAULT_COMBINED_RESULT,
    residual_selector_path: str | Path = DEFAULT_RESIDUAL_SELECTOR_RESULT,
) -> dict[str, Any]:
    """Apply the strongest committed E1 evidence to the Gate 0 report."""

    combined = Path(combined_path)
    if combined.exists():
        report = _apply_combined_oracle(report, combined)
        selector = Path(residual_selector_path)
        if selector.exists():
            report = _attach_residual_selector_falsification(report, selector)
        return report

    adjoint = Path(adjoint_path)
    if adjoint.exists():
        result = json.loads(adjoint.read_text(encoding="utf-8"))
        observed = float(
            result["best_observed_tokens_per_full_repair_equivalent"]
        )
        repair = float(
            result["positive_signed_margin_ranking"][
                "first_repair_match"
            ]["full_model_repair_fraction_per_token"]
        )
        return _apply_observed_efficiency(
            report,
            observed=observed,
            repair_fraction_per_token=repair,
            source=adjoint,
            decision=str(result["decision"]),
            exact_token_match=True,
            warm_decode_fast_fraction=0.0,
            status="rejected-per-token-rank32-local-repair",
            rejected_mechanisms=[
                "exact-span Atlas warm-decode fast path",
                "rank-32 approximate capsule plus exact layer-suffix repair",
                "rank-32 approximate capsule plus output-row tile repair",
                "rank-32 residual-energy 2D tile repair",
                "rank-32 exact-target adjoint 2D tile repair per token",
            ],
            next_candidate="block-shared combined traffic/compute oracle",
        )

    residual = Path(residual_path)
    if residual.exists():
        result = json.loads(residual.read_text(encoding="utf-8"))
        first = result["first_repair_match"]
        observed = float(first["tokens_per_full_repair_equivalent"])
        repair = float(first["full_model_repair_fraction_per_token"])
        return _apply_observed_efficiency(
            report,
            observed=observed,
            repair_fraction_per_token=repair,
            source=residual,
            decision=str(result["decision"]),
            exact_token_match=True,
            warm_decode_fast_fraction=0.0,
            status="rejected-residual-energy-tile-selector",
            rejected_mechanisms=[
                "exact-span Atlas warm-decode fast path",
                "rank-32 approximate capsule plus exact layer-suffix repair",
                "rank-32 approximate capsule plus output-row tile repair",
                "rank-32 residual-energy 2D tile repair",
            ],
            next_candidate="final-token adjoint 2D tile repair",
        )

    oracle = Path(oracle_path)
    if oracle.exists():
        result = json.loads(oracle.read_text(encoding="utf-8"))
        first = result["row_tile_oracle"]["first_repair_match"]
        observed = float(first["tokens_per_full_repair_equivalent"])
        repair = float(first["full_model_repair_fraction_per_token"])
        return _apply_observed_efficiency(
            report,
            observed=observed,
            repair_fraction_per_token=repair,
            source=oracle,
            decision=str(result["overall_decision"]),
            exact_token_match=True,
            warm_decode_fast_fraction=0.0,
            status="rejected-rank32-layer-and-row-tile-repair",
            rejected_mechanisms=[
                "exact-span Atlas warm-decode fast path",
                "rank-32 approximate capsule plus exact layer-suffix repair",
                "rank-32 approximate capsule plus output-row tile repair",
            ],
        )

    exact_span = Path(exact_span_path)
    if not exact_span.exists():
        return report

    result = json.loads(exact_span.read_text(encoding="utf-8"))
    warm = result["warm_decode_repair"]
    observed = float(warm["tokens_per_full_repair_equivalent"])
    warm_tokens = float(result["warm_decode_tokens"])
    repair = float(warm["full_model_repair_fraction"]) / warm_tokens
    return _apply_observed_efficiency(
        report,
        observed=observed,
        repair_fraction_per_token=repair,
        source=exact_span,
        decision=str(result["decision"]),
        exact_token_match=bool(result["exact_token_match"]),
        warm_decode_fast_fraction=float(
            result["aggregate"]["warm_decode_fast_fraction"]
        ),
        status="rejected-exact-span-hot-path",
        rejected_mechanisms=["exact-span Atlas warm-decode fast path"],
    )
