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


def _portable_source(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _apply_observed_efficiency(
    report: dict[str, Any],
    *,
    observed: float,
    source: Path,
    decision: str,
    exact_token_match: bool,
    warm_decode_fast_fraction: float | None,
    status: str,
    rejected_mechanisms: list[str],
) -> dict[str, Any]:
    required = float(
        report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ]
    )
    report["mechanism"].update(
        {
            "observed": observed,
            "observed_source": _portable_source(source),
            "pass": observed >= required,
            "shortfall_factor": required / observed,
            "observed_exact_token_match": exact_token_match,
            "observed_warm_decode_fast_fraction": warm_decode_fast_fraction,
        }
    )
    report["gates"]["observed_repair_efficiency"] = observed >= required
    report["observed_component_decision"] = decision
    report["rejected_mechanisms"] = rejected_mechanisms
    report["status"] = status if observed < required else (
        "gate0-candidate-ready-for-e2-falsification"
    )
    return report


def apply_real_model_observation(
    report: dict[str, Any],
    exact_span_path: str | Path = DEFAULT_EXACT_SPAN_RESULT,
    oracle_path: str | Path = DEFAULT_ORACLE_RESULT,
) -> dict[str, Any]:
    """Apply the strongest committed E1 evidence to the Gate 0 report."""

    oracle = Path(oracle_path)
    if oracle.exists():
        result = json.loads(oracle.read_text(encoding="utf-8"))
        observed = float(
            result["row_tile_oracle"]["first_repair_match"][
                "tokens_per_full_repair_equivalent"
            ]
        )
        return _apply_observed_efficiency(
            report,
            observed=observed,
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
    observed = float(
        result["warm_decode_repair"][
            "tokens_per_full_repair_equivalent"
        ]
    )
    return _apply_observed_efficiency(
        report,
        observed=observed,
        source=exact_span,
        decision=str(result["decision"]),
        exact_token_match=bool(result["exact_token_match"]),
        warm_decode_fast_fraction=float(
            result["aggregate"]["warm_decode_fast_fraction"]
        ),
        status="rejected-exact-span-hot-path",
        rejected_mechanisms=[
            "exact-span Atlas warm-decode fast path",
        ],
    )
