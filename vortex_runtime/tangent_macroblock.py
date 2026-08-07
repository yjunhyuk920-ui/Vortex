"""Pure accounting and Gate helpers for EXP-078A.

The heavyweight unchanged-checkpoint runner lives under ``experiments/exp_078a``.
This module deliberately has no Torch dependency so the construction-amortization
bound and the favorable-oracle decision can be reproduced independently.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


class TangentMacroblockError(ValueError):
    """Raised when an EXP-078A contract input is malformed."""


def nearest_rank(values: Iterable[float | int], percentile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise TangentMacroblockError("percentile population must be nonempty")
    if not 0.0 <= percentile <= 1.0:
        raise TangentMacroblockError("percentile must be in [0, 1]")
    rank = max(1, math.ceil(percentile * len(rows)))
    return rows[rank - 1]


def dense_macroblock_cost(
    *,
    hidden_size: int,
    intermediate_size: int,
    active_paths: int = 1,
) -> dict[str, float | int]:
    """Return MAC and storage ratios for an exact frozen-coefficient MLP map.

    One active SwiGLU path is

    ``Wd @ diag(SiLU(Wg @ anchor)) @ Wu``.

    Multiple active experts sum into the same hidden-by-hidden macro matrix.
    Materialization is counted as the direct scaled matrix product.  No claim is
    made that this is an algebraic lower bound for every possible constructor.
    """
    if hidden_size <= 0 or intermediate_size <= 0 or active_paths <= 0:
        raise TangentMacroblockError("all dimensions and path counts must be positive")
    exact_macs = 3 * active_paths * hidden_size * intermediate_size
    hot_macs = hidden_size * hidden_size
    materialization_macs = (
        active_paths * hidden_size * hidden_size * intermediate_size
    )
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "active_paths": active_paths,
        "exact_macs_per_token": exact_macs,
        "hot_macro_macs_per_token": hot_macs,
        "direct_materialization_macs": materialization_macs,
        "hot_fraction_of_exact": hot_macs / exact_macs,
        "hot_storage_fraction_of_exact_mlp_weights": hot_macs / exact_macs,
        "materialization_exact_token_equivalents": materialization_macs
        / exact_macs,
    }


def charged_cycle_fraction(
    *,
    hot_tokens: int,
    hot_fraction: float,
    construction_token_equivalents: float,
    exact_anchor_token_equivalents: float = 1.0,
) -> float:
    """Average exact-MLP-equivalent work for one anchor plus hot-token cycle."""
    if hot_tokens < 0:
        raise TangentMacroblockError("hot token count must be nonnegative")
    values = (hot_fraction, construction_token_equivalents, exact_anchor_token_equivalents)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise TangentMacroblockError("cost terms must be finite and nonnegative")
    return (
        exact_anchor_token_equivalents
        + construction_token_equivalents
        + hot_tokens * hot_fraction
    ) / (hot_tokens + 1)


def minimum_hot_tokens(
    *,
    allowance_fraction: float,
    hot_fraction: float,
    construction_token_equivalents: float,
    exact_anchor_token_equivalents: float = 1.0,
) -> int | None:
    """Minimum integer hot-token lifetime whose charged cycle meets allowance."""
    values = (
        allowance_fraction,
        hot_fraction,
        construction_token_equivalents,
        exact_anchor_token_equivalents,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise TangentMacroblockError("cost terms must be finite and nonnegative")
    if allowance_fraction <= hot_fraction:
        return None
    numerator = (
        exact_anchor_token_equivalents
        + construction_token_equivalents
        - allowance_fraction
    )
    if numerator <= 0.0:
        return 0
    candidate = max(0, math.ceil(numerator / (allowance_fraction - hot_fraction)))
    while charged_cycle_fraction(
        hot_tokens=candidate,
        hot_fraction=hot_fraction,
        construction_token_equivalents=construction_token_equivalents,
        exact_anchor_token_equivalents=exact_anchor_token_equivalents,
    ) > allowance_fraction + 1e-15:
        candidate += 1
    while candidate > 0 and charged_cycle_fraction(
        hot_tokens=candidate - 1,
        hot_fraction=hot_fraction,
        construction_token_equivalents=construction_token_equivalents,
        exact_anchor_token_equivalents=exact_anchor_token_equivalents,
    ) <= allowance_fraction + 1e-15:
        candidate -= 1
    return candidate


def valid_prefix_length(
    *,
    target_top_tokens: Sequence[int],
    candidate_top_tokens: Sequence[int],
    token_kls: Sequence[float],
    max_token_kl: float,
) -> int:
    if not (
        len(target_top_tokens) == len(candidate_top_tokens) == len(token_kls)
    ):
        raise TangentMacroblockError("token vectors must have equal length")
    if not math.isfinite(max_token_kl) or max_token_kl < 0.0:
        raise TangentMacroblockError("max token KL must be finite and nonnegative")
    for index, (target, candidate, kl) in enumerate(
        zip(target_top_tokens, candidate_top_tokens, token_kls)
    ):
        value = float(kl)
        if not math.isfinite(value) or value < -1e-6:
            raise TangentMacroblockError("token KL must be finite and nonnegative")
        if int(target) != int(candidate) or max(0.0, value) > max_token_kl:
            return index
    return len(token_kls)


def aggregate_quality_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise TangentMacroblockError("quality rows are empty")
    token_count = sum(int(row["token_count"]) for row in rows)
    if token_count <= 0:
        raise TangentMacroblockError("token count must be positive")
    kls: list[float] = []
    for row in rows:
        row_kls = [max(0.0, float(value)) for value in row["token_kls"]]
        if len(row_kls) != int(row["token_count"]):
            raise TangentMacroblockError("token KL count mismatch")
        if any(not math.isfinite(value) for value in row_kls):
            raise TangentMacroblockError("token KL must be finite")
        kls.extend(row_kls)
    matches = sum(int(row["top1_matches"]) for row in rows)
    valid = [int(row["valid_prefix_length"]) for row in rows]
    mlp_errors = [float(row["mlp_relative_l2"]) for row in rows]
    if any(not math.isfinite(value) or value < 0.0 for value in mlp_errors):
        raise TangentMacroblockError("MLP error must be finite and nonnegative")
    return {
        "case_count": len(rows),
        "token_count": token_count,
        "top1_matches": matches,
        "top1_agreement": matches / token_count,
        "mean_kl": sum(kls) / len(kls),
        "p95_kl": nearest_rank(kls, 0.95),
        "valid_prefix_p05": nearest_rank(valid, 0.05),
        "valid_prefix_p50": nearest_rank(valid, 0.50),
        "valid_prefix_p95": nearest_rank(valid, 0.95),
        "mlp_relative_l2_p50": nearest_rank(mlp_errors, 0.50),
        "mlp_relative_l2_p95": nearest_rank(mlp_errors, 0.95),
    }


def family_aggregates(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families = sorted({str(row["family"]) for row in rows})
    return {
        family: aggregate_quality_rows(
            [row for row in rows if str(row["family"]) == family]
        )
        for family in families
    }


def gate_decision(
    *,
    baseline_mismatches: int,
    population: dict[str, Any],
    families: dict[str, dict[str, Any]],
    observed_horizon: int,
    required_p50_hot_tokens: int | None,
    required_p05_hot_tokens: int | None,
    min_top1_agreement: float,
    min_family_top1_agreement: float,
    max_mean_kl: float,
    max_p95_kl: float,
) -> tuple[dict[str, bool], str]:
    controls = baseline_mismatches == 0
    quality = (
        float(population["top1_agreement"]) >= min_top1_agreement
        and float(population["mean_kl"]) <= max_mean_kl
        and float(population["p95_kl"]) <= max_p95_kl
    )
    family = bool(families) and all(
        float(row["top1_agreement"]) >= min_family_top1_agreement
        for row in families.values()
    )
    lifetime = (
        required_p50_hot_tokens is not None
        and required_p05_hot_tokens is not None
        and float(population["valid_prefix_p50"]) >= required_p50_hot_tokens
        and float(population["valid_prefix_p05"]) >= required_p05_hot_tokens
    )
    censored = (
        quality
        and family
        and int(population["valid_prefix_p05"]) == observed_horizon
        and int(population["valid_prefix_p50"]) == observed_horizon
        and (
            required_p50_hot_tokens is None
            or required_p05_hot_tokens is None
            or observed_horizon < required_p50_hot_tokens
            or observed_horizon < required_p05_hot_tokens
        )
    )
    gates = {
        "controls_passed": controls,
        "population_quality_passed": quality,
        "family_quality_passed": family,
        "required_lifetime_observed": lifetime,
        "all_observations_right_censored": censored,
    }
    if not controls:
        decision = "INVALID_TANGENT_MACROBLOCK_CONTROL_FAILURE"
    elif lifetime and quality and family:
        decision = "PROMOTE_TO_CHARGED_TANGENT_MACROBLOCK_CONSTRUCTION_GATE"
    elif censored:
        decision = "INCONCLUSIVE_TANGENT_LIFETIME_EXTEND_TRACE"
    else:
        decision = "REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH"
    return gates, decision
