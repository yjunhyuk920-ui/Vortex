"""Pure accounting and aggregation helpers for EXP-077A.

The heavyweight Qwen runner deliberately lives under ``experiments/exp_077a``.
This module keeps the registered fraction, percentile, and quality Gate logic
independent of Torch so CI can test the contract without model dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Sequence


class FractalOracleError(ValueError):
    """Raised when an EXP-077A contract input is malformed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_rank(values: Iterable[float | int], percentile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise FractalOracleError("percentile population must be nonempty")
    if not 0.0 <= percentile <= 1.0:
        raise FractalOracleError("percentile must be in [0, 1]")
    rank = max(1, math.ceil(percentile * len(rows)))
    return rows[rank - 1]


def selected_channel_count(intermediate_size: int, fraction: float) -> int:
    if intermediate_size <= 0:
        raise FractalOracleError("intermediate size must be positive")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise FractalOracleError("fraction must be finite and in (0, 1]")
    # The registered fraction is a hard budget ceiling, so fractional channel
    # counts round down. A one-channel minimum only affects tiny test shapes.
    return min(intermediate_size, max(1, math.floor(intermediate_size * fraction)))


def selected_parameter_fraction(
    *, hidden_size: int, intermediate_size: int, selected_channels: int
) -> float:
    if hidden_size <= 0 or intermediate_size <= 0:
        raise FractalOracleError("dimensions must be positive")
    if not 0 < selected_channels <= intermediate_size:
        raise FractalOracleError("selected channels outside intermediate width")
    full = 3 * hidden_size * intermediate_size
    selected = 3 * hidden_size * selected_channels
    return selected / full


def registered_teacher_forcing_tokens(
    trace: dict[str, Any], *, token_count: int
) -> tuple[list[int], list[int]]:
    """Replay EXP-076's exact target-verification conditioning.

    EXP-076 verifies ``[first_target_token, *proposal_tokens]`` against a
    cloned target cache. Its verification outputs are labels, not the next
    inputs. For N scored positions, the full-sequence equivalent therefore
    appends N-1 conditioning tokens while expecting the prefix token plus the
    first N-1 verification outputs.
    """
    if token_count <= 0:
        raise FractalOracleError("token count must be positive")
    first = trace.get("first_target_token")
    proposals = trace.get("proposal_tokens")
    verification = trace.get("target_verification_tokens")
    if not isinstance(first, int):
        raise FractalOracleError("missing first target token")
    if not isinstance(proposals, list) or not all(
        isinstance(value, int) for value in proposals
    ):
        raise FractalOracleError("missing proposal token trace")
    if not isinstance(verification, list) or not all(
        isinstance(value, int) for value in verification
    ):
        raise FractalOracleError("missing target verification trace")
    required_suffix = token_count - 1
    if len(proposals) < max(0, required_suffix - 1):
        raise FractalOracleError("proposal trace is too short")
    if len(verification) < required_suffix:
        raise FractalOracleError("verification trace is too short")
    conditioning = [first, *proposals[: max(0, required_suffix - 1)]]
    expected = [first, *verification[:required_suffix]]
    return conditioning[:required_suffix], expected


def homogeneous_length_batches(lengths: Sequence[int]) -> list[list[int]]:
    """Return stable index groups that never require recurrent-state padding."""
    if not lengths or any(not isinstance(value, int) or value <= 0 for value in lengths):
        raise FractalOracleError("sequence lengths must be nonempty positive integers")
    groups: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    return list(groups.values())


def validate_prompt_and_trace_ids(
    prompts: dict[str, Any], trace_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    for split in ("build", "evaluation"):
        rows = prompts.get(split)
        if not isinstance(rows, list) or not rows:
            raise FractalOracleError(f"missing prompt split: {split}")
        for row in rows:
            if row.get("id") is None or row.get("family") is None:
                raise FractalOracleError("prompt row is missing id/family")
            prompt_rows.append({**row, "split": split})
    prompt_by_id = {str(row["id"]): row for row in prompt_rows}
    if len(prompt_by_id) != len(prompt_rows):
        raise FractalOracleError("prompt ids must be unique")
    trace_by_id: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id or prompt_id in trace_by_id:
            raise FractalOracleError("trace prompt ids must be nonempty and unique")
        trace_by_id[prompt_id] = row
    if set(prompt_by_id) != set(trace_by_id):
        raise FractalOracleError("prompt and trace id sets differ")
    for prompt_id, prompt in prompt_by_id.items():
        trace = trace_by_id[prompt_id]
        if trace.get("split") != prompt["split"]:
            raise FractalOracleError(f"split mismatch for {prompt_id}")
        if trace.get("family") != prompt["family"]:
            raise FractalOracleError(f"family mismatch for {prompt_id}")
        if not isinstance(trace.get("first_target_token"), int):
            raise FractalOracleError(f"missing first target token for {prompt_id}")
        if not isinstance(trace.get("target_verification_tokens"), list):
            raise FractalOracleError(f"missing target trace for {prompt_id}")
        if not isinstance(trace.get("proposal_tokens"), list):
            raise FractalOracleError(f"missing proposal trace for {prompt_id}")
    return {
        "prompt_count": len(prompt_rows),
        "build_count": len(prompts["build"]),
        "evaluation_count": len(prompts["evaluation"]),
        "families": sorted({str(row["family"]) for row in prompt_rows}),
        "joined_core_sha256": canonical_sha256(
            [
                {
                    "prompt_id": prompt_id,
                    "split": prompt_by_id[prompt_id]["split"],
                    "family": prompt_by_id[prompt_id]["family"],
                    "first_target_token": trace_by_id[prompt_id][
                        "first_target_token"
                    ],
                    "proposal_tokens": trace_by_id[prompt_id]["proposal_tokens"],
                    "target_verification_tokens": trace_by_id[prompt_id][
                        "target_verification_tokens"
                    ],
                }
                for prompt_id in sorted(prompt_by_id)
            ]
        ),
    }


def aggregate_quality_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise FractalOracleError("quality rows are empty")
    token_count = sum(int(row["token_count"]) for row in rows)
    if token_count <= 0:
        raise FractalOracleError("token count must be positive")
    kls: list[float] = []
    for row in rows:
        row_kls = [float(value) for value in row["token_kls"]]
        if len(row_kls) != int(row["token_count"]):
            raise FractalOracleError("token KL count mismatch")
        if any(not math.isfinite(value) or value < -1e-6 for value in row_kls):
            raise FractalOracleError("invalid token KL")
        kls.extend(max(0.0, value) for value in row_kls)
    matches = sum(int(row["top1_matches"]) for row in rows)
    relative_errors = [float(row["mlp_relative_l2"]) for row in rows]
    if any(not math.isfinite(value) or value < 0.0 for value in relative_errors):
        raise FractalOracleError("invalid relative error")
    return {
        "case_count": len(rows),
        "token_count": token_count,
        "top1_matches": matches,
        "top1_agreement": matches / token_count,
        "mean_kl": sum(kls) / len(kls),
        "p95_kl": nearest_rank(kls, 0.95),
        "mlp_relative_l2_p50": nearest_rank(relative_errors, 0.50),
        "mlp_relative_l2_p95": nearest_rank(relative_errors, 0.95),
    }


def family_aggregates(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families = sorted({str(row["family"]) for row in rows})
    return {
        family: aggregate_quality_rows(
            [row for row in rows if str(row["family"]) == family]
        )
        for family in families
    }


def quality_gate(
    *,
    population: dict[str, Any],
    families: dict[str, dict[str, Any]],
    baseline_mismatches: int,
    selected_fraction: float,
    max_fraction: float,
    min_top1_agreement: float,
    min_family_top1_agreement: float,
    max_mean_kl: float,
    max_p95_kl: float,
) -> dict[str, bool]:
    controls_passed = baseline_mismatches == 0
    budget_passed = selected_fraction <= max_fraction + 1e-12
    population_passed = (
        float(population["top1_agreement"]) >= min_top1_agreement
        and float(population["mean_kl"]) <= max_mean_kl
        and float(population["p95_kl"]) <= max_p95_kl
    )
    family_passed = bool(families) and all(
        float(row["top1_agreement"]) >= min_family_top1_agreement
        for row in families.values()
    )
    return {
        "controls_passed": controls_passed,
        "budget_passed": budget_passed,
        "population_passed": population_passed,
        "family_passed": family_passed,
    }


def gate_decision(gate: dict[str, bool]) -> str:
    if not gate.get("controls_passed", False):
        return "INVALID_FRACTAL_ORACLE_CONTROL_FAILURE"
    if all(
        gate.get(name, False)
        for name in ("budget_passed", "population_passed", "family_passed")
    ):
        return "PROMOTE_TO_FRACTAL_ALL_OPERATOR_AND_SELECTOR_COST_GATE"
    return "REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH"
