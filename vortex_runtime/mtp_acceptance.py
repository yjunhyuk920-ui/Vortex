"""Pure accounting and correctness helpers for EXP-076.

The real Qwen runner lives under ``experiments/exp_076`` so the repository test
suite does not need the heavyweight, pinned model environment.  This module
contains the fail-closed logic that can be tested independently.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Sequence


class MtpAcceptanceError(ValueError):
    """Raised when an EXP-076 contract input is malformed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_prompt_manifest(
    manifest: dict[str, Any],
    *,
    required_families: Sequence[str],
    build_per_family: int,
    evaluation_per_family: int,
) -> dict[str, Any]:
    required = set(required_families)
    if not required or len(required) != len(required_families):
        raise MtpAcceptanceError("required prompt families must be unique")
    seen_ids: set[str] = set()
    counts: dict[str, dict[str, int]] = {
        split: {family: 0 for family in required_families}
        for split in ("build", "evaluation")
    }
    prompt_hashes: dict[str, str] = {}
    for split in ("build", "evaluation"):
        rows = manifest.get(split)
        if not isinstance(rows, list):
            raise MtpAcceptanceError(f"missing prompt split: {split}")
        for row in rows:
            if not isinstance(row, dict):
                raise MtpAcceptanceError("prompt rows must be objects")
            prompt_id = row.get("id")
            family = row.get("family")
            prompt = row.get("prompt")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise MtpAcceptanceError("prompt id must be nonempty")
            if prompt_id in seen_ids:
                raise MtpAcceptanceError(f"duplicate prompt id: {prompt_id}")
            if family not in required:
                raise MtpAcceptanceError(f"unregistered prompt family: {family}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise MtpAcceptanceError(f"empty prompt: {prompt_id}")
            seen_ids.add(prompt_id)
            counts[split][family] += 1
            prompt_hashes[prompt_id] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
    for family in required_families:
        if counts["build"][family] != build_per_family:
            raise MtpAcceptanceError(f"wrong build count for {family}")
        if counts["evaluation"][family] != evaluation_per_family:
            raise MtpAcceptanceError(f"wrong evaluation count for {family}")
    return {
        "counts": counts,
        "prompt_hashes": prompt_hashes,
        "manifest_core_sha256": canonical_sha256(manifest),
    }


def longest_matching_prefix(
    proposal_tokens: Sequence[int], target_tokens: Sequence[int]
) -> int:
    length = 0
    for proposal, target in zip(proposal_tokens, target_tokens):
        if int(proposal) != int(target):
            break
        length += 1
    return length


def exact_commit_tokens(
    first_target_token: int,
    proposal_tokens: Sequence[int],
    target_verification_tokens: Sequence[int],
) -> list[int]:
    """Commit exact target token, matching drafts, then correction/bonus.

    ``target_verification_tokens`` must contain one token beyond the proposal so
    an all-accepted proposal can still commit the exact bonus token.
    """
    if len(target_verification_tokens) < len(proposal_tokens) + 1:
        raise MtpAcceptanceError("verification must include correction/bonus token")
    accepted = longest_matching_prefix(proposal_tokens, target_verification_tokens)
    correction = int(target_verification_tokens[accepted])
    return [int(first_target_token), *map(int, proposal_tokens[:accepted]), correction]


def nearest_rank(values: Iterable[float | int], percentile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise MtpAcceptanceError("percentile population must be nonempty")
    if not 0.0 <= percentile <= 1.0:
        raise MtpAcceptanceError("percentile must be in [0, 1]")
    rank = max(1, math.ceil(percentile * len(rows)))
    return rows[rank - 1]


def ideal_minimum_accepted_tokens(
    *,
    target_block_parameter_reads: int,
    baseline_parameters: int,
    allowed_multiplier: float,
    proposal_parameters_per_token: int,
) -> int | None:
    if target_block_parameter_reads <= 0 or baseline_parameters <= 0:
        raise MtpAcceptanceError("parameter counts must be positive")
    if proposal_parameters_per_token < 0:
        raise MtpAcceptanceError("proposal parameters must be nonnegative")
    allowance = allowed_multiplier * baseline_parameters
    remaining = allowance - proposal_parameters_per_token
    if not math.isfinite(remaining) or remaining <= 0:
        return None
    return math.ceil(target_block_parameter_reads / remaining)


def realized_normalized_parameter_traffic(
    *,
    configured_k: int,
    accepted_prefix: int,
    target_block_parameter_reads: int,
    proposal_parameters_per_token: int,
    baseline_parameters: int,
) -> float | None:
    """Fully charge rejected proposals under the EXP-074 token convention.

    EXP-074 amortized a target block over perfectly accepted draft tokens and
    charged one proposal-model parameter stream per proposed token.  Here K can
    exceed the accepted prefix, so all K proposal streams remain in the numerator.
    Zero accepted drafts have no finite normalized value under that convention.
    """
    if configured_k <= 0 or accepted_prefix < 0 or accepted_prefix > configured_k:
        raise MtpAcceptanceError("invalid K or accepted prefix")
    if baseline_parameters <= 0 or target_block_parameter_reads <= 0:
        raise MtpAcceptanceError("parameter counts must be positive")
    if proposal_parameters_per_token < 0:
        raise MtpAcceptanceError("proposal parameters must be nonnegative")
    if accepted_prefix == 0:
        return None
    total = target_block_parameter_reads + configured_k * proposal_parameters_per_token
    return total / (accepted_prefix * baseline_parameters)


def derive_k_row(
    *,
    configured_k: int,
    proposal_tokens: Sequence[int],
    target_verification_tokens: Sequence[int],
    target_block_parameter_reads: int,
    proposal_parameters_per_token: int,
    baseline_parameters: int,
) -> dict[str, Any]:
    if configured_k > len(proposal_tokens):
        raise MtpAcceptanceError("configured K exceeds proposal length")
    if configured_k >= len(target_verification_tokens):
        raise MtpAcceptanceError("verification lacks correction/bonus token")
    proposal = list(map(int, proposal_tokens[:configured_k]))
    targets = list(map(int, target_verification_tokens[: configured_k + 1]))
    accepted = longest_matching_prefix(proposal, targets)
    normalized = realized_normalized_parameter_traffic(
        configured_k=configured_k,
        accepted_prefix=accepted,
        target_block_parameter_reads=target_block_parameter_reads,
        proposal_parameters_per_token=proposal_parameters_per_token,
        baseline_parameters=baseline_parameters,
    )
    return {
        "configured_k": configured_k,
        "accepted_prefix": accepted,
        "zero_accept": accepted == 0,
        "all_drafts_accepted": accepted == configured_k,
        "first_correction_token": targets[accepted],
        "normalized_parameter_traffic": normalized,
        "proposal_tokens": proposal,
        "target_verification_tokens": targets,
        "per_position_accepted": [position < accepted for position in range(configured_k)],
    }


def select_build_k(rows: Sequence[dict[str, Any]], registered_k: Sequence[int]) -> int:
    """Select one fixed K on build prompts using a preregistered minimax rule."""
    if not registered_k:
        raise MtpAcceptanceError("registered K set is empty")
    candidates: list[tuple[tuple[float, float, float, int], int]] = []
    for configured_k in registered_k:
        subset = [row for row in rows if int(row["configured_k"]) == configured_k]
        if not subset:
            raise MtpAcceptanceError(f"missing build rows for K={configured_k}")
        traffic = [row["normalized_parameter_traffic"] for row in subset]
        finite = [float(value) for value in traffic if value is not None]
        zero_rate = sum(value is None for value in traffic) / len(traffic)
        p50 = nearest_rank(finite, 0.50) if finite else math.inf
        p95 = nearest_rank(finite, 0.95) if len(finite) == len(traffic) else math.inf
        score = (p95, p50, zero_rate, int(configured_k))
        candidates.append((score, int(configured_k)))
    return min(candidates)[1]


def acceptance_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise MtpAcceptanceError("acceptance rows are empty")
    accepted = [int(row["accepted_prefix"]) for row in rows]
    traffic = [row["normalized_parameter_traffic"] for row in rows]
    finite = [float(value) for value in traffic if value is not None]
    max_k = max(int(row["configured_k"]) for row in rows)
    return {
        "count": len(rows),
        "accepted_prefix_p05": nearest_rank(accepted, 0.05),
        "accepted_prefix_p50": nearest_rank(accepted, 0.50),
        "accepted_prefix_p95": nearest_rank(accepted, 0.95),
        "accepted_prefix_max": max(accepted),
        "zero_accept_rate": sum(value == 0 for value in accepted) / len(accepted),
        "normalized_traffic_p50": (
            nearest_rank(finite, 0.50) if len(finite) == len(rows) else None
        ),
        "normalized_traffic_p95": (
            nearest_rank(finite, 0.95) if len(finite) == len(rows) else None
        ),
        "per_position_acceptance": [
            sum(value >= position for value in accepted) / len(accepted)
            for position in range(1, max_k + 1)
        ],
    }


def gate_decision(
    *,
    integrity_passed: bool,
    acceptance_passed: bool,
    family_passed: bool,
    traffic_passed: bool,
) -> str:
    if not integrity_passed:
        return "INVALID_NATIVE_MTP_EXECUTION_CONTROL_FAILURE"
    if acceptance_passed and family_passed and traffic_passed:
        return "PROMOTE_TO_QWEN35_35B_A3B_ROUTE_UNION_TRACE_GATE"
    return "REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE"
