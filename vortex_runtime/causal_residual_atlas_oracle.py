"""Pure reference helpers for the EXP-083A favorable Atlas Gate.

The heavyweight checkpoint runner lives under ``experiments/exp_083a``.  This
module fixes the prompt-only SVD convention, native-anchored page arithmetic,
and fail-closed aggregate decision independently of Torch and the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np


PROMOTE_DECISION = (
    "PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE"
)
REJECT_DECISION = "REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH"
INVALID_DECISION = "INVALID_CAUSAL_RESIDUAL_ATLAS_ORACLE_CONTROL_FAILURE"
INFRASTRUCTURE_DECISION = "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("percentile population must be nonempty")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    rank = max(1, math.ceil(percentile * len(rows)))
    return rows[rank - 1]


@dataclass(frozen=True)
class CanonicalPrefixBasis:
    basis: np.ndarray
    singular_values: np.ndarray
    numerical_cutoff: float
    numerical_rank: int
    retained_rank: int


def canonical_prefix_basis(
    prefix_inputs: np.ndarray,
    *,
    maximum_rank: int,
) -> CanonicalPrefixBasis:
    """Return the frozen prompt-only float64 SVD basis.

    The numerical threshold and sign convention exactly match the
    preregistration.  ``numpy.argmax`` supplies the lowest-index tie break for
    equal absolute coordinates.
    """

    source = np.asarray(prefix_inputs, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("prefix_inputs must be a nonempty matrix")
    if not np.all(np.isfinite(source)):
        raise ValueError("prefix_inputs must be finite")
    if maximum_rank <= 0:
        raise ValueError("maximum_rank must be positive")

    _, singular_values, right_vectors = np.linalg.svd(
        source,
        full_matrices=False,
    )
    sigma_max = float(singular_values[0])
    cutoff = (
        max(source.shape)
        * float(np.finfo(np.float64).eps)
        * sigma_max
    )
    numerical_rank = int(np.count_nonzero(singular_values > cutoff))
    retained_rank = min(maximum_rank, numerical_rank)
    basis = np.ascontiguousarray(right_vectors[:retained_rank].T)
    for column_index in range(basis.shape[1]):
        column = basis[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            basis[:, column_index] *= -1.0
    if basis.shape[1] and not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[1]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("SVD basis is not orthonormal")
    return CanonicalPrefixBasis(
        basis=basis,
        singular_values=np.ascontiguousarray(singular_values),
        numerical_cutoff=cutoff,
        numerical_rank=numerical_rank,
        retained_rank=retained_rank,
    )


@dataclass(frozen=True)
class NativeAnchoredPages:
    input_residual: np.ndarray
    residual_image: np.ndarray
    page_outputs: np.ndarray
    exact_unread_error_norms: np.ndarray
    certified_unread_radii: np.ndarray
    block_frobenius_bounds: np.ndarray
    block_residual_norms: np.ndarray
    all_page_output: np.ndarray


def native_anchored_pages(
    weight: np.ndarray,
    basis: np.ndarray,
    current_input: np.ndarray,
    native_output: np.ndarray,
    *,
    page_columns: int,
) -> NativeAnchoredPages:
    """Enumerate the frozen illegal native-anchored one-page candidates."""

    matrix = np.asarray(weight, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    vector = np.asarray(current_input, dtype=np.float64)
    native = np.asarray(native_output, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("weight must be a nonempty matrix")
    if vector.shape != (matrix.shape[1],):
        raise ValueError("current_input shape mismatch")
    if native.shape != (matrix.shape[0],):
        raise ValueError("native_output shape mismatch")
    if q.ndim != 2 or q.shape[0] != matrix.shape[1]:
        raise ValueError("basis shape mismatch")
    if page_columns <= 0:
        raise ValueError("page_columns must be positive")
    if not all(
        np.all(np.isfinite(value))
        for value in (matrix, q, vector, native)
    ):
        raise ValueError("weight, basis, input, and output must be finite")
    if q.shape[1] and not np.allclose(
        q.T @ q,
        np.eye(q.shape[1]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("basis must be orthonormal")

    input_residual = vector - q @ (q.T @ vector)
    residual_image = matrix @ input_residual
    contributions: list[np.ndarray] = []
    bounds: list[float] = []
    residual_norms: list[float] = []
    for start in range(0, matrix.shape[1], page_columns):
        stop = min(matrix.shape[1], start + page_columns)
        block = matrix[:, start:stop]
        residual_block = input_residual[start:stop]
        contributions.append(block @ residual_block)
        bounds.append(float(np.linalg.norm(block, ord="fro")))
        residual_norms.append(float(np.linalg.norm(residual_block)))

    contribution_matrix = np.stack(contributions, axis=0)
    page_outputs = (
        native[None, :] - residual_image[None, :] + contribution_matrix
    )
    exact_errors = np.linalg.norm(
        residual_image[None, :] - contribution_matrix,
        axis=1,
    )
    block_bounds = np.asarray(bounds, dtype=np.float64)
    block_residual_norms = np.asarray(residual_norms, dtype=np.float64)
    total_radius = float(np.sum(block_bounds * block_residual_norms))
    radii = np.maximum(
        0.0,
        total_radius - block_bounds * block_residual_norms,
    )
    all_page_output = (
        native - residual_image + contribution_matrix.sum(axis=0)
    )
    return NativeAnchoredPages(
        input_residual=np.ascontiguousarray(input_residual),
        residual_image=np.ascontiguousarray(residual_image),
        page_outputs=np.ascontiguousarray(page_outputs),
        exact_unread_error_norms=np.ascontiguousarray(exact_errors),
        certified_unread_radii=np.ascontiguousarray(radii),
        block_frobenius_bounds=np.ascontiguousarray(block_bounds),
        block_residual_norms=np.ascontiguousarray(block_residual_norms),
        all_page_output=np.ascontiguousarray(all_page_output),
    )


def target_to_candidate_kls(
    target_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> np.ndarray:
    """Return KL(target || candidate) for every candidate row."""

    target = np.asarray(target_logits, dtype=np.float64)
    candidates = np.asarray(candidate_logits, dtype=np.float64)
    if target.ndim != 1 or target.size < 2:
        raise ValueError("target_logits must be a vector of size >= 2")
    if candidates.ndim != 2 or candidates.shape[1] != target.size:
        raise ValueError("candidate_logits must be [candidates, vocabulary]")
    if candidates.shape[0] == 0:
        raise ValueError("candidate population must be nonempty")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(candidates)):
        raise ValueError("logits must be finite")

    target_shifted = target - float(np.max(target))
    target_log_probs = target_shifted - math.log(
        float(np.exp(target_shifted).sum())
    )
    target_probs = np.exp(target_log_probs)
    rows = []
    for logits in candidates:
        shifted = logits - float(np.max(logits))
        candidate_log_probs = shifted - math.log(float(np.exp(shifted).sum()))
        divergence = float(
            np.sum(target_probs * (target_log_probs - candidate_log_probs))
        )
        rows.append(max(0.0, divergence))
    return np.asarray(rows, dtype=np.float64)


def summarize_gate(
    branch_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
    *,
    expected_token_states: int,
    expected_projection_branches: int,
    expected_families: Sequence[str],
    prompts_per_family: int,
    maximum_mean_kl: float,
    maximum_p95_kl: float,
    control_failures: Sequence[str] = (),
    leakage_failures: Sequence[str] = (),
    malformed_state_count: int = 0,
    execution_complete: bool,
) -> dict[str, Any]:
    """Aggregate a complete pass or the first valid scientific failure."""

    if expected_token_states <= 0 or expected_projection_branches <= 0:
        raise ValueError("expected populations must be positive")
    if prompts_per_family <= 0 or not expected_families:
        raise ValueError("family contract must be nonempty")
    if malformed_state_count < 0:
        raise ValueError("malformed_state_count must be non-negative")

    kls = [float(row["selected_kl"]) for row in branch_rows]
    if any(not math.isfinite(value) or value < 0 for value in kls):
        raise ValueError("branch KL values must be finite and non-negative")
    branch_successes = sum(bool(row["top1_match"]) for row in branch_rows)
    observed_branch_failure = branch_successes != len(branch_rows)
    observed_token_failure = any(not bool(row["success"]) for row in token_rows)
    mean_kl = sum(kls) / len(kls) if kls else None
    p95_kl = nearest_rank(kls, 0.95) if kls else None

    family_successes = {
        family: sum(
            bool(row["success"])
            for row in token_rows
            if row.get("family") == family
        )
        for family in expected_families
    }
    token_successes = sum(bool(row["success"]) for row in token_rows)
    gates = {
        "zero_control_failures": len(control_failures) == 0,
        "zero_leakage_failures": len(leakage_failures) == 0,
        "zero_malformed_states": malformed_state_count == 0,
        "complete_token_population": (
            execution_complete and len(token_rows) == expected_token_states
        ),
        "complete_branch_population": (
            execution_complete
            and len(branch_rows) == expected_projection_branches
        ),
        "all_tokens_succeed": (
            execution_complete and token_successes == expected_token_states
        ),
        "all_branches_preserve_top1": (
            execution_complete
            and branch_successes == expected_projection_branches
        ),
        "all_families_succeed": (
            execution_complete
            and all(
                family_successes[family] == prompts_per_family
                for family in expected_families
            )
        ),
        "mean_kl_within_limit": (
            mean_kl is not None and mean_kl <= maximum_mean_kl
        ),
        "p95_kl_within_limit": (
            p95_kl is not None and p95_kl <= maximum_p95_kl
        ),
    }
    if not (
        gates["zero_control_failures"]
        and gates["zero_leakage_failures"]
        and gates["zero_malformed_states"]
    ):
        decision = INVALID_DECISION
    elif observed_branch_failure or observed_token_failure:
        decision = REJECT_DECISION
    elif not execution_complete:
        decision = INFRASTRUCTURE_DECISION
    elif all(gates.values()):
        decision = PROMOTE_DECISION
    else:
        decision = REJECT_DECISION

    return {
        "decision": decision,
        "execution_complete": execution_complete,
        "evaluated_token_states": len(token_rows),
        "evaluated_projection_branches": len(branch_rows),
        "token_successes": token_successes,
        "branch_successes": branch_successes,
        "family_successes": family_successes,
        "selected_mean_kl": mean_kl,
        "selected_p95_kl": p95_kl,
        "control_failures": list(control_failures),
        "leakage_failures": list(leakage_failures),
        "malformed_state_count": malformed_state_count,
        "observed_scientific_failure": (
            observed_branch_failure or observed_token_failure
        ),
        "gates": gates,
    }


def deterministic_core(
    *,
    registered_inputs: dict[str, Any],
    basis_rows: Sequence[dict[str, Any]],
    page_rows: Sequence[dict[str, Any]],
    branch_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Return the wall-clock-free evidence payload hashed by runner/replay."""

    return {
        "registered_inputs": registered_inputs,
        "basis_rows": list(basis_rows),
        "page_rows": list(page_rows),
        "branch_rows": list(branch_rows),
        "token_rows": list(token_rows),
        "control_rows": list(control_rows),
        "gate": gate,
    }
