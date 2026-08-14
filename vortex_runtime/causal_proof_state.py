"""Pure state, accounting, and Gate helpers for EXP-079A.

EXP-079A studies a cold-backed causal proof state for dense projections.  A
small procedural DCT pilot subspace is mapped through each unchanged matrix and
kept hot.  Everything outside that image remains an exact deferred dependency,
grouped into disjoint output-row L2 balls.  An oracle may reveal complete cold
row blocks, but a decision may never commit while its proof uncertainty is
open.

The heavyweight unchanged-checkpoint runner lives under ``experiments/exp_079a``.
This module deliberately depends only on NumPy and the Python standard library.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np


class CausalProofStateError(ValueError):
    """Raised when an EXP-079A contract input is malformed."""


def nearest_rank(values: Iterable[float | int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CausalProofStateError("percentile population must be nonempty")
    if not 0.0 <= percentile <= 1.0:
        raise CausalProofStateError("percentile must be in [0, 1]")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def packed_q4_bytes(parameter_count: int) -> int:
    if parameter_count <= 0:
        raise CausalProofStateError("parameter count must be positive")
    return math.ceil(parameter_count * 4 / 8)


def target_equivalent_fraction(
    *, baseline_billions: float, target_billions: float, latency_multiple: float
) -> float:
    values = (baseline_billions, target_billions, latency_multiple)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise CausalProofStateError("target-equivalent inputs must be positive")
    return latency_multiple * baseline_billions / target_billions


def dct_pilot_basis(width: int, rank: int) -> np.ndarray:
    """Return the first ``rank`` orthonormal DCT-II directions in float64."""
    if width <= 0 or rank <= 0 or rank > width:
        raise CausalProofStateError("pilot rank must be in [1, width]")
    positions = np.arange(width, dtype=np.float64) + 0.5
    frequencies = np.arange(rank, dtype=np.float64)
    basis = np.cos(np.pi * positions[:, None] * frequencies[None, :] / width)
    basis[:, 0] *= math.sqrt(1.0 / width)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / width)
    return np.ascontiguousarray(basis)


def block_ranges(rows: int, row_block: int) -> tuple[tuple[int, int], ...]:
    if rows <= 0 or row_block <= 0:
        raise CausalProofStateError("row dimensions must be positive")
    return tuple(
        (start, min(rows, start + row_block))
        for start in range(0, rows, row_block)
    )


def pilot_sidecar_bytes(
    *,
    rows: int,
    columns: int,
    pilot_rank: int,
    row_block: int,
    image_scalar_bytes: int = 2,
    norm_scalar_bytes: int = 4,
    matrix_metadata_bytes: int = 32,
) -> int:
    if rows <= 0 or columns <= 0:
        raise CausalProofStateError("matrix dimensions must be positive")
    if pilot_rank <= 0 or pilot_rank > columns:
        raise CausalProofStateError("pilot rank outside input width")
    if min(row_block, image_scalar_bytes, norm_scalar_bytes) <= 0:
        raise CausalProofStateError("sidecar widths must be positive")
    if matrix_metadata_bytes < 0:
        raise CausalProofStateError("metadata bytes must be nonnegative")
    block_count = len(block_ranges(rows, row_block))
    return (
        rows * pilot_rank * image_scalar_bytes
        + block_count * norm_scalar_bytes
        + matrix_metadata_bytes
    )


@dataclass(frozen=True)
class UniformBudgetPlan:
    total_parameter_count: int
    total_q4_bytes: int
    allowance_fraction: float
    allowance_bytes: int
    matrix_count: int
    rows: int
    columns: int
    pilot_rank: int
    row_block: int
    blocks_per_matrix: int
    selected_blocks_per_matrix: int
    selected_rows_per_matrix: int
    sidecar_bytes_per_matrix: int
    sidecar_bytes_total: int
    cold_bytes_per_block: int
    cold_bytes_total: int
    charged_bytes_total: int
    charged_traffic_fraction: float
    charged_operation_fraction: float
    proof_state_bytes: int
    unspent_allowance_bytes: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def uniform_budget_plan(
    *,
    total_parameter_count: int,
    matrix_count: int,
    rows: int,
    columns: int,
    pilot_rank: int,
    row_block: int,
    allowance_fraction: float,
    image_scalar_bytes: int = 2,
    norm_scalar_bytes: int = 4,
    matrix_metadata_bytes: int = 32,
    proof_record_bytes: int = 16,
) -> UniformBudgetPlan:
    """Allocate one global target fraction uniformly across equal matrices.

    All non-registered model work is deliberately free.  Every matrix first
    pays its complete hot sidecar, then receives the same integer number of
    exact Q4 cold row-block reads.  The integer rounding is fail-closed: unused
    bytes are not credited as fractional block work.
    """
    if total_parameter_count <= 0 or matrix_count <= 0:
        raise CausalProofStateError("population counts must be positive")
    if not math.isfinite(allowance_fraction) or not 0.0 < allowance_fraction <= 1.0:
        raise CausalProofStateError("allowance fraction must be in (0, 1]")
    if proof_record_bytes <= 0:
        raise CausalProofStateError("proof record width must be positive")

    total_bytes = packed_q4_bytes(total_parameter_count)
    allowance = math.floor(total_bytes * allowance_fraction)
    per_sidecar = pilot_sidecar_bytes(
        rows=rows,
        columns=columns,
        pilot_rank=pilot_rank,
        row_block=row_block,
        image_scalar_bytes=image_scalar_bytes,
        norm_scalar_bytes=norm_scalar_bytes,
        matrix_metadata_bytes=matrix_metadata_bytes,
    )
    sidecars = matrix_count * per_sidecar
    cold_block = packed_q4_bytes(min(row_block, rows) * columns)
    block_count = len(block_ranges(rows, row_block))
    available = max(0, allowance - sidecars)
    selected_blocks = min(block_count, available // (matrix_count * cold_block))
    selected_rows = min(rows, selected_blocks * row_block)
    cold_total = matrix_count * selected_blocks * cold_block
    charged = sidecars + cold_total

    # Favorable operation accounting: procedural basis construction, selector,
    # error measurement, nonlinear propagation, and fallback are free at E1.
    # Count only pilot projection/application, one radius scan per row block,
    # and exact MACs for selected cold rows.
    operations_per_matrix = (
        columns * pilot_rank
        + rows * pilot_rank
        + block_count
        + selected_rows * columns
    )
    charged_operations = matrix_count * operations_per_matrix
    proof_bytes = matrix_count * block_count * proof_record_bytes
    return UniformBudgetPlan(
        total_parameter_count=total_parameter_count,
        total_q4_bytes=total_bytes,
        allowance_fraction=allowance_fraction,
        allowance_bytes=allowance,
        matrix_count=matrix_count,
        rows=rows,
        columns=columns,
        pilot_rank=pilot_rank,
        row_block=row_block,
        blocks_per_matrix=block_count,
        selected_blocks_per_matrix=selected_blocks,
        selected_rows_per_matrix=selected_rows,
        sidecar_bytes_per_matrix=per_sidecar,
        sidecar_bytes_total=sidecars,
        cold_bytes_per_block=cold_block,
        cold_bytes_total=cold_total,
        charged_bytes_total=charged,
        charged_traffic_fraction=charged / total_bytes,
        charged_operation_fraction=charged_operations / total_parameter_count,
        proof_state_bytes=proof_bytes,
        unspent_allowance_bytes=allowance - charged,
    )


@dataclass(frozen=True)
class PilotEnclosure:
    image: np.ndarray
    residual_block_frobenius: np.ndarray
    row_ranges: tuple[tuple[int, int], ...]


def build_pilot_enclosure(
    weight: Any, *, pilot_basis: Any, row_block: int
) -> PilotEnclosure:
    """Build an exact-real pilot image and disjoint row-block L2 radii."""
    matrix = np.asarray(weight, dtype=np.float64)
    basis = np.asarray(pilot_basis, dtype=np.float64)
    if matrix.ndim != 2 or basis.ndim != 2:
        raise CausalProofStateError("weight and pilot basis must be matrices")
    if matrix.shape[1] != basis.shape[0]:
        raise CausalProofStateError("pilot basis width does not match matrix")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), rtol=0.0, atol=1e-11):
        raise CausalProofStateError("pilot basis must be orthonormal")
    image = matrix @ basis
    residual = matrix - image @ basis.T
    ranges = block_ranges(int(matrix.shape[0]), row_block)
    norms = np.asarray(
        [np.linalg.norm(residual[start:stop], ord="fro") for start, stop in ranges],
        dtype=np.float64,
    )
    return PilotEnclosure(
        image=np.ascontiguousarray(image),
        residual_block_frobenius=norms,
        row_ranges=ranges,
    )


def minimum_remaining_l2_radius(
    residual_block_frobenius: Sequence[float],
    *,
    input_l2: float,
    selected_blocks: int,
) -> float:
    """Return the smallest registered sound radius after whole-block reveals.

    Row blocks occupy disjoint output coordinates.  Their squared L2 bounds can
    therefore be added without discarding cross-coordinate correlation.  The
    best bound-only selector removes the largest squared block radii.
    """
    norms = np.asarray(residual_block_frobenius, dtype=np.float64)
    if norms.ndim != 1 or norms.size == 0:
        raise CausalProofStateError("residual block norms must be nonempty")
    if np.any(~np.isfinite(norms)) or np.any(norms < 0.0):
        raise CausalProofStateError("residual block norms must be finite and nonnegative")
    if not math.isfinite(input_l2) or input_l2 < 0.0:
        raise CausalProofStateError("input norm must be finite and nonnegative")
    if selected_blocks < 0 or selected_blocks > norms.size:
        raise CausalProofStateError("selected block count outside population")
    if selected_blocks == norms.size:
        return 0.0
    ordered = np.sort(np.square(norms))[::-1]
    remaining = ordered[selected_blocks:]
    return float(input_l2 * math.sqrt(float(np.sum(remaining))))


def oracle_block_mask(
    residual_output: Any, *, selected_blocks: int, row_block: int
) -> np.ndarray:
    """Return the exact-error-minimizing row mask for each flattened vector.

    This selector is intentionally non-deployable: it sees the complete exact
    residual output.  Failure after this grant is a valid favorable-ceiling
    rejection for the registered pilot and row-block action space.
    """
    residual = np.asarray(residual_output, dtype=np.float64)
    if residual.ndim < 1 or residual.shape[-1] <= 0:
        raise CausalProofStateError("residual output must have a row dimension")
    ranges = block_ranges(int(residual.shape[-1]), row_block)
    if selected_blocks < 0 or selected_blocks > len(ranges):
        raise CausalProofStateError("selected block count outside population")
    flat = residual.reshape(-1, residual.shape[-1])
    mask = np.zeros_like(flat, dtype=bool)
    if selected_blocks:
        energies = np.stack(
            [np.square(flat[:, start:stop]).sum(axis=1) for start, stop in ranges],
            axis=1,
        )
        chosen = np.argpartition(
            energies, kth=len(ranges) - selected_blocks, axis=1
        )[:, -selected_blocks:]
        for vector_index, block_indices in enumerate(chosen):
            for block_index in block_indices:
                start, stop = ranges[int(block_index)]
                mask[vector_index, start:stop] = True
    return mask.reshape(residual.shape)


@dataclass(frozen=True)
class ProofMachineState:
    phase: str
    allowance_bytes: int
    charged_bytes: int
    queried_blocks: int
    total_blocks: int
    winner_margin: float
    remaining_uncertainty: float
    exact_fallback_used: bool = False

    @property
    def budget_remaining(self) -> int:
        return self.allowance_bytes - self.charged_bytes

    @property
    def certifiable(self) -> bool:
        return self.winner_margin > 2.0 * self.remaining_uncertainty

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "budget_remaining": self.budget_remaining,
            "certifiable": self.certifiable,
        }


def start_proof_machine(
    *, allowance_bytes: int, total_blocks: int, winner_margin: float
) -> ProofMachineState:
    if allowance_bytes <= 0 or total_blocks <= 0:
        raise CausalProofStateError("proof capacity must be positive")
    if not math.isfinite(winner_margin) or winner_margin <= 0.0:
        raise CausalProofStateError("winner margin must be positive")
    return ProofMachineState(
        phase="prepared",
        allowance_bytes=allowance_bytes,
        charged_bytes=0,
        queried_blocks=0,
        total_blocks=total_blocks,
        winner_margin=winner_margin,
        remaining_uncertainty=math.inf,
    )


def install_hot_bound(
    state: ProofMachineState, *, sidecar_bytes: int, uncertainty: float
) -> ProofMachineState:
    if state.phase != "prepared":
        raise CausalProofStateError("hot bound can only be installed once")
    if sidecar_bytes < 0 or not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise CausalProofStateError("invalid hot-bound action")
    charged = state.charged_bytes + sidecar_bytes
    phase = "bounded" if charged <= state.allowance_bytes else "fallback_required"
    return replace(
        state,
        phase=phase,
        charged_bytes=charged,
        remaining_uncertainty=uncertainty,
    )


def refine_proof(
    state: ProofMachineState, *, cold_bytes: int, next_uncertainty: float
) -> ProofMachineState:
    if state.phase not in {"bounded", "refining"}:
        raise CausalProofStateError("proof refinement is not legal in this phase")
    if cold_bytes <= 0:
        raise CausalProofStateError("cold refinement must charge positive bytes")
    if (
        not math.isfinite(next_uncertainty)
        or next_uncertainty < 0.0
        or next_uncertainty > state.remaining_uncertainty
    ):
        raise CausalProofStateError("refinement uncertainty must decrease monotonically")
    if state.queried_blocks >= state.total_blocks:
        raise CausalProofStateError("all cold blocks are already resolved")
    charged = state.charged_bytes + cold_bytes
    phase = "refining" if charged <= state.allowance_bytes else "fallback_required"
    return replace(
        state,
        phase=phase,
        charged_bytes=charged,
        queried_blocks=state.queried_blocks + 1,
        remaining_uncertainty=next_uncertainty,
    )


def certify_proof(state: ProofMachineState) -> ProofMachineState:
    if state.phase not in {"bounded", "refining"}:
        raise CausalProofStateError("certification is not legal in this phase")
    if state.charged_bytes > state.allowance_bytes or not state.certifiable:
        raise CausalProofStateError("winner is not certified inside the budget")
    return replace(state, phase="certified")


def commit_proof(state: ProofMachineState) -> ProofMachineState:
    if state.phase != "certified":
        raise CausalProofStateError("only a certified proof may commit")
    return replace(state, phase="committed")


def exact_fallback(state: ProofMachineState) -> ProofMachineState:
    if state.phase == "committed":
        raise CausalProofStateError("a committed decision cannot fall back")
    return replace(
        state,
        phase="fallback",
        queried_blocks=state.total_blocks,
        remaining_uncertainty=0.0,
        exact_fallback_used=True,
    )


def aggregate_gate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise CausalProofStateError("Gate rows must be nonempty")
    token_count = sum(int(row["token_count"]) for row in rows)
    if token_count <= 0:
        raise CausalProofStateError("Gate token population must be positive")
    kls: list[float] = []
    for row in rows:
        row_kls = [max(0.0, float(value)) for value in row["token_kls"]]
        if len(row_kls) != int(row["token_count"]):
            raise CausalProofStateError("token KL count mismatch")
        if any(not math.isfinite(value) for value in row_kls):
            raise CausalProofStateError("token KL must be finite")
        kls.extend(row_kls)
    relative_errors = [float(row["mlp_relative_l2"]) for row in rows]
    proof_ratios = [float(row["minimum_sound_radius_ratio"]) for row in rows]
    values = (*relative_errors, *proof_ratios)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise CausalProofStateError("relative metrics must be finite and nonnegative")
    matches = sum(int(row["top1_matches"]) for row in rows)
    return {
        "case_count": len(rows),
        "token_count": token_count,
        "top1_matches": matches,
        "top1_agreement": matches / token_count,
        "mean_kl": sum(kls) / len(kls),
        "p95_kl": nearest_rank(kls, 0.95),
        "mlp_relative_l2_p50": nearest_rank(relative_errors, 0.50),
        "mlp_relative_l2_p95": nearest_rank(relative_errors, 0.95),
        "minimum_sound_radius_ratio_p50": nearest_rank(proof_ratios, 0.50),
        "minimum_sound_radius_ratio_p95": nearest_rank(proof_ratios, 0.95),
    }


def family_aggregates(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        family: aggregate_gate_rows(
            [row for row in rows if str(row["family"]) == family]
        )
        for family in sorted({str(row["family"]) for row in rows})
    }


def gate_decision(
    *,
    baseline_mismatches: int,
    budget_fraction: float,
    maximum_budget_fraction: float,
    population: dict[str, Any],
    families: dict[str, dict[str, Any]],
    min_top1_agreement: float,
    min_family_top1_agreement: float,
    max_mean_kl: float,
    max_p95_kl: float,
    max_radius_ratio_p50: float,
    max_radius_ratio_p95: float,
) -> tuple[dict[str, bool], str]:
    controls = baseline_mismatches == 0
    budget = budget_fraction <= maximum_budget_fraction + 1e-12
    quality = (
        float(population["top1_agreement"]) >= min_top1_agreement
        and float(population["mean_kl"]) <= max_mean_kl
        and float(population["p95_kl"]) <= max_p95_kl
    )
    family = bool(families) and all(
        float(row["top1_agreement"]) >= min_family_top1_agreement
        for row in families.values()
    )
    proof = (
        float(population["minimum_sound_radius_ratio_p50"])
        <= max_radius_ratio_p50
        and float(population["minimum_sound_radius_ratio_p95"])
        <= max_radius_ratio_p95
    )
    gates = {
        "controls_passed": controls,
        "budget_passed": budget,
        "population_quality_passed": quality,
        "family_quality_passed": family,
        "local_proof_radius_passed": proof,
    }
    if not controls:
        decision = "INVALID_CAUSAL_PROOF_STATE_CONTROL_FAILURE"
    elif all(gates.values()):
        decision = "PROMOTE_TO_NONLINEAR_CAUSAL_PROOF_PROPAGATION_GATE"
    else:
        decision = "REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH"
    return gates, decision
