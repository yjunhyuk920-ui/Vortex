"""Finite E0 screens for two exact rank-one-query representations.

The gates in this module are deliberately more favorable than a deployable
runtime.  They grant optimal preprocessing, exact real arithmetic, free
address generation, free block reductions, and free verification.  A failure
therefore rejects only the named direct evaluator; it is not a lower bound for
all nonlinear cell-probe data structures.
"""

from __future__ import annotations

from math import ceil, comb, e, floor, log2

import numpy as np


FULL_405B_GFLOP_PER_TOKEN = 811.698487296
ALLOWED_GFLOP_PER_TOKEN = 9.6
ALLOWED_WORK_FRACTION = ALLOWED_GFLOP_PER_TOKEN / FULL_405B_GFLOP_PER_TOKEN
REGISTERED_HIDDEN_SIZE = 16_384
REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES = 883
REGISTERED_GLOBAL_ROW_COORDINATES = 19_997_952
REGISTERED_GLOBAL_COLUMN_COORDINATES = 19_111_936
DECISION = "REJECT_SPIKY_AND_POWER_FOLDS_KEEP_GENERAL_PROBE_GAP_OPEN"


def spiky_component_scalar(
    left: np.ndarray,
    right: np.ndarray,
    row_factor: np.ndarray,
    column_factor: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> float:
    """Evaluate one masked rank-one (spiky) component.

    Positive equal labels select a diagonal rectangle.  Label zero is outside
    the component support.  The equation is

        sum_k <left[I_k], row_factor[I_k]>
              <column_factor[J_k], right[J_k]>.

    The function uses float64 only as a small reference control.  It does not
    claim native BF16/FP32 accumulation-order equivalence.
    """

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    rows = np.asarray(row_factor, dtype=np.float64)
    columns = np.asarray(column_factor, dtype=np.float64)
    row_ids = np.asarray(row_labels, dtype=np.int64)
    column_ids = np.asarray(column_labels, dtype=np.int64)
    if (
        left_array.ndim != 1
        or right_array.ndim != 1
        or rows.shape != left_array.shape
        or columns.shape != right_array.shape
        or row_ids.shape != left_array.shape
        or column_ids.shape != right_array.shape
        or np.any(row_ids < 0)
        or np.any(column_ids < 0)
    ):
        raise ValueError("spiky component arrays must be aligned 1-D vectors")

    answer = 0.0
    labels = np.intersect1d(
        np.unique(row_ids[row_ids > 0]),
        np.unique(column_ids[column_ids > 0]),
        assume_unique=True,
    )
    for label in labels:
        row_mask = row_ids == label
        column_mask = column_ids == label
        answer += float(np.dot(left_array[row_mask], rows[row_mask])) * float(
            np.dot(columns[column_mask], right_array[column_mask])
        )
    return answer


def spiky_direct_description_gate(
    *,
    dimension: int = REGISTERED_HIDDEN_SIZE,
    allowed_work_fraction: float = ALLOWED_WORK_FRACTION,
) -> dict[str, object]:
    """Apply a finite description-count Gate to direct spiky evaluation.

    Let L be the total active row/column factor incidences in a sum of spiky
    matrices.  Direct evaluation of an arbitrary rank-one scalar needs those
    L query/factor interactions even when block combines are free.

    For a fixed total L, all block-mask sequences are bounded by
    ``2 (4 e N)^L``.  For any fixed masks, Warren's theorem bounds the sign
    patterns of the resulting degree-two polynomials by
    ``(8 e N^2 / L)^L``.  Thus at most
    ``2 (32 e^2 N^3 / L)^L`` sign matrices have such a representation.

    The bound grants arbitrary real factors and exact associative arithmetic.
    It is used only when its logarithm is strictly below N^2.
    """

    if dimension < 16:
        raise ValueError("dimension must be at least 16")
    if not 0.0 < allowed_work_fraction < 1.0:
        raise ValueError("allowed_work_fraction must lie strictly in (0, 1)")

    cells = dimension * dimension
    incidence_budget = floor(allowed_work_fraction * cells)
    if incidence_budget < 1 or incidence_budget >= cells:
        raise ValueError("registered incidence budget is outside the proof range")

    representation_base = (
        32.0 * e * e * dimension * dimension * dimension / incidence_budget
    )
    representable_log2_upper = 1.0 + incidence_budget * log2(
        representation_base
    )
    all_sign_matrices_log2 = float(cells)
    hard_exponent = all_sign_matrices_log2 - representable_log2_upper

    paper_component_lower = dimension / (12.0 * log2(dimension))
    paper_component_integer_lower = ceil(paper_component_lower)
    return {
        "dimension": dimension,
        "allowed_work_fraction": allowed_work_fraction,
        "allowed_factor_incidence_terms": incidence_budget,
        "mask_family_log2_bound_formula": "log2(2*(4*e*N)^L)",
        "fixed_mask_sign_pattern_log2_bound_formula": (
            "log2((8*e*N^2/L)^L)"
        ),
        "combined_representable_log2_upper": representable_log2_upper,
        "all_sign_matrices_log2": all_sign_matrices_log2,
        "hard_instance_exponent_bits": hard_exponent,
        "representable_fraction_log2_upper": -hard_exponent,
        "hard_sign_matrix_exists": hard_exponent > 0.0,
        "paper_random_boolean_component_lower_continuous": (
            paper_component_lower
        ),
        "paper_random_boolean_component_lower_integer": (
            paper_component_integer_lower
        ),
        "paper_dense_factor_diagnostic_fraction": (
            2.0 * paper_component_lower / dimension
        ),
        "paper_dense_factor_diagnostic_over_allowance": (
            (2.0 * paper_component_lower / dimension)
            / allowed_work_fraction
        ),
        "favorable_grants": [
            "optimal decomposition",
            "arbitrary real factors",
            "exact associative arithmetic",
            "free preprocessing and selector",
            "free block products and reductions",
            "free metadata, storage, traffic, verification, and fallback",
        ],
        "decision": (
            "REJECT_OMEGA_SPIKECUT_DIRECT_EVALUATOR_BY_FINITE_DESCRIPTION_GATE"
            if hard_exponent > 0.0
            else "INCONCLUSIVE"
        ),
    }


def model_wide_spiky_description_gate(
    *,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    matrix_instances: int = REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
    global_row_coordinates: int = REGISTERED_GLOBAL_ROW_COORDINATES,
    global_column_coordinates: int = REGISTERED_GLOBAL_COLUMN_COORDINATES,
    allowed_gflop_per_token: float = ALLOWED_GFLOP_PER_TOKEN,
) -> dict[str, object]:
    """Count global masked-factor descriptions under the full compute grant.

    Every registered matrix is embedded in one disjoint global row/column
    universe.  The count even permits a single spiky component to couple
    blocks from different matrices and ignores all invalid cross-matrix cells.
    Those grants only enlarge the representation family.

    One active factor/query interaction receives one ideal FMA (two FLOPs),
    so the full 9.6-GFLOP allowance grants 4.8 billion incidences.  The same
    mask/Warren argument then uses P output polynomials rather than N^2.
    """

    if coefficient_count <= 0 or matrix_instances <= 0:
        raise ValueError("coefficient and matrix counts must be positive")
    if global_row_coordinates <= 0 or global_column_coordinates <= 0:
        raise ValueError("global coordinate counts must be positive")
    if allowed_gflop_per_token <= 0.0:
        raise ValueError("allowed_gflop_per_token must be positive")

    factor_incidence_budget = floor(allowed_gflop_per_token * 1e9 / 2.0)
    if factor_incidence_budget >= coefficient_count:
        raise ValueError("factor incidence budget must be sub-dense")
    global_dimension = max(global_row_coordinates, global_column_coordinates)
    representation_base = (
        32.0
        * e
        * e
        * global_dimension
        * coefficient_count
        / factor_incidence_budget
    )
    representable_log2_upper = 1.0 + factor_incidence_budget * log2(
        representation_base
    )
    hard_exponent = coefficient_count - representable_log2_upper
    return {
        "coefficient_count": coefficient_count,
        "matrix_instances": matrix_instances,
        "global_row_coordinates": global_row_coordinates,
        "global_column_coordinates": global_column_coordinates,
        "global_counting_dimension": global_dimension,
        "allowed_factor_incidence_terms": factor_incidence_budget,
        "combined_representable_log2_upper": representable_log2_upper,
        "all_sign_checkpoints_log2": float(coefficient_count),
        "hard_checkpoint_exponent_bits": hard_exponent,
        "representable_fraction_log2_upper": -hard_exponent,
        "hard_sign_checkpoint_exists": hard_exponent > 0.0,
        "global_cross_matrix_components_granted": True,
        "invalid_cross_matrix_cells_ignored": True,
        "decision": (
            "REJECT_OMEGA_SPIKECUT_DIRECT_EVALUATOR_BY_MODEL_WIDE_DESCRIPTION_GATE"
            if hard_exponent > 0.0
            else "INCONCLUSIVE"
        ),
    }


def _maximum_root_rank(*, power: int, expanded_term_cap: int) -> int:
    root_rank = 0
    while comb(root_rank + power, power) <= expanded_term_cap:
        root_rank += 1
    return root_rank


def powerfold_expansion_gate(
    *,
    dimension: int = REGISTERED_HIDDEN_SIZE,
    allowed_work_fraction: float = ALLOWED_WORK_FRACTION,
    powers: tuple[int, ...] = (2, 3, 4, 5),
) -> dict[str, object]:
    """Screen entrywise-power roots by their exact multinomial expansion.

    If ``Z=A B^T`` has rank r and ``W=Z**p`` entrywise for integer p, then W
    has an explicit separable expansion with ``C(r+p-1,p)`` terms.  Direct
    rank-one query evaluation is therefore an ordinary static low-rank
    evaluator after expansion, not a new runtime information source.
    """

    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if not 0.0 < allowed_work_fraction < 1.0:
        raise ValueError("allowed_work_fraction must lie strictly in (0, 1)")
    if not powers or any(power < 2 for power in powers):
        raise ValueError("powers must contain integers at least two")

    expanded_term_cap = floor(allowed_work_fraction * dimension / 2.0)
    thresholds: dict[str, dict[str, int]] = {}
    for power in powers:
        root_rank = _maximum_root_rank(
            power=power,
            expanded_term_cap=expanded_term_cap,
        )
        terms = comb(root_rank + power - 1, power)
        next_terms = comb(root_rank + power, power)
        thresholds[str(power)] = {
            "maximum_root_rank_within_free_term_cap": root_rank,
            "expanded_terms_at_maximum": terms,
            "expanded_terms_at_next_rank": next_terms,
        }

    return {
        "dimension": dimension,
        "allowed_work_fraction": allowed_work_fraction,
        "expanded_separable_term_cap": expanded_term_cap,
        "term_formula": "binom(root_rank + power - 1, power)",
        "power_thresholds": thresholds,
        "exact_runtime_normal_form": "STATIC_LOW_RANK_AFTER_MULTINOMIAL_EXPANSION",
        "paper_epmf_covers_nonnegative_magnitudes_not_signed_native_weights": True,
        "native_accumulation_order_preserved": False,
        "decision": "REJECT_OMEGA_POWERFOLD_AS_RELABELED_STATIC_LOW_RANK_EXECUTION",
    }


def derive_audit() -> dict[str, object]:
    spiky_square = spiky_direct_description_gate()
    spiky_model = model_wide_spiky_description_gate()
    power = powerfold_expansion_gate()
    return {
        "name": "spiky_power_rank_one_frontier",
        "decision": DECISION,
        "registered_budget": {
            "full_405b_gflop_per_token": FULL_405B_GFLOP_PER_TOKEN,
            "allowed_gflop_per_token": ALLOWED_GFLOP_PER_TOKEN,
            "allowed_work_fraction": ALLOWED_WORK_FRACTION,
            "old_2_5_percent_assumption_used": False,
        },
        "omega_spikecut": {
            "model_wide_authoritative_gate": spiky_model,
            "single_square_diagnostic": spiky_square,
            "decision": spiky_model["decision"],
        },
        "omega_powerfold": power,
        "claim_boundary": {
            "finite_e0_only": True,
            "native_numerical_execution_proved": False,
            "all_spiky_or_entrywise_nonlinear_algorithms_rejected": False,
            "general_nonlinear_probe_gap_resolved": False,
            "whole_model_runtime_built": False,
            "target_achieved": False,
        },
    }
