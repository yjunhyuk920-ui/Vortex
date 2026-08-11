"""Cheap falsification screens for exact dense-Transformer shortcuts.

The screens in this module deliberately grant each candidate every omitted
cost for free.  They answer only a necessary question: even in that favorable
model, is there enough exact reuse or removable work to approach the
registered 405B/native-4B compute envelope?
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


FULL_405B_GFLOP_PER_TOKEN = 811.698487296
ALLOWED_GFLOP_PER_TOKEN = 9.6
REQUIRED_ELIMINATION_FRACTION = (
    1.0 - ALLOWED_GFLOP_PER_TOKEN / FULL_405B_GFLOP_PER_TOKEN
)
DECISION = "REJECT_SCREENED_SHORTCUTS_KEEP_SEARCHING_NEW_INFORMATION_SOURCE"


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("at least one value is required")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def float32_words(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return array.view(np.uint32)


def temporal_identity_metrics(prefix_inputs: np.ndarray) -> dict[str, object]:
    """Measure exact same-coordinate reuse and natural contiguous runs."""

    words = float32_words(prefix_inputs)
    if words.ndim != 2 or min(words.shape) < 2:
        raise ValueError("prefix_inputs must have at least two rows and columns")

    consecutive = np.mean(words[1:] == words[:-1], axis=1)
    run_fractions = (
        1 + np.sum(words[:, 1:] != words[:, :-1], axis=1)
    ) / words.shape[1]
    distinct_fractions = np.asarray(
        [np.unique(row).size / row.size for row in words],
        dtype=np.float64,
    )

    seen = [set() for _ in range(words.shape[1])]
    historical_hits: list[float] = []
    for row in words:
        hits = 0
        for coordinate, value in enumerate(row):
            integer = int(value)
            hits += integer in seen[coordinate]
            seen[coordinate].add(integer)
        historical_hits.append(hits / row.size)

    return {
        "shape": list(words.shape),
        "consecutive_same_coordinate_fraction": _quantiles(consecutive),
        "consecutive_same_coordinate_overall": float(
            np.mean(words[1:] == words[:-1])
        ),
        "natural_run_fraction": _quantiles(run_fractions),
        "within_vector_distinct_value_fraction": _quantiles(
            distinct_fractions
        ),
        "unlimited_same_coordinate_history_hit_fraction": _quantiles(
            historical_hits
        ),
        "last_history_hit_fraction": float(historical_hits[-1]),
    }


def favorable_rounding_absorption_upper(
    weight: np.ndarray, vector: np.ndarray
) -> dict[str, float]:
    """Return an intentionally loose upper bound on FP32-absorbable products.

    A product receives free credit whenever its magnitude is no larger than
    twice the FP32 ULP of the *entire row L1 sum*.  This threshold is at least
    as favorable as an accumulator-local ULP screen and is not a native-order
    exactness proof.
    """

    matrix = np.asarray(weight, dtype=np.float32)
    current = np.asarray(vector, dtype=np.float32)
    if matrix.ndim != 2 or current.shape != (matrix.shape[1],):
        raise ValueError("weight/vector shape mismatch")
    products = np.asarray(matrix * current[None, :], dtype=np.float32)
    row_l1 = np.sum(np.abs(products), axis=1, dtype=np.float64).astype(
        np.float32
    )
    ulps = np.nextafter(row_l1, np.float32(np.inf)) - row_l1
    fractions = np.mean(
        np.abs(products).astype(np.float64) <= 2.0 * ulps[:, None],
        axis=1,
    )
    return _quantiles(fractions)


def exact_product_reuse_metrics(
    weight: np.ndarray, vector: np.ndarray
) -> dict[str, object]:
    """Measure exact repeated products and perfect opposite-pair cancellation."""

    matrix = np.asarray(weight, dtype=np.float32)
    current = np.asarray(vector, dtype=np.float32)
    if matrix.ndim != 2 or current.shape != (matrix.shape[1],):
        raise ValueError("weight/vector shape mismatch")
    products = np.ascontiguousarray(
        matrix * current[None, :], dtype=np.float32
    )
    words = products.view(np.uint32)

    unique_fractions: list[float] = []
    cancellation_fractions: list[float] = []
    for row in words:
        values, counts = np.unique(row, return_counts=True)
        multiplicities = {
            int(value): int(count) for value, count in zip(values, counts)
        }
        unique_fractions.append(values.size / row.size)
        cancelled = 0
        for value, count in multiplicities.items():
            opposite = value ^ 0x80000000
            if value < opposite and opposite in multiplicities:
                cancelled += 2 * min(count, multiplicities[opposite])
        cancellation_fractions.append(cancelled / row.size)

    unique = _quantiles(unique_fractions)
    cancellation = _quantiles(cancellation_fractions)
    return {
        "unique_product_fraction": unique,
        "best_duplicate_reuse_fraction": 1.0 - unique["min"],
        "perfect_opposite_pair_cancellation_fraction": cancellation,
        "best_combined_local_elimination_upper": min(
            1.0,
            (1.0 - unique["min"]) + cancellation["max"],
        ),
    }


def exact_value_multiplicity_metrics(
    weight_words: np.ndarray,
) -> dict[str, object]:
    """Count exact coefficient symbols along both matrix orientations."""

    words = np.asarray(weight_words)
    if words.ndim != 2 or min(words.shape) < 1:
        raise ValueError("weight_words must be a nonempty matrix")
    unique_per_column = np.asarray(
        [np.unique(words[:, index]).size for index in range(words.shape[1])],
        dtype=np.int64,
    )
    unique_per_row = np.asarray(
        [np.unique(words[index, :]).size for index in range(words.shape[0])],
        dtype=np.int64,
    )
    return {
        "shape": list(words.shape),
        "unique_values_per_column": _quantiles(unique_per_column),
        "unique_values_per_row": _quantiles(unique_per_row),
        "columns_with_at_most_84_values": int(
            np.count_nonzero(unique_per_column <= 84)
        ),
        "rows_with_at_most_84_values": int(
            np.count_nonzero(unique_per_row <= 84)
        ),
        # The Pollard-pseudodimension theorem has dominant ratio T/m for
        # d=1.  Using the minimum observed T grants the theorem its best
        # possible exponent and every logarithm/constant for free.
        "best_possible_d1_forward_work_ratio": float(
            np.min(unique_per_column) / words.shape[0]
        ),
        "best_possible_d1_transpose_work_ratio": float(
            np.min(unique_per_row) / words.shape[1]
        ),
    }


def derive_audit(
    *,
    weight: np.ndarray,
    weight_words: np.ndarray,
    prefix_inputs: np.ndarray,
    current_input: np.ndarray,
) -> dict[str, object]:
    temporal = temporal_identity_metrics(prefix_inputs)
    absorption = favorable_rounding_absorption_upper(weight, current_input)
    products = exact_product_reuse_metrics(weight, current_input)
    values = exact_value_multiplicity_metrics(weight_words)

    observed_elimination_uppers = {
        "rounding_absorption": absorption["max"],
        "consecutive_coordinate_reuse": temporal[
            "consecutive_same_coordinate_fraction"
        ]["max"],
        "natural_run_reuse": 1.0
        - temporal["natural_run_fraction"]["min"],
        "unlimited_coordinate_history_cache": temporal[
            "unlimited_same_coordinate_history_hit_fraction"
        ]["max"],
        "duplicate_products": products["best_duplicate_reuse_fraction"],
        "perfect_opposite_product_pairs": products[
            "perfect_opposite_pair_cancellation_fraction"
        ]["max"],
    }
    decisions = {
        name: (
            "REJECT_BELOW_FAVORABLE_NECESSARY_ELIMINATION"
            if value < REQUIRED_ELIMINATION_FRACTION
            else "INCONCLUSIVE"
        )
        for name, value in observed_elimination_uppers.items()
    }
    best_pollard_ratio = min(
        values["best_possible_d1_forward_work_ratio"],
        values["best_possible_d1_transpose_work_ratio"],
    )
    decisions["pollard_low_complexity_matvec"] = (
        "REJECT_EVEN_IDEAL_D1_WORK_RATIO_EXCEEDS_ALLOWANCE"
        if best_pollard_ratio
        > ALLOWED_GFLOP_PER_TOKEN / FULL_405B_GFLOP_PER_TOKEN
        else "INCONCLUSIVE"
    )

    return {
        "name": "native_exact_shortcut_frontier",
        "decision": DECISION,
        "registered_budget": {
            "full_405b_gflop_per_token": FULL_405B_GFLOP_PER_TOKEN,
            "allowed_gflop_per_token": ALLOWED_GFLOP_PER_TOKEN,
            "allowed_work_fraction": (
                ALLOWED_GFLOP_PER_TOKEN / FULL_405B_GFLOP_PER_TOKEN
            ),
            "required_elimination_fraction": REQUIRED_ELIMINATION_FRACTION,
            "old_2_5_percent_assumption_used": False,
        },
        "favorable_rounding_absorption_upper": absorption,
        "temporal_identity": temporal,
        "exact_product_reuse": products,
        "exact_value_multiplicity": values,
        "candidate_elimination_uppers": observed_elimination_uppers,
        "candidate_decisions": decisions,
        "claim_boundary": {
            "single_real_checkpoint_tensor_observation": True,
            "necessary_condition_only": True,
            "native_accumulation_order_proved": False,
            "whole_model_runtime_built": False,
            "universal_impossibility_proved": False,
            "target_achieved": False,
        },
    }
