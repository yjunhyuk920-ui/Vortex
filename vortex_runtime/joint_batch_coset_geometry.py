"""Exact finite geometry for joint rank-one functional batches.

The columns of the projective system containing all nonzero binary rank-one
``m x n`` masks form the Segre product of two binary simplex systems.  The
generalized-weight formula for chained product codes therefore gives the
largest number of rank-one masks that can lie in a linear subspace.

This module records the exact small formula, a finite registered counting
screen, and the factor-envelope observation for a K-query batch.  It does not
construct the required dictionary and it is not a nonlinear cell-probe lower
bound.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import lru_cache


REGISTERED_PARAMETERS = 405_849_243_648
GLOBAL_ADVICE_BITS = 8 * 8 * (1 << 30)
FAVORABLE_GLOBAL_DICTIONARY_ATOMS = REGISTERED_PARAMETERS + GLOBAL_ADVICE_BITS
REGISTERED_SQUARE_SIDE = 16_384
REGISTERED_BATCH_QUERIES = 32
WORD_BITS = 64

DECISION = (
    "REJECT_LITERAL_FACTOR_ENVELOPE_CATALOG_AND_COUNTING_ONLY_PROMOTION_"
    "KEEP_IMPLICIT_JOINT_FACTOR_ENVELOPE_GENERATOR_OPEN"
)


def binary_simplex_weight(dimension: int, subcode_dimension: int) -> int:
    """Return the generalized Hamming weight of a binary simplex code."""

    if dimension <= 0 or not 0 <= subcode_dimension <= dimension:
        raise ValueError("invalid simplex dimensions")
    if subcode_dimension == 0:
        return 0
    return (1 << dimension) - (1 << (dimension - subcode_dimension))


def product_simplex_generalized_weight(
    left_dimension: int,
    right_dimension: int,
    subcode_dimension: int,
) -> int:
    """Exact chained-product formula, intended for finite control sizes.

    For the two simplex factors, Schaathun's product-code theorem reduces to

        min sum_i 2**(m-i) d_{t_i}(S_n),

    over non-increasing ``n >= t_1 >= ... >= t_m >= 0`` with sum ``r``.
    The exponential state space is deliberately guarded: registered-scale
    arithmetic uses the closed two-strip formula below, not this control DP.
    """

    if left_dimension <= 0 or right_dimension <= 0:
        raise ValueError("factor dimensions must be positive")
    ambient = left_dimension * right_dimension
    if not 0 <= subcode_dimension <= ambient:
        raise ValueError("subcode dimension is outside the product")
    if ambient > 144:
        raise ValueError("exact partition DP is restricted to finite controls")

    @lru_cache(maxsize=None)
    def visit(index: int, maximum_part: int, remaining: int) -> int | None:
        if index == left_dimension:
            return 0 if remaining == 0 else None
        best: int | None = None
        for part in range(min(maximum_part, remaining), -1, -1):
            suffix = visit(index + 1, part, remaining - part)
            if suffix is None:
                continue
            delta_left = 1 << (left_dimension - index - 1)
            score = (
                delta_left * binary_simplex_weight(right_dimension, part)
                + suffix
            )
            if best is None or score < best:
                best = score
        return best

    result = visit(0, right_dimension, subcode_dimension)
    if result is None:
        raise AssertionError("product partition unexpectedly absent")
    return result


def maximum_rank_one_points_control(
    rows: int, columns: int, subspace_dimension: int
) -> int:
    """Exact maximum for finite controls via generalized Hamming weights."""

    ambient = rows * columns
    if not 0 <= subspace_dimension <= ambient:
        raise ValueError("subspace dimension is outside the matrix space")
    points = ((1 << rows) - 1) * ((1 << columns) - 1)
    codimension = ambient - subspace_dimension
    return points - product_simplex_generalized_weight(
        rows, columns, codimension
    )


def square_two_strip_maximum_rank_one_points(
    side: int, subspace_dimension: int
) -> int:
    """Exact Segre intersection for the first two square strips.

    For ``0 <= t <= n``, a fixed left factor times a t-space attains
    ``2**t-1``.  For ``n <= t <= 2*n``, the chained-product formula is
    attained by a full first row plus ``t-n`` coordinates of a second row
    (or its transpose): ``2**n + 2**(t-n+1) - 3``.
    """

    if side <= 0 or not 0 <= subspace_dimension <= 2 * side:
        raise ValueError("two-strip dimensions are invalid")
    if subspace_dimension <= side:
        return (1 << subspace_dimension) - 1
    return (
        (1 << side)
        + (1 << (subspace_dimension - side + 1))
        - 3
    )


def support_choice_log2_upper(items: int, selected: int) -> Decimal:
    """Return ``log2 sum_{j<=t} C(S,j) <= t log2(e*S/t)``."""

    if items <= 0 or selected <= 0 or selected > items // 2:
        raise ValueError("require 0 < selected <= items/2")
    with localcontext() as context:
        context.prec = 80
        return Decimal(selected) * (
            (Decimal(items) / Decimal(selected)).ln() + Decimal(1)
        ) / Decimal(2).ln()


def registered_square_joint_batch_counting_screen() -> dict[str, object]:
    """Give every global atom to one square and apply a proof-safe count.

    A support of at most ``t < 2n`` spans fewer than ``2**(n+1)`` rank-one
    masks.  There are fewer than ``(e*S/t)**t`` such supports.  The ordered
    K-batch universe has more than ``2**(K*(2n-2))`` elements because one
    square has ``(2**n-1)**2 > 2**(2n-2)`` rank-one masks.  This yields a
    finite necessary atom count, but perfect word packing makes it far too
    small to resolve the VORTEX physical budget.
    """

    side = REGISTERED_SQUARE_SIDE
    query_count = REGISTERED_BATCH_QUERIES
    items = FAVORABLE_GLOBAL_DICTIONARY_ATOMS
    query_log2_lower = query_count * (2 * side - 2)
    intersection_log2_upper = query_count * (side + 1)
    support_threshold = Decimal(query_log2_lower - intersection_log2_upper)

    last_ruled_out = 0
    for selected in range(1, 2 * side):
        if support_choice_log2_upper(items, selected) < support_threshold:
            last_ruled_out = selected
        else:
            break
    minimum_atoms = last_ruled_out + 1
    minimum_words = (minimum_atoms + WORD_BITS - 1) // WORD_BITS
    return {
        "square_side": side,
        "ordered_batch_queries": query_count,
        "all_global_dictionary_atoms_granted_to_one_square": items,
        "rank_one_batch_log2_cardinality_strict_lower": query_log2_lower,
        "per_support_batch_log2_coverage_strict_upper": (
            intersection_log2_upper
        ),
        "support_name_log2_threshold": str(support_threshold),
        "largest_bit_atom_support_ruled_out": last_ruled_out,
        "minimum_bit_atoms_not_ruled_out": minimum_atoms,
        "minimum_perfectly_packed_64bit_words_not_ruled_out": minimum_words,
        "minimum_perfectly_packed_bytes_not_ruled_out": minimum_words * 8,
        "is_target_scale_lower_bound": False,
    }


def joint_factor_envelope(
    rows: int, columns: int, query_count: int
) -> dict[str, int | str]:
    """Return the exact universal factor-space envelope of a query batch."""

    if rows <= 0 or columns <= 0 or query_count <= 0:
        raise ValueError("shape and query count must be positive")
    left_dimension = min(rows, query_count)
    right_dimension = min(columns, query_count)
    return {
        "rows": rows,
        "columns": columns,
        "query_count": query_count,
        "left_factor_span_at_most": left_dimension,
        "right_factor_span_at_most": right_dimension,
        "tensor_envelope_dimension_at_most": (
            left_dimension * right_dimension
        ),
        "equation": "span{r_i u_i^T} subseteq R tensor U",
    }


def factor_envelope_catalog_screen(
    side: int = REGISTERED_SQUARE_SIDE,
    query_count: int = REGISTERED_BATCH_QUERIES,
) -> dict[str, object]:
    """Reject literal materialization of every K-factor-subspace pair."""

    if side <= query_count or query_count <= 0:
        raise ValueError("require side > query_count > 0")
    # The binary Gaussian coefficient [n choose K]_2 is strictly greater
    # than 2**(K*(n-K)); choose left and right subspaces independently.
    pair_count_log2_strict_lower = 2 * query_count * (side - query_count)
    envelope_cells = query_count * query_count
    return {
        "side": side,
        "query_count": query_count,
        "factor_subspace_pair_count_log2_strict_lower": (
            pair_count_log2_strict_lower
        ),
        "exact_summaries_per_literal_envelope": envelope_cells,
        "literal_catalog_summary_count_strict_lower": (
            f"{envelope_cells} * 2^{pair_count_log2_strict_lower}"
        ),
        "literal_catalog_fits": False,
        "implicit_shared_generator_rejected": False,
    }


def derive_audit() -> dict[str, object]:
    tiny = {
        f"2x{columns}": [
            maximum_rank_one_points_control(2, columns, dimension)
            for dimension in range(2 * columns + 1)
        ]
        for columns in (2, 3, 4)
    }
    envelope = joint_factor_envelope(
        REGISTERED_SQUARE_SIDE,
        REGISTERED_SQUARE_SIDE,
        REGISTERED_BATCH_QUERIES,
    )
    counting = registered_square_joint_batch_counting_screen()
    catalog = factor_envelope_catalog_screen()
    return {
        "classification": "E0_JOINT_BATCH_COSET_GEOMETRY",
        "contract": {
            "registered_parameters": REGISTERED_PARAMETERS,
            "global_advice_bits": GLOBAL_ADVICE_BITS,
            "favorable_global_dictionary_atoms": (
                FAVORABLE_GLOBAL_DICTIONARY_ATOMS
            ),
            "binary_linear_dictionary_screen_only": True,
        },
        "segre_product_code_theorem": {
            "projective_system": "binary rank-one masks = S_m Segre S_n",
            "maximum_intersection": (
                "|Q|-d_(mn-t)(S_m tensor S_n)"
            ),
            "tiny_exact_sequences": tiny,
            "prototype_commit": "0206a08",
        },
        "joint_factor_envelope": envelope,
        "literal_factor_envelope_catalog": catalog,
        "general_dictionary_counting": counting,
        "claim_boundary": {
            "rank_one_intersection_formula_exact": True,
            "joint_factor_envelope_exact": True,
            "literal_envelope_catalog_rejected": True,
            "general_dictionary_counting_resolves_target": False,
            "implicit_joint_envelope_generator_constructed": False,
            "nonlinear_advice_or_adaptive_probes_covered": False,
            "native_numerical_lift": False,
            "target_candidate": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_SEGRE_WEIGHT_HIERARCHY_AS_EXACT_BATCH_GEOMETRY",
            "REJECT_LITERAL_FACTOR_ENVELOPE_CATALOG",
            "REJECT_COUNTING_ONLY_PROMOTION",
            "KEEP_IMPLICIT_JOINT_FACTOR_ENVELOPE_GENERATOR_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
