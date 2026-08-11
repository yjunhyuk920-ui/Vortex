"""Finite E0 audit for a cold sparse functional dictionary.

The candidate is deliberately reduced to the most favorable exact binary
interface.  A compiler stores linear forms ``c_j = <g_j, W>`` of an arbitrary
binary checkpoint.  A query is answered only when its coefficient mask can be
written as the XOR of a small number of dictionary atoms ``g_j``.

This module does *not* construct a dictionary aligned with the rank-one query
set.  It records two useful boundaries:

* literal local/all-query tables have fatal persistent-space expansion; and
* query cardinality alone is far too weak to reject a near-linear global
  non-systematic dictionary.

All arithmetic is deterministic.  No model or hardware action occurs.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction


REGISTERED_PARAMETERS = 405_849_243_648
REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
GLOBAL_ADVICE_BITS = 8 * 8 * (1 << 30)
DICTIONARY_FORM_BITS = REGISTERED_PARAMETERS + GLOBAL_ADVICE_BITS
WORD_BITS = 64
REGISTERED_P50_FRACTION = Fraction(8, 675)

LAYERS = 126
HIDDEN = 16_384
KV = 1_024
INTERMEDIATE = 53_248
VOCAB = 128_256

DECISION = (
    "REJECT_LITERAL_TABLE_AND_NAME_ONLY_LDC_PROMOTION_"
    "KEEP_ALIGNED_SPARSE_FUNCTIONAL_DICTIONARY_UNCONSTRUCTED"
)


def _log2(value: Decimal) -> Decimal:
    return value.ln() / Decimal(2).ln()


def registered_matrix_shapes() -> tuple[tuple[int, int, int], ...]:
    """Return ``(rows, columns, multiplicity)`` for the frozen dense model."""

    return (
        (HIDDEN, HIDDEN, 2 * LAYERS),
        (KV, HIDDEN, 2 * LAYERS),
        (INTERMEDIATE, HIDDEN, 2 * LAYERS),
        (HIDDEN, INTERMEDIATE, LAYERS),
        (VOCAB, HIDDEN, 1),
        (HIDDEN, VOCAB, 1),
    )


def rank_one_tuple_cardinality_boundary() -> dict[str, int | str]:
    """Bound the number of independently selectable binary rank-one tuples.

    A binary ``m x n`` block has exactly

        1 + (2**m - 1) * (2**n - 1)

    distinct rank-at-most-one masks.  The product over disjoint blocks is
    strictly below ``2**L`` where ``L=sum(m+n)``.  Every dimension is at least
    1,024, so the product of the normalized factors is greater than one half:
    ``prod(1-x_i) >= 1-sum(x_i) > 1/2``.  Hence its base-two logarithm has the
    exact integer floor ``L-1`` without materializing a 39-million-bit integer.

    This independently selectable tuple is a stronger audit interface, not a
    theorem that ordinary causal Transformer traces realize every tuple.
    """

    shapes = registered_matrix_shapes()
    matrix_count = sum(multiplicity for _, _, multiplicity in shapes)
    description_bits = sum(
        (rows + columns) * multiplicity
        for rows, columns, multiplicity in shapes
    )
    parameters = sum(
        rows * columns * multiplicity
        for rows, columns, multiplicity in shapes
    )
    minimum_side = min(min(rows, columns) for rows, columns, _ in shapes)

    if parameters != REGISTERED_PARAMETERS:
        raise AssertionError("registered shape population drifted")
    # Each normalized rank-one count is at least
    # 1 - 2**(-rows) - 2**(-columns).  The total deficit is far below 1/2.
    if matrix_count >= (1 << (minimum_side - 2)):
        raise AssertionError("finite product lower witness no longer holds")

    return {
        "matrix_count": matrix_count,
        "minimum_matrix_side": minimum_side,
        "parameters_reconstructed_from_shapes": parameters,
        "pair_description_bits_upper_bound": description_bits,
        "strict_query_count_lower_exponent": description_bits - 1,
        "exact_floor_log2_query_count": description_bits - 1,
        "literal_one_bit_table_address_bits": description_bits,
        "trace_reachability": "NOT_ESTABLISHED",
    }


def sparse_choice_log2_upper(
    *, items: int, probes: int, choice_bits_per_item: int = 0
) -> Decimal:
    """Upper-bound choices made with at most ``probes`` selected items.

    For ``1 <= t <= S/2``,

        sum(j<=t, C(S,j)) <= (e*S/t)**t < (3*S/t)**t.

    If each selected word permits at most ``2**b`` internal linear choices,
    multiply by ``2**(b*t)``.  The result is a favorable *upper* bound; once
    it exceeds the query count, cardinality counting simply stops deciding.
    """

    if items <= 0 or choice_bits_per_item < 0:
        raise ValueError("items must be positive and choice bits non-negative")
    if probes == 0:
        return Decimal(0)
    if probes < 0 or probes > items // 2:
        raise ValueError("require 0 <= probes <= items/2")
    with localcontext() as context:
        context.prec = 80
        ratio = Decimal(3 * items) / Decimal(probes)
        return Decimal(probes) * (
            Decimal(choice_bits_per_item) + _log2(ratio)
        )


def largest_cardinality_ruled_out_probe_count(
    *, items: int, strict_query_lower_exponent: int,
    choice_bits_per_item: int = 0,
) -> int:
    """Largest ``t`` whose choice upper bound cannot cover the query set."""

    if strict_query_lower_exponent <= 0:
        raise ValueError("query exponent must be positive")
    low, high = 0, items // 2
    threshold = Decimal(strict_query_lower_exponent)
    while low < high:
        middle = (low + high + 1) // 2
        if sparse_choice_log2_upper(
            items=items,
            probes=middle,
            choice_bits_per_item=choice_bits_per_item,
        ) <= threshold:
            low = middle
        else:
            high = middle - 1
    return low


def sparse_subset_log2_lower(*, items: int, subset_size: int) -> Decimal:
    """Elementary lower bound ``log2 C(items,t) >= t log2(items/t)``."""

    if items <= 0 or subset_size <= 0 or subset_size > items:
        raise ValueError("invalid subset parameters")
    with localcontext() as context:
        context.prec = 80
        return Decimal(subset_size) * _log2(
            Decimal(items) / Decimal(subset_size)
        )


def minimum_capacity_witness_probe_count(
    *, items: int, query_log2_upper_bound: int,
    choice_bits_per_item_lower: int = 0,
) -> int:
    """First ``t`` whose raw combination count reaches all query names.

    This is only a capacity witness.  It says there are enough possible sparse
    sums in principle; it does not align any of them with a rank-one query.
    """

    if query_log2_upper_bound <= 0 or choice_bits_per_item_lower < 0:
        raise ValueError("invalid capacity-witness inputs")
    low, high = 1, items // 2
    threshold = Decimal(query_log2_upper_bound)
    while low < high:
        middle = (low + high) // 2
        capacity = sparse_subset_log2_lower(
            items=items, subset_size=middle
        ) + Decimal(choice_bits_per_item_lower * middle)
        if capacity >= threshold:
            high = middle
        else:
            low = middle + 1
    return low


def high_girth_union_log2_upper(
    *, columns: int, maximum_dependency_weight: int
) -> Decimal:
    """Union-bound exponent for a random binary dictionary dependency."""

    if maximum_dependency_weight <= 0:
        raise ValueError("dependency weight must be positive")
    return sparse_choice_log2_upper(
        items=columns,
        probes=maximum_dependency_weight,
    )


def direct_square_table_case(block_side: int) -> dict[str, object]:
    """Finite storage/query cost of one literal rank-one table per tile."""

    if block_side <= 0:
        raise ValueError("block side must be positive")
    raw_cells = block_side * block_side
    stored_nonzero_rank_one_forms = ((1 << block_side) - 1) ** 2
    storage_ratio = Fraction(stored_nonzero_rank_one_forms, raw_cells)
    query_fraction = Fraction(1, raw_cells)
    persistent_bits = (
        storage_ratio * REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    return {
        "block_side": block_side,
        "raw_cells_per_tile": raw_cells,
        "stored_nonzero_rank_one_forms_per_tile": (
            stored_nonzero_rank_one_forms
        ),
        "storage_ratio_over_one_bit_source": str(storage_ratio),
        "storage_ratio_decimal": float(storage_ratio),
        "worst_query_fraction": str(query_fraction),
        "worst_query_fraction_decimal": float(query_fraction),
        "favorable_persistent_bits": str(persistent_bits),
        "favorable_persistent_tib": float(
            persistent_bits / (8 * (1 << 40))
        ),
    }


def _dictionary_counting_case(
    *, items: int, choice_bits_upper: int,
    choice_bits_lower: int, dependency_multiplier: int,
    query_floor: int, query_upper: int,
) -> dict[str, object]:
    ruled_out = largest_cardinality_ruled_out_probe_count(
        items=items,
        strict_query_lower_exponent=query_floor,
        choice_bits_per_item=choice_bits_upper,
    )
    first_not_ruled_out = ruled_out + 1
    capacity = minimum_capacity_witness_probe_count(
        items=items,
        query_log2_upper_bound=query_upper,
        choice_bits_per_item_lower=choice_bits_lower,
    )
    dependency_weight = dependency_multiplier * capacity
    dependency_union = high_girth_union_log2_upper(
        columns=DICTIONARY_FORM_BITS,
        maximum_dependency_weight=dependency_weight,
    )
    if dependency_union >= Decimal(REGISTERED_PARAMETERS):
        raise AssertionError("random high-girth existence witness failed")
    # For a random D x S binary matrix, row-rank deficiency is at most
    # (2^D-1)2^-S < 2^(D-S), by union-bounding nonzero left-kernel vectors.
    # The small-dependency probability is at most
    # 2^(dependency_union-D).  Both exponents are negative by enormous
    # margins, and their sum is below one, so a full-row-rank dictionary with
    # the stated girth exists.  This is still not an aligned dictionary.
    rank_deficiency_probability_log2_upper = (
        REGISTERED_PARAMETERS - DICTIONARY_FORM_BITS
    )
    small_dependency_probability_log2_upper = (
        dependency_union - Decimal(REGISTERED_PARAMETERS)
    )
    if rank_deficiency_probability_log2_upper >= 0:
        raise AssertionError("full-rank random witness lost redundancy")
    if small_dependency_probability_log2_upper >= -1:
        raise AssertionError("combined random existence margin is too small")
    return {
        "items": items,
        "choice_bits_upper_per_selected_item": choice_bits_upper,
        "choice_bits_lower_per_selected_item": choice_bits_lower,
        "largest_probe_count_ruled_out_by_cardinality": ruled_out,
        "minimum_probe_count_not_ruled_out_by_cardinality": (
            first_not_ruled_out
        ),
        "log2_choice_upper_at_ruled_out": str(
            sparse_choice_log2_upper(
                items=items,
                probes=ruled_out,
                choice_bits_per_item=choice_bits_upper,
            )
        ),
        "log2_choice_upper_at_first_not_ruled_out": str(
            sparse_choice_log2_upper(
                items=items,
                probes=first_not_ruled_out,
                choice_bits_per_item=choice_bits_upper,
            )
        ),
        "capacity_only_witness_probe_count": capacity,
        "capacity_only_log2_lower": str(
            sparse_subset_log2_lower(
                items=items, subset_size=capacity
            ) + Decimal(choice_bits_lower * capacity)
        ),
        "maximum_dependency_weight_for_unique_capacity_sums": (
            dependency_weight
        ),
        "random_dictionary_bad_dependency_union_log2_upper": str(
            dependency_union
        ),
        "random_dictionary_bad_dependency_union_below_data_bits": True,
        "random_dictionary_small_dependency_probability_log2_upper": str(
            small_dependency_probability_log2_upper
        ),
        "random_dictionary_rank_deficiency_probability_log2_upper": (
            rank_deficiency_probability_log2_upper
        ),
        "full_rank_high_girth_dictionary_exists_by_union_bound": True,
        "capacity_is_aligned_to_rank_one_queries": False,
    }


def derive_audit() -> dict[str, object]:
    query = rank_one_tuple_cardinality_boundary()
    query_floor = int(query["strict_query_count_lower_exponent"])
    query_upper = int(query["pair_description_bits_upper_bound"])
    dictionary_words = DICTIONARY_FORM_BITS // WORD_BITS
    if dictionary_words * WORD_BITS != DICTIONARY_FORM_BITS:
        raise AssertionError("dictionary bit count is not word aligned")

    bit_case = _dictionary_counting_case(
        items=DICTIONARY_FORM_BITS,
        choice_bits_upper=0,
        choice_bits_lower=0,
        dependency_multiplier=2,
        query_floor=query_floor,
        query_upper=query_upper,
    )
    # A selected 64-bit word permits at most 2^64 internal subsets.  For the
    # capacity lower witness we count only 2^63 choices per word.  Two answers
    # using t words can differ in at most 128*t underlying form bits.
    word_case = _dictionary_counting_case(
        items=dictionary_words,
        choice_bits_upper=WORD_BITS,
        choice_bits_lower=WORD_BITS - 1,
        dependency_multiplier=2 * WORD_BITS,
        query_floor=query_floor,
        query_upper=query_upper,
    )
    word_case["minimum_not_ruled_out_payload_bytes"] = (
        int(word_case["minimum_probe_count_not_ruled_out_by_cardinality"])
        * (WORD_BITS // 8)
    )
    word_case["capacity_only_payload_bytes"] = (
        int(word_case["capacity_only_witness_probe_count"])
        * (WORD_BITS // 8)
    )

    minimum_target_block_side = 1
    while Fraction(1, minimum_target_block_side**2) > (
        REGISTERED_P50_FRACTION
    ):
        minimum_target_block_side += 1
    target_table = direct_square_table_case(minimum_target_block_side)
    two_by_two = direct_square_table_case(2)

    target_coefficients = (
        REGISTERED_P50_FRACTION
        * REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    return {
        "classification": "E0_SPARSE_FUNCTIONAL_DICTIONARY_FRONTIER",
        "contract": {
            "registered_parameters": REGISTERED_PARAMETERS,
            "registered_non_embedding_coefficients": (
                REGISTERED_NON_EMBEDDING_COEFFICIENTS
            ),
            "global_advice_bits": GLOBAL_ADVICE_BITS,
            "favorable_dictionary_form_bits": DICTIONARY_FORM_BITS,
            "favorable_dictionary_words_64": dictionary_words,
            "registered_p50_fraction": str(REGISTERED_P50_FRACTION),
            "registered_p50_coefficient_budget": str(target_coefficients),
            "binary_screen_only": True,
        },
        "candidate_equation": {
            "stored_cells": "c_j = <g_j, W> over GF(2)",
            "query_decomposition": "q = XOR(j in T(q), g_j)",
            "exact_answer": "<q,W> = XOR(j in T(q), c_j)",
            "cold_source_may_be_reencoded_non_systematically": True,
        },
        "independent_rank_one_tuple": query,
        "literal_table": {
            "one_answer_bit_per_tuple_exceeds_bits": (
                f"2^{query_floor}"
            ),
            "address_bits": query_upper,
            "decision": "REJECT",
        },
        "direct_square_tables": {
            "smallest_nontrivial_uniform_tile": two_by_two,
            "minimum_side_meeting_coefficient_fraction": (
                minimum_target_block_side
            ),
            "target_fraction_tile": target_table,
            "decision": "REJECT_PERSISTENT_SPACE_EXPANSION",
        },
        "near_linear_sparse_dictionary_counting": {
            "bit_forms": bit_case,
            "word_64_favorable": word_case,
            "interpretation": (
                "CARDINALITY_HAS_ROOM_BUT_SUPPLIES_NO_ALIGNED_DICTIONARY"
            ),
        },
        "claim_boundary": {
            "full_literal_answer_table": "REJECTED",
            "direct_local_rank_one_truth_tables": "REJECTED",
            "standard_coordinate_ldc_is_functional_decoder": False,
            "explicit_rank_one_aligned_dictionary": False,
            "sparse_query_decoder": False,
            "implicit_physical_address_layout": False,
            "native_numerical_lift": False,
            "native_accumulation_order": False,
            "complete_405b_cost_equation": False,
            "target_candidate": False,
            "general_nonlinear_lower_bound": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_FULL_BILINEAR_EVALUATION_TABLE",
            "REJECT_DIRECT_LOCAL_RANK_ONE_TRUTH_TABLE",
            "REJECT_STANDARD_COORDINATE_LDC_NAME_AS_CONSTRUCTION",
            "REJECT_QUERY_CARDINALITY_AS_EITHER_CONSTRUCTOR_OR_TARGET_LOWER_BOUND",
            "KEEP_SPARSE_FUNCTIONAL_DICTIONARY_AS_UNCONSTRUCTED_OPEN_INTERFACE",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
