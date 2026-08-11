"""Finite audit of globally nonlocal nonlinear rank-one routes.

The universal VORTEX claim needs an exact bounded-word data structure for
every checkpoint, not merely a Boolean decision structure or an asymptotic
lower bound whose premises fail for rank-one queries.  This module records
three durable boundaries using integer arithmetic only:

* the Larsen--Williams all-zero-rectangle construction;
* the limited-independence premise in Korten--Pitassi--Impagliazzo (2025);
* the information required by an exact summary of every subrectangle.

None of these boundaries is promoted into a general cell-probe impossibility
claim.  They close specific proposed lifts and leave the unrestricted
globally nonlinear numerical rank-one problem open.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt


REGISTERED_SQUARE_DIMENSION = 16_384
NATIVE_WORD_BITS = 64
REQUESTED_FRACTION = Fraction(1, 40)
GLOBAL_ADVICE_BYTES = 8 * (1 << 30)

DECISION = (
    "REJECT_BOOLEAN_ZERO_RECTANGLE_AND_LIMITED_INDEPENDENCE_LIFTS_"
    "KEEP_GENERAL_NONLINEAR_NUMERICAL_RANK_ONE_GAP_OPEN"
)


def _perfect_square_root(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    root = isqrt(value)
    if root * root != value:
        raise ValueError(f"{name} must be a perfect square")
    return root


def larsen_williams_leading_terms(
    dimension: int = REGISTERED_SQUARE_DIMENSION,
    word_bits: int = NATIVE_WORD_BITS,
) -> dict[str, object]:
    """Substitute finite values into Theorem 2.3's leading monomials.

    The paper states big-O bounds, so these values deliberately omit unknown
    constants and poly-machine effects.  They are favorable screens, not
    concrete upper bounds for the registered hardware.
    """

    dimension_root = _perfect_square_root(dimension, name="dimension")
    word_root = _perfect_square_root(word_bits, name="word_bits")
    if word_bits % 8:
        raise ValueError("word_bits must be byte aligned")

    matrix_bits = dimension * dimension
    matrix_bytes = matrix_bits // 8
    n_to_three_halves = dimension * dimension_root
    leading_probe_words = n_to_three_halves // word_root
    leading_probe_bytes = leading_probe_words * (word_bits // 8)
    leading_redundancy_bits = n_to_three_halves * word_root
    leading_redundancy_bytes = leading_redundancy_bits // 8

    return {
        "dimension": dimension,
        "word_bits": word_bits,
        "matrix_bits": matrix_bits,
        "matrix_bytes": matrix_bytes,
        "leading_probe_words": leading_probe_words,
        "leading_probe_bytes": leading_probe_bytes,
        "leading_redundancy_bits": leading_redundancy_bits,
        "leading_redundancy_bytes": leading_redundancy_bytes,
        "leading_probe_byte_fraction_of_one_bit_matrix": str(
            Fraction(leading_probe_bytes, matrix_bytes)
        ),
        "leading_probe_byte_percent_of_one_bit_matrix": (
            100 * leading_probe_bytes / matrix_bytes
        ),
        "leading_redundancy_fraction_of_one_bit_matrix": str(
            Fraction(leading_redundancy_bits, matrix_bits)
        ),
        "leading_redundancy_percent_of_one_bit_matrix": (
            100 * leading_redundancy_bits / matrix_bits
        ),
        "published_bounds_have_hidden_constants": True,
        "published_output": "BOOLEAN_SEMIRING_RECTANGLE_NONEMPTINESS_BIT",
        "exact_parity_count_signed_sum_or_native_numeric_output": False,
        "multiquery_32_token_amortization_supplied": False,
        "target_certificate": False,
    }


def rank_one_three_query_dependency(
    *, word_bits: int = NATIVE_WORD_BITS, minimum_even_probes: int = 2
) -> dict[str, object]:
    """Show why KPI25 limited independence cannot cover full rank-one queries.

    Over GF(2), fix nonzero r and choose distinct nonzero u and v with
    u+v nonzero.  The three distinct rank-one queries r*u^T, r*v^T, and
    r*(u+v)^T XOR to zero.  Their answers therefore XOR to zero for every
    database W and under every distribution over W.  No distribution
    supported on the columns of this query problem can be 3-wise independent.
    """

    if word_bits <= 0:
        raise ValueError("word_bits must be positive")
    if minimum_even_probes <= 0 or minimum_even_probes % 2:
        raise ValueError("minimum_even_probes must be a positive even integer")

    maximum_possible_k = 2
    theorem_strict_threshold = minimum_even_probes * word_bits + 1
    minimum_integer_k_required = theorem_strict_threshold + 1
    return {
        "field": "GF(2)",
        "requires_query_dimension_at_least": 2,
        "three_distinct_queries": [
            "r tensor u",
            "r tensor v",
            "r tensor (u+v)",
        ],
        "query_identity": "q1 XOR q2 XOR q3 = 0",
        "answer_identity_for_every_database": "a1 XOR a2 XOR a3 = 0",
        "support_size_of_answer_triple_at_most": 4,
        "support_size_required_for_three_wise_independence": 8,
        "maximum_k_wise_independence_of_full_distinct_query_family": (
            maximum_possible_k
        ),
        "uniform_database_achieves_pairwise_independence": True,
        "kpi_theorem_requires_even_t": True,
        "substituted_even_probe_count": minimum_even_probes,
        "substituted_word_bits": word_bits,
        "kpi_strict_premise": "k > t*w + 1",
        "strict_threshold_t_times_w_plus_one": theorem_strict_threshold,
        "minimum_integer_k_required": minimum_integer_k_required,
        "premise_can_hold_for_full_rank_one_family": (
            maximum_possible_k > theorem_strict_threshold
        ),
        "target_lower_bound_obtained": False,
    }


def exact_subrectangle_summary_lower_bound(
    *, rows: int, columns: int, alphabet_size: int
) -> dict[str, object]:
    """Count bits needed by a no-raw-probe exact subrectangle summary.

    If one summary must answer the exact aggregate of every subrectangle and
    the singleton aggregate returns the cell value, singleton queries recover
    every cell.  The summary map is injective, hence it has at least as many
    states as the raw block.  This is a scoped lemma about fully covered
    regions; it is not a lower bound for arbitrary adaptive global schemes.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if alphabet_size < 2 or alphabet_size & (alphabet_size - 1):
        raise ValueError("alphabet_size must be a power of two")

    cells = rows * columns
    symbol_bits = alphabet_size.bit_length() - 1
    information_bits = cells * symbol_bits
    return {
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "alphabet_size": alphabet_size,
        "bits_per_symbol": symbol_bits,
        "proof": [
            "query every singleton rectangle",
            "each singleton answer recovers its cell",
            "equal summaries imply equal cells at every position",
            "therefore the summary map is injective",
        ],
        "minimum_summary_bits": information_bits,
        "raw_information_bits": information_bits,
        "compression_below_raw_information_possible": False,
        "scope": (
            "SUMMARY_ALONE_ANSWERS_EVERY_SUBRECTANGLE_WITHOUT_RAW_PROBES"
        ),
        "covers_arbitrary_adaptive_global_data_structures": False,
    }


def scalarization_of_static_matvec_lower_bound(
    *, dimension: int, whole_vector_probe_lower_bound: int
) -> dict[str, object]:
    """Record the strongest scalar consequence of a whole-MatVec bound.

    A rank-one scalar oracle can recover all n coordinates of W*u with n
    calls using left vectors e_i.  Therefore a T-probe lower bound for the
    whole vector implies only ceil(T/n) probes for at least one scalar call.
    This direction is valid but loses a factor n.
    """

    if dimension <= 0 or whole_vector_probe_lower_bound <= 0:
        raise ValueError("dimension and lower bound must be positive")
    scalar_probe_lower_bound = (
        whole_vector_probe_lower_bound + dimension - 1
    ) // dimension
    return {
        "dimension": dimension,
        "whole_vector_probe_lower_bound": whole_vector_probe_lower_bound,
        "number_of_basis_left_queries": dimension,
        "implied_scalar_probe_lower_bound": scalar_probe_lower_bound,
        "loss_factor": dimension,
        "reaches_one_fortieth_of_matrix_words": (
            scalar_probe_lower_bound * 40 >= dimension * dimension
        ),
    }


def derive_audit() -> dict[str, object]:
    larsen_williams = larsen_williams_leading_terms()
    limited_independence = rank_one_three_query_dependency()
    exact_summary = exact_subrectangle_summary_lower_bound(
        rows=REGISTERED_SQUARE_DIMENSION,
        columns=REGISTERED_SQUARE_DIMENSION,
        alphabet_size=2,
    )
    # Deliberately grant an ideal full-scan n^2-word MatVec lower bound.  Even
    # this over-favorable premise scalarizes to n words, far below n^2/40.
    ideal_scalarization = scalarization_of_static_matvec_lower_bound(
        dimension=REGISTERED_SQUARE_DIMENSION,
        whole_vector_probe_lower_bound=(
            REGISTERED_SQUARE_DIMENSION * REGISTERED_SQUARE_DIMENSION
        ),
    )

    return {
        "classification": "E0_GLOBAL_NONLINEAR_RANK_ONE_FRONTIER",
        "decision": DECISION,
        "registered_target": {
            "requested_fraction": str(REQUESTED_FRACTION),
            "registered_square_dimension": REGISTERED_SQUARE_DIMENSION,
            "native_word_bits": NATIVE_WORD_BITS,
            "global_advice_bytes": GLOBAL_ADVICE_BYTES,
            "required_output": (
                "REFERENCE_EXACT_NATIVE_NUMERICAL_RANK_ONE_SCALAR"
            ),
        },
        "larsen_williams_2017": larsen_williams,
        "korten_pitassi_impagliazzo_2025": limited_independence,
        "exact_aggregate_summary_lemma": exact_summary,
        "cgl_static_matvec_scalarization_boundary": ideal_scalarization,
        "dynamic_2026_boundary": {
            "problem": "DYNAMIC_MULTIPHASE_INNER_PRODUCT_OVER_GF2",
            "published_lower_bound": (
                "Omega((log n / log log n)^2) UPDATE_OR_TOTAL_PROBES"
            ),
            "contains_updates": True,
            "static_checkpoint_query_model": False,
            "target_scale_certificate": False,
        },
        "claim_boundary": {
            "zero_rectangle_numeric_lift": "REJECTED",
            "kpi_limited_independence_application": "PREMISE FALSE",
            "whole_matvec_lower_bound_as_scalar_target_bound": (
                "TOO_WEAK_AFTER_N_FOLD_SCALARIZATION"
            ),
            "general_nonlinear_numerical_rank_one_constructor": "NOT FOUND",
            "general_nonlinear_numerical_rank_one_lower_bound": "NOT PROVED",
            "universal_2_5_percent_guarantee": "NOT ESTABLISHED",
            "universal_2_5_percent_impossibility": "NOT PROVED",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "primary_sources": {
            "larsen_williams": "https://arxiv.org/abs/1605.01695",
            "korten_pitassi_impagliazzo": (
                "https://eccc.weizmann.ac.il/report/2025/030/"
            ),
            "clifford_gronlund_larsen": (
                "https://arxiv.org/abs/1504.01836"
            ),
            "dynamic_multiphase_2026": (
                "https://eccc.weizmann.ac.il/report/2026/047/"
            ),
        },
    }
