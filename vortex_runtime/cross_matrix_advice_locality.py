"""Finite E0 bounds for globally shared linear checkpoint advice.

The declared model is the systematic linear coefficient-use model over
``F_2``.  A checkpoint vector ``w`` is stored verbatim in cold state and a
single hot advice string stores linear forms spanning a subspace ``U``.  A
query coefficient vector ``q`` is answered by choosing ``u in U`` and using
the raw residual ``q + u``.  Consequently every non-zero component of ``u``
outside the query's matrix block must be cancelled by a charged raw
coefficient use.

This module proves a localization theorem and a conservative global
span-to-cover lower bound.  It does *not* prove a target-sized direct sum for
rank-one bilinear queries, and it is not a lower bound for nonlinear or
data-dependent cell-probe structures.
"""

from __future__ import annotations

from fractions import Fraction

from vortex_runtime.bilinear_cross_residual_frontier import (
    HOT_SIDE_BITS,
    P50_TARGET_FRACTION,
    binary_entropy,
)
from vortex_runtime.lower_bound_audit import TensorSpec, llama_405b_tensor_plan
from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)


# A deliberately conservative rational covering-radius witness.  The full
# hot-advice rate plus H_2(13/50) stays below one with a margin above 0.003.
COVER_RADIUS_FRACTION_WITNESS = Fraction(13, 50)
IDEAL_WORD_BITS = 64
IDEAL_CACHE_LINE_BITS = 512


def registered_non_embedding_specs() -> tuple[TensorSpec, ...]:
    """Return the frozen 883 non-embedding matrix specifications."""

    return tuple(
        spec for spec in llama_405b_tensor_plan() if spec.name != "embedding"
    )


def maximum_rank_one_span_length() -> int:
    """Maximum binary rank needed by a registered matrix block.

    Every ``m x n`` binary matrix is a sum of at most ``min(m, n)`` rank-one
    matrices.  Blockwise decompositions can be aligned, so an arbitrary vector
    over all registered blocks is a sum of at most the maximum such value of
    fixed-block rank-one tuples.
    """

    return max(
        min(spec.rows, spec.columns)
        for spec in registered_non_embedding_specs()
    )


def shortened_local_dimension_upper_bound(
    local_shortened_dimension: int,
    outside_support_size: int,
) -> int:
    """Dimension available after allowing a fixed outside cancellation set.

    If ``U_i = U intersect V_i`` and ``T`` is a fixed outside coordinate set,
    projection into block ``i`` of ``U intersect (V_i + E_T)`` has dimension
    at most ``dim(U_i) + |T|``.  This function exposes that exact finite bound
    for calculators and boundary tests.
    """

    if local_shortened_dimension < 0 or outside_support_size < 0:
        raise ValueError("dimensions and support sizes must be non-negative")
    return local_shortened_dimension + outside_support_size


def outside_cancellation_weight(
    query_word: int,
    advice_word: int,
    local_block_mask: int,
    ambient_width: int,
) -> int:
    """Count raw residual bits forced outside a block-local query.

    ``query_word`` must be supported in ``local_block_mask``.  In the identity
    ``q = u + e``, the outside part of ``e`` is exactly the outside part of
    ``u`` because ``q`` is zero there.
    """

    if ambient_width < 0:
        raise ValueError("ambient width must be non-negative")
    ambient_mask = (1 << ambient_width) - 1 if ambient_width else 0
    if query_word < 0 or advice_word < 0 or local_block_mask < 0:
        raise ValueError("bit words and masks must be non-negative")
    if (query_word | advice_word | local_block_mask) & ~ambient_mask:
        raise ValueError("word or mask exceeds the ambient width")
    if query_word & ~local_block_mask:
        raise ValueError("query is not supported in the declared local block")
    residual = query_word ^ advice_word
    return (residual & (ambient_mask ^ local_block_mask)).bit_count()


def covering_entropy_witness_holds(
    hot_bits: int = HOT_SIDE_BITS,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    radius_fraction: Fraction = COVER_RADIUS_FRACTION_WITNESS,
) -> bool:
    """Check ``hot_rate + H_2(radius_fraction) < 1``."""

    if hot_bits < 0 or coefficient_count <= 0:
        raise ValueError("invalid hot state or coefficient count")
    if not 0 <= radius_fraction <= Fraction(1, 2):
        raise ValueError("radius fraction must be in [0, 1/2]")
    return (
        hot_bits / coefficient_count + binary_entropy(radius_fraction) < 1.0
    )


def certified_covering_radius_cells(
    hot_bits: int = HOT_SIDE_BITS,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    radius_fraction: Fraction = COVER_RADIUS_FRACTION_WITNESS,
) -> int:
    """Finite lower bound on the covering radius of any advice subspace.

    If an ``s``-dimensional binary subspace covered the ambient cube within
    radius ``delta*D``, sphere covering and the entropy volume bound would give
    ``s/D + H_2(delta) >= 1``.  The registered witness violates that necessary
    condition, hence the integer radius is strictly greater than ``delta*D``.
    """

    if not covering_entropy_witness_holds(
        hot_bits, coefficient_count, radius_fraction
    ):
        raise ValueError("the requested entropy witness does not certify a bound")
    return (
        radius_fraction.numerator * coefficient_count
        // radius_fraction.denominator
        + 1
    )


def joint_rank_one_probe_lower_bound(
    hot_bits: int = HOT_SIDE_BITS,
    coefficient_count: int = REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    span_length: int | None = None,
) -> int:
    """Conservative total coefficient-use lower bound for one query tuple.

    Let ``P`` be the worst-case sum of raw residual weights for an independent
    tuple containing one rank-one query per registered block.  Every ambient
    block vector is a sum of at most ``k`` such tuples, where ``k`` is the
    maximum matrix dimension.  Therefore the advice subspace's covering
    radius is at most ``k*P``.  Combining this with the finite sphere-covering
    witness yields ``P >= ceil(R/k)``.
    """

    if span_length is None:
        span_length = maximum_rank_one_span_length()
    if span_length <= 0:
        raise ValueError("span length must be positive")
    radius = certified_covering_radius_cells(hot_bits, coefficient_count)
    return (radius + span_length - 1) // span_length


def ideal_packed_probe_count(coefficient_bits: int, word_bits: int) -> int:
    """Best-case packed word count, with all useful bits perfectly co-located."""

    if coefficient_bits < 0 or word_bits <= 0:
        raise ValueError("invalid coefficient or word size")
    return (coefficient_bits + word_bits - 1) // word_bits


def derive_frontier() -> dict[str, object]:
    """Return the frozen localization theorem and registered finite frontier."""

    specs = registered_non_embedding_specs()
    coefficient_count = sum(spec.parameters for spec in specs)
    matrix_count = sum(spec.count for spec in specs)
    if coefficient_count != REGISTERED_NON_EMBEDDING_COEFFICIENTS:
        raise AssertionError("registered non-embedding population drifted")

    hot_rate = Fraction(HOT_SIDE_BITS, coefficient_count)
    span_length = maximum_rank_one_span_length()
    cover_radius = certified_covering_radius_cells()
    coefficient_uses = joint_rank_one_probe_lower_bound()
    coefficient_fraction = Fraction(coefficient_uses, coefficient_count)
    target_cells = P50_TARGET_FRACTION * coefficient_count
    ideal_words = ideal_packed_probe_count(coefficient_uses, IDEAL_WORD_BITS)
    ideal_lines = ideal_packed_probe_count(
        coefficient_uses, IDEAL_CACHE_LINE_BITS
    )
    entropy_value = binary_entropy(COVER_RADIUS_FRACTION_WITNESS)

    return {
        "classification": "E0_SCOPED_JOINT_LINEAR_ADVICE_LOCALIZATION_BOUND",
        "model": "SYSTEMATIC_LINEAR_COEFFICIENT_USE_WITH_GLOBAL_ADVICE",
        "normalization": {
            "matrix_count": matrix_count,
            "non_embedding_coefficients": coefficient_count,
            "binary_cold_gib": coefficient_count / (8 * GIB),
            "q4_cold_gib": coefficient_count / (2 * GIB),
            "hot_advice_bits": HOT_SIDE_BITS,
            "hot_advice_gib": HOT_SIDE_BITS / (8 * GIB),
            "hot_advice_rate": str(hot_rate),
            "hot_advice_rate_decimal": float(hot_rate),
            "p50_target_fraction": str(P50_TARGET_FRACTION),
            "p50_target_fraction_decimal": float(P50_TARGET_FRACTION),
            "p50_target_cells": str(target_cells),
            "p50_target_cells_decimal": float(target_cells),
        },
        "localization_theorem": {
            "outside_advice_content_equals_raw_cancellation": True,
            "zero_outside_probe_space": "U intersect V_i",
            "sum_zero_outside_local_dimensions_at_most_hot_dimension": True,
            "fixed_outside_support_dimension_bound": "dim(U_i)+|T|",
            "one_fixed_outside_probe_buys_at_most_local_dimensions": 1,
            "projection_dimensions_are_not_additive": True,
            "query_adaptive_support_union_not_collapsed_to_one_subspace": True,
        },
        "span_cover_witness": {
            "radius_fraction": str(COVER_RADIUS_FRACTION_WITNESS),
            "binary_entropy_radius": entropy_value,
            "hot_rate_plus_entropy": float(hot_rate) + entropy_value,
            "strict_margin_below_one": (
                1.0 - float(hot_rate) - entropy_value
            ),
            "holds": covering_entropy_witness_holds(),
            "certified_covering_radius_cells": cover_radius,
            "maximum_rank_one_tuple_span_length": span_length,
        },
        "registered_lower_bound": {
            "coefficient_uses": coefficient_uses,
            "coefficient_fraction": str(coefficient_fraction),
            "coefficient_fraction_decimal": float(coefficient_fraction),
            "coefficient_percent": 100.0 * float(coefficient_fraction),
            "bound_over_p50_target": float(
                coefficient_fraction / P50_TARGET_FRACTION
            ),
            "p50_target_over_bound": float(
                P50_TARGET_FRACTION / coefficient_fraction
            ),
            "ideal_64_bit_word_probes": ideal_words,
            "ideal_512_bit_line_probes": ideal_lines,
            "ideal_packed_payload_bytes": ideal_words * (IDEAL_WORD_BITS // 8),
        },
        "favorable_cost_grants": {
            "representation": "all 8 GiB are linear binary advice",
            "build_amortization": "FREE",
            "advice_reads_and_linear_decode": "FREE",
            "query_selector_and_address_discovery": "FREE",
            "verification": "FREE",
            "miss_and_fallback": "FREE",
            "kv_workspace_and_runtime_state": "OMITTED",
            "physical_latency": "NOT MAPPED",
        },
        "decision": (
            "ESTABLISH_GLOBAL_LINEAR_ADVICE_LOCALIZATION_"
            "BUT_BOUND_INSUFFICIENT_FOR_TARGET_REJECTION"
        ),
        "claim_boundary": {
            "free_cross_matrix_projection_reuse": "RULED OUT IN DECLARED MODEL",
            "target_sized_direct_sum_for_rank_one_queries": "NOT PROVED",
            "general_nonlinear_data_structure": "NOT RULED OUT",
            "data_dependent_adaptive_probe_algorithm": "NOT RULED OUT",
            "word_or_address_granular_hardware_lower_bound": "NOT PROVED",
            "causal_transformer_query_restriction": "NOT TESTED",
            "model_or_hardware_execution": "NOT RUN",
            "core_candidate": "NONE",
        },
    }
