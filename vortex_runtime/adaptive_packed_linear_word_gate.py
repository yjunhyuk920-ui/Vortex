"""Capacity Gate for adaptive probes of packed linear checkpoint words.

Each stored physical word may expose ``w`` completely unrelated binary linear
functionals of one checkpoint bit-plane.  The query algorithm may choose its
next address from the query and every earlier word value, and may apply an
arbitrary exact decoder to the returned words.

Run that decoder on the all-zero checkpoint.  Its transcript fixes at most
``p`` word addresses.  Any checkpoint perturbation in the common kernel of
the functionals in those words follows the same transcript.  Exactness of a
rank-one parity query therefore forces its coefficient mask into the span of
at most ``p*w`` exposed functionals.  The exact product-simplex weight
hierarchy bounds how many rank-one masks any such span can contain.  Taking a
union over all word supports gives the finite necessary condition implemented
here.

The result is deliberately scoped to stored *linear* bits and a block-local
layout.  It does not constrain nonlinear checkpoint encodings or a word that
jointly mixes independently charged matrices through a global construction.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from vortex_runtime.determinantal_rank_amplification_gate import (
    maximum_rank_one_points,
)
from vortex_runtime.segre_sparse_cover_fourier_gate import (
    REGISTERED_P50_FRACTION,
    proportional_cells,
)


WORD_BITS = 64
FAVORABLE_Q4_LANES = 4

DECISION = (
    "REJECT_BLOCK_LOCAL_ADAPTIVE_PACKED_LINEAR_WORDS_"
    "KEEP_NONLINEAR_AND_GLOBAL_CROSS_MATRIX_WORDS_OPEN"
)


def nonzero_rank_one_masks(rows: int, columns: int) -> int:
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    return ((1 << rows) - 1) * ((1 << columns) - 1)


def packed_word_support_capacity(
    *, rows: int, columns: int, words: int, word_bits: int, probes: int
) -> int:
    """Union-bound the rank-one masks reachable with ``probes`` words.

    A support of ``j`` words spans at most ``j*word_bits`` binary directions.
    The exact maximum number of nonzero rank-one masks in a subspace of that
    dimension is supplied by the product-simplex generalized weights.  The
    sum can overcount overlaps, so failure is a rigorous impossibility while
    success is only capacity feasibility.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if words <= 0 or word_bits <= 0:
        raise ValueError("word count and width must be positive")
    if probes < 0 or probes > words:
        raise ValueError("probe count must lie in [0, words]")

    ambient = rows * columns
    capacity = 0
    for selected in range(1, probes + 1):
        span_dimension = min(ambient, selected * word_bits)
        intersection = maximum_rank_one_points(
            rows=rows,
            columns=columns,
            subspace_dimension=span_dimension,
        )
        capacity += comb(words, selected) * intersection
    return capacity


def minimum_packed_word_probes(
    *, rows: int, columns: int, words: int, word_bits: int = WORD_BITS
) -> tuple[int, list[dict[str, object]]]:
    """Return the first probe count not rejected by the support union."""

    required = nonzero_rank_one_masks(rows, columns)
    cases: list[dict[str, object]] = []
    for probes in range(1, words + 1):
        capacity = packed_word_support_capacity(
            rows=rows,
            columns=columns,
            words=words,
            word_bits=word_bits,
            probes=probes,
        )
        ratio = Fraction(capacity, required)
        cases.append(
            {
                "probes": probes,
                "maximum_span_dimension": min(
                    rows * columns, probes * word_bits
                ),
                "support_union_capacity": str(capacity),
                "required_nonzero_rank_one_masks": str(required),
                "capacity_to_required_ratio": str(ratio),
                "capacity_to_required_ratio_decimal": float(ratio),
                "rejected": capacity < required,
            }
        )
        if capacity >= required:
            return probes, cases
    raise AssertionError("all stored words together must span the source")


def proportional_packed_word_case(
    *,
    rows: int,
    columns: int,
    word_bits: int = WORD_BITS,
    favorable_value_lanes: int = FAVORABLE_Q4_LANES,
) -> dict[str, object]:
    """Give one bit-plane all proportional advice and pad partial words."""

    if favorable_value_lanes <= 0:
        raise ValueError("favorable value lanes must be positive")
    ambient = rows * columns
    stored_linear_bits = proportional_cells(ambient)
    # Padding the final partial word with more independent summaries can only
    # help the proposed layout, so this is a deliberately favorable grant.
    words = (stored_linear_bits + word_bits - 1) // word_bits
    minimum_probes, cases = minimum_packed_word_probes(
        rows=rows,
        columns=columns,
        words=words,
        word_bits=word_bits,
    )
    traffic_fraction = Fraction(
        minimum_probes * word_bits,
        favorable_value_lanes * ambient,
    )
    return {
        "rows": rows,
        "columns": columns,
        "ambient_bit_plane_dimension": ambient,
        "stored_linear_bits": stored_linear_bits,
        "padded_physical_words": words,
        "word_bits": word_bits,
        "favorable_value_lanes": favorable_value_lanes,
        "minimum_probes_not_count_rejected": minimum_probes,
        "minimum_physical_bits": minimum_probes * word_bits,
        "favorable_source_bits": favorable_value_lanes * ambient,
        "favorable_traffic_fraction": str(traffic_fraction),
        "favorable_traffic_fraction_decimal": float(traffic_fraction),
        "registered_target_fraction": str(REGISTERED_P50_FRACTION),
        "minimum_to_target_multiplier": str(
            traffic_fraction / REGISTERED_P50_FRACTION
        ),
        "minimum_to_target_multiplier_decimal": float(
            traffic_fraction / REGISTERED_P50_FRACTION
        ),
        "registered_target_rejected": (
            traffic_fraction > REGISTERED_P50_FRACTION
        ),
        "probe_cases": cases,
    }


def best_proportional_case(
    *,
    maximum_side: int = 128,
    word_bits: int = WORD_BITS,
    favorable_value_lanes: int = FAVORABLE_Q4_LANES,
) -> dict[str, object]:
    """Find the most favorable rectangular local case in a finite scan."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
    best: dict[str, object] | None = None
    passing_target_count = 0
    checked = 0
    for rows in range(1, maximum_side + 1):
        for columns in range(rows, maximum_side + 1):
            case = proportional_packed_word_case(
                rows=rows,
                columns=columns,
                word_bits=word_bits,
                favorable_value_lanes=favorable_value_lanes,
            )
            checked += 1
            if not case["registered_target_rejected"]:
                passing_target_count += 1
            if best is None or Fraction(case["favorable_traffic_fraction"]) < Fraction(
                best["favorable_traffic_fraction"]
            ):
                best = case
    if best is None:
        raise AssertionError("finite scan unexpectedly empty")
    return {
        "maximum_side": maximum_side,
        "rectangles_checked": checked,
        "registered_target_passing_count": passing_target_count,
        "best_case": best,
    }


def derive_audit() -> dict[str, object]:
    """Return the registered finite witnesses and exact claim boundary."""

    frontier = proportional_packed_word_case(rows=31, columns=42)
    if frontier["minimum_probes_not_count_rejected"] != 8:
        raise AssertionError("registered 31x42 word threshold drifted")
    if frontier["probe_cases"][-2]["probes"] != 7:
        raise AssertionError("registered last rejected word count drifted")
    if not frontier["probe_cases"][-2]["rejected"]:
        raise AssertionError("registered seven-word rejection drifted")

    scan = best_proportional_case()
    best = scan["best_case"]
    if (best["rows"], best["columns"]) != (118, 128):
        raise AssertionError("registered best word rectangle drifted")
    if best["minimum_probes_not_count_rejected"] != 22:
        raise AssertionError("registered best word threshold drifted")
    if scan["registered_target_passing_count"] != 0:
        raise AssertionError("a registered local word case unexpectedly passed")

    return {
        "classification": "E0_ADAPTIVE_PACKED_LINEAR_WORD_GATE",
        "contract": {
            "stored_word_bits_are_arbitrary_binary_linear_forms": True,
            "word_addresses_may_depend_on_query_and_prior_values": True,
            "decoder_postprocessing_is_arbitrary_and_exact": True,
            "zero_checkpoint_transcript_fixes_support": True,
            "layout_is_block_local": True,
            "partial_word_is_padded_to_full_width": True,
            "all_global_advice_is_granted_proportionally_to_one_plane": True,
        },
        "proof_equations": {
            "zero_path_span": "q in span_F2(functionals in zero-path words)",
            "support_span_dimension": "dim <= j*w",
            "segre_intersection": (
                "M(a,b,d)=(2^q-1)(2^b-1)+2^q(2^r-1), d=q*b+r"
            ),
            "necessary_union": (
                "sum_(j=1)^p C(M,j) M(a,b,min(j*w,a*b)) "
                ">= (2^a-1)(2^b-1)"
            ),
        },
        "thirty_one_by_forty_two_case": frontier,
        "side_one_to_128_scan": scan,
        "claim_boundary": {
            "block_local_independent_linear_bits_per_word_rejected": True,
            "adaptive_addresses_covered": True,
            "arbitrary_exact_postprocessing_covered": True,
            "nonlinear_stored_words_covered": False,
            "global_cross_matrix_word_mixing_covered": False,
            "joint_thirty_two_query_union_covered": False,
            "native_numerical_semantics_covered": False,
            "surviving_runtime_candidate": False,
        },
        "decision": DECISION,
    }
