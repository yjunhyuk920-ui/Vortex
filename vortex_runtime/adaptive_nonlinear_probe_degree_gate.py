"""Low-degree rank amplification for arbitrary adaptive nonlinear encodings.

An adaptive depth-``t`` decision tree over stored cells is a cylinder
polynomial involving at most ``t`` cell coordinates per monomial.  Multiplying
the exact characters for ``r`` rank-one queries produces the character of a
rank-at-most-``r`` matrix and uses cylinder degree at most ``r*t``.  Distinct
matrix characters are linearly independent on every injective encoding of the
source.  This yields an exact capacity inequality for arbitrary nonlinear
stored cells, adaptive addresses, and arbitrary deterministic postprocessing.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction

from vortex_runtime.segre_sparse_cover_fourier_gate import (
    REGISTERED_P50_FRACTION,
    proportional_cells,
)


WORD_BITS = 64
FAVORABLE_Q4_LANES = 4

DECISION = (
    "EXTEND_DETERMINANTAL_CAPACITY_TO_ARBITRARY_NONLINEAR_ADAPTIVE_CELLS_"
    "KEEP_CAPACITY_FEASIBLE_WORD_ENCODINGS_OPEN_UNCONSTRUCTED"
)


@lru_cache(maxsize=None)
def _cylinder_dimension_table(
    cells: int, alphabet_bits: int
) -> tuple[int, ...]:
    if cells <= 0 or alphabet_bits <= 0:
        raise ValueError("cell count and alphabet width must be positive")
    nonconstant_univariate_functions = (1 << alphabet_bits) - 1
    dimensions = [1]
    term = 1
    total = 1
    for degree in range(1, cells + 1):
        term = (
            term
            * (cells - degree + 1)
            * nonconstant_univariate_functions
            // degree
        )
        total += term
        dimensions.append(total)
    return tuple(dimensions)


def cylinder_polynomial_dimension(
    *, cells: int, alphabet_bits: int, degree: int
) -> int:
    """Dimension of functions with cell-support degree at most ``degree``."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    table = _cylinder_dimension_table(cells, alphabet_bits)
    return table[min(cells, degree)]


@lru_cache(maxsize=None)
def cumulative_binary_matrix_rank_counts(
    *, rows: int, columns: int
) -> tuple[int, ...]:
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    counts = [1]
    cumulative = 1
    exact_rank_count = 1
    for rank in range(1, min(rows, columns) + 1):
        # From N_r=[a choose r]_2[b choose r]_2|GL(r,2)|,
        #
        # N_r/N_(r-1) =
        #   ((2^a-2^(r-1))(2^b-2^(r-1))) /
        #   (2^(r-1)(2^r-1)).
        #
        # This exact recurrence avoids rebuilding r products for every rank
        # in the complete rectangle scan.
        exact_rank_count = (
            exact_rank_count
            * ((1 << rows) - (1 << (rank - 1)))
            * ((1 << columns) - (1 << (rank - 1)))
            // ((1 << (rank - 1)) * ((1 << rank) - 1))
        )
        cumulative += exact_rank_count
        counts.append(cumulative)
    return tuple(counts)


def adaptive_nonlinear_probe_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    alphabet_bits: int,
    probes: int,
) -> dict[str, object]:
    """Check every determinantal-rank low-degree necessity exactly."""

    if cells <= 0 or alphabet_bits <= 0:
        raise ValueError("cell parameters must be positive")
    if probes < 0 or probes > cells:
        raise ValueError("probe count must lie in [0, cells]")
    ranks = cumulative_binary_matrix_rank_counts(rows=rows, columns=columns)
    checked: list[dict[str, object]] = []
    first_violation: dict[str, object] | None = None
    closest: tuple[Fraction, int] | None = None
    for rank in range(1, len(ranks)):
        degree = min(cells, rank * probes)
        dimension = cylinder_polynomial_dimension(
            cells=cells,
            alphabet_bits=alphabet_bits,
            degree=degree,
        )
        required = ranks[rank]
        ratio = Fraction(dimension, required)
        case = {
            "maximum_matrix_rank": rank,
            "cylinder_degree": degree,
            "cylinder_space_dimension": str(dimension),
            "required_character_count": str(required),
            "dimension_to_required_ratio": str(ratio),
            "dimension_to_required_ratio_decimal": float(ratio),
            "rejected": dimension < required,
        }
        checked.append(case)
        if closest is None or ratio < closest[0]:
            closest = (ratio, len(checked) - 1)
        if first_violation is None and dimension < required:
            first_violation = case
    if closest is None:
        raise AssertionError("at least one rank must be checked")
    return {
        "rows": rows,
        "columns": columns,
        "source_bits": rows * columns,
        "cells": cells,
        "alphabet_bits": alphabet_bits,
        "probes": probes,
        "rejected": first_violation is not None,
        "first_violation": first_violation,
        "closest_case": checked[closest[1]],
        "rank_cases": checked,
    }


def minimum_adaptive_nonlinear_probes(
    *, rows: int, columns: int, cells: int, alphabet_bits: int
) -> dict[str, object]:
    """Return the first probe count passing every degree-capacity inequality."""

    ranks = cumulative_binary_matrix_rank_counts(rows=rows, columns=columns)
    dimensions = _cylinder_dimension_table(cells, alphabet_bits)
    last_rejected: dict[str, object] | None = None
    for probes in range(cells + 1):
        first_bad: dict[str, object] | None = None
        for rank in range(1, len(ranks)):
            degree = min(cells, rank * probes)
            if dimensions[degree] < ranks[rank]:
                ratio = Fraction(dimensions[degree], ranks[rank])
                first_bad = {
                    "probes": probes,
                    "maximum_matrix_rank": rank,
                    "cylinder_degree": degree,
                    "dimension_to_required_ratio": str(ratio),
                    "dimension_to_required_ratio_decimal": float(ratio),
                }
                break
        if first_bad is None:
            return {
                "minimum_probes_not_degree_rejected": probes,
                "last_rejected_witness": last_rejected,
            }
        last_rejected = first_bad
    raise AssertionError("reading every cell must determine every query")


def proportional_nonlinear_word_case(
    *,
    rows: int,
    columns: int,
    word_bits: int = WORD_BITS,
    favorable_value_lanes: int = FAVORABLE_Q4_LANES,
) -> dict[str, object]:
    """Grant padded nonlinear words and compare their capacity threshold."""

    if favorable_value_lanes <= 0:
        raise ValueError("favorable value lanes must be positive")
    source_bits = rows * columns
    granted_bits = proportional_cells(source_bits)
    words = (granted_bits + word_bits - 1) // word_bits
    threshold = minimum_adaptive_nonlinear_probes(
        rows=rows,
        columns=columns,
        cells=words,
        alphabet_bits=word_bits,
    )
    probes = int(threshold["minimum_probes_not_degree_rejected"])
    traffic = Fraction(
        probes * word_bits, favorable_value_lanes * source_bits
    )
    return {
        "rows": rows,
        "columns": columns,
        "source_bit_plane_dimension": source_bits,
        "granted_encoded_bits": granted_bits,
        "padded_words": words,
        "word_bits": word_bits,
        "favorable_value_lanes": favorable_value_lanes,
        **threshold,
        "minimum_physical_bits": probes * word_bits,
        "favorable_traffic_fraction": str(traffic),
        "favorable_traffic_fraction_decimal": float(traffic),
        "registered_target_fraction": str(REGISTERED_P50_FRACTION),
        "minimum_to_target_multiplier": str(
            traffic / REGISTERED_P50_FRACTION
        ),
        "minimum_to_target_multiplier_decimal": float(
            traffic / REGISTERED_P50_FRACTION
        ),
        "registered_target_rejected": traffic > REGISTERED_P50_FRACTION,
        "constructor_exists": False,
    }


def scan_proportional_nonlinear_words(
    *, maximum_side: int = 128
) -> dict[str, object]:
    """Scan the finite registered rectangle range with cached word spaces."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
    best: dict[str, object] | None = None
    first_by_area: dict[str, object] | None = None
    passing = 0
    checked = 0
    for rows in range(1, maximum_side + 1):
        for columns in range(rows, maximum_side + 1):
            case = proportional_nonlinear_word_case(
                rows=rows, columns=columns
            )
            checked += 1
            traffic = Fraction(case["favorable_traffic_fraction"])
            if best is None or traffic < Fraction(
                best["favorable_traffic_fraction"]
            ):
                best = case
            if not case["registered_target_rejected"]:
                passing += 1
                if first_by_area is None:
                    first_by_area = case
                else:
                    current_key = (
                        case["source_bit_plane_dimension"],
                        case["rows"],
                        case["columns"],
                    )
                    first_key = (
                        first_by_area["source_bit_plane_dimension"],
                        first_by_area["rows"],
                        first_by_area["columns"],
                    )
                    if current_key < first_key:
                        first_by_area = case
    if best is None or first_by_area is None:
        raise AssertionError("registered finite scan unexpectedly empty")
    return {
        "maximum_side": maximum_side,
        "rectangles_checked": checked,
        "registered_target_passing_count": passing,
        "smallest_area_capacity_survivor": first_by_area,
        "best_traffic_capacity_case": best,
    }


def derive_audit() -> dict[str, object]:
    """Return the theorem witnesses and the first honest open word target."""

    tiny_gap = minimum_adaptive_nonlinear_probes(
        rows=2, columns=3, cells=7, alphabet_bits=1
    )
    if tiny_gap["minimum_probes_not_degree_rejected"] != 2:
        raise AssertionError("registered 2x3 nonlinear capacity gap drifted")

    bit_frontier = minimum_adaptive_nonlinear_probes(
        rows=31, columns=42, cells=proportional_cells(31 * 42), alphabet_bits=1
    )
    if bit_frontier["minimum_probes_not_degree_rejected"] != 15:
        raise AssertionError("registered nonlinear bit threshold drifted")

    word_frontier = proportional_nonlinear_word_case(rows=31, columns=42)
    if word_frontier["minimum_probes_not_degree_rejected"] != 2:
        raise AssertionError("registered nonlinear word threshold drifted")
    if not word_frontier["registered_target_rejected"]:
        raise AssertionError("registered local word target unexpectedly passed")

    scan = scan_proportional_nonlinear_words()
    first = scan["smallest_area_capacity_survivor"]
    best = scan["best_traffic_capacity_case"]
    if (first["rows"], first["columns"]) != (25, 108):
        raise AssertionError("registered first nonlinear word survivor drifted")
    if first["source_bit_plane_dimension"] != 2_700:
        raise AssertionError("registered first survivor area drifted")
    if (best["rows"], best["columns"]) != (128, 128):
        raise AssertionError("registered best nonlinear word case drifted")
    if scan["registered_target_passing_count"] != 4_257:
        raise AssertionError("registered nonlinear word pass count drifted")

    return {
        "classification": "E0_ADAPTIVE_NONLINEAR_PROBE_DEGREE_GATE",
        "contract": {
            "stored_cells_may_be_arbitrary_checkpoint_functions": True,
            "encoding_need_not_be_linear_or_systematic": True,
            "addresses_may_depend_on_query_and_prior_values": True,
            "postprocessing_is_arbitrary_deterministic_exact_logic": True,
            "all_rank_one_queries_include_matrix_units": True,
            "exactness_forces_encoding_injectivity": True,
            "layout_is_one_block_local_matrix": True,
        },
        "proof_equations": {
            "decision_tree_degree": (
                "depth-t adaptive decoder lies in cylinder degree <= t"
            ),
            "rank_product": "chi_(q1+...+qr)=product_i chi_qi",
            "product_degree": "degree <= r*t",
            "cylinder_dimension": (
                "V_A(S,d)=sum_(j=0)^d C(S,j)(A-1)^j"
            ),
            "necessary_rank_inequality": (
                "RankLeq(a,b,r) <= V_A(S,min(S,r*t)) for every r"
            ),
        },
        "two_by_three_degree_capacity_gap": tiny_gap,
        "thirty_one_by_forty_two_bit_cells": bit_frontier,
        "thirty_one_by_forty_two_nonlinear_words": word_frontier,
        "side_one_to_128_nonlinear_word_scan": scan,
        "claim_boundary": {
            "arbitrary_nonlinear_block_local_bit_cells_covered": True,
            "arbitrary_nonlinear_block_local_word_cells_covered": True,
            "adaptive_addresses_covered": True,
            "capacity_survivor_is_a_construction": False,
            "global_cross_matrix_encoding_covered": False,
            "joint_thirty_two_query_physical_union_covered": False,
            "native_numerical_semantics_covered": False,
            "surviving_runtime_candidate": False,
        },
        "decision": DECISION,
    }
