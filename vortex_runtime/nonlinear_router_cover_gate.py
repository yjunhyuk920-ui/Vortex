"""Exact cover gates for bounded-word rank-one query routers.

The active nonlinear-word frontier permits stored cells to be arbitrary
functions of a binary checkpoint and permits arbitrary deterministic decoding.
Generic decision-tree degree counting does not reject the registered
``25 x 108`` / 50-word / two-probe seed.

The preregistered implementation first closed two stronger submodels without
assuming that payload cells are linear:

1. nonadaptive two-word routing, where the queried pair is fixed by ``q``;
2. adaptive routing whose first word is an affine function of the checkpoint,
   while the second word may be arbitrary nonlinear and its address may depend
   on the first value.

The proof converts each address class into a low-dimensional linear subspace of
query coefficient vectors.  Exact Segre/product-code geometry then limits how
many rank-one masks one such subspace can contain.

The proof then strengthened: the affine hypothesis is unnecessary.  Repeated
largest-fiber selection closes arbitrary deterministic adaptive nonlinear cells
whenever the resulting Segre-subspace cover is too small.  The registered
25x108 two-probe seed and every target-feasible word router with both sides at
most 128 are closed; the first local shapes not closed by this gate have area
5,400 and require four probes.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from math import comb

from vortex_runtime.segre_sparse_cover_fourier_gate import (
    REGISTERED_P50_FRACTION,
    proportional_cells,
)


REGISTERED_ROWS = 25
REGISTERED_COLUMNS = 108
REGISTERED_CELLS = 50
REGISTERED_WORD_BITS = 64
REGISTERED_PROBES = 2

FAVORABLE_Q4_LANES = 4

DECISION = (
    "REJECT_FULLY_NONLINEAR_ADAPTIVE_TWO_WORD_25x108_AND_ALL_SIDE_LE128_"
    "TARGET_WORD_ROUTERS_REJECT_AREA5400_UNDER_STRONG_32_QUERY_UNION_"
    "KEEP_CAUSAL_REACHABILITY_AND_GLOBAL_NATIVE_PRODUCER_OPEN"
)


def rank_one_mask_count(rows: int, columns: int) -> int:
    """Number of nonzero binary rank-one ``rows x columns`` matrices."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    return ((1 << rows) - 1) * ((1 << columns) - 1)


@lru_cache(maxsize=None)
def _maximum_weighted_deficit(
    rows: int,
    columns: int,
    dimension: int,
) -> tuple[int, tuple[int, ...]]:
    """Solve the exact product-simplex deficit partition.

    Assume ``rows <= columns``.  Schaathun's chained-product generalized
    weight formula for the binary simplex product can be written in terms of
    nondecreasing deficits

    ``0 <= z_0 <= ... <= z_(rows-1) <= columns``

    with ``sum z_i = dimension``.  Maximizing

    ``sum_i 2**(rows-i-1+z_i)``

    gives the maximum number of rank-one projective points in a linear
    subspace of the requested dimension, after subtracting ``2**rows-1``.

    The DP works in the *small subspace dimension* rather than the large
    codimension.  The registered 25x108, dimension-128 case has only 15,487
    memoized states on CPython and is exact integer arithmetic.
    """

    if rows <= 0 or columns <= 0 or rows > columns:
        raise ValueError("require 0 < rows <= columns")
    if not 0 <= dimension <= rows * columns:
        raise ValueError("subspace dimension is outside the matrix space")

    @lru_cache(maxsize=None)
    def visit(
        index: int, minimum_deficit: int, remaining: int
    ) -> tuple[int, tuple[int, ...]] | None:
        if index == rows:
            return (0, ()) if remaining == 0 else None

        slots_after = rows - index - 1
        best: tuple[int, tuple[int, ...]] | None = None
        upper = min(columns, remaining)
        for deficit in range(minimum_deficit, upper + 1):
            suffix_remaining = remaining - deficit
            if suffix_remaining < slots_after * deficit:
                continue
            if suffix_remaining > slots_after * columns:
                continue
            suffix = visit(index + 1, deficit, suffix_remaining)
            if suffix is None:
                continue
            score = (1 << (rows - index - 1 + deficit)) + suffix[0]
            candidate = (score, (deficit,) + suffix[1])
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best

    result = visit(0, 0, dimension)
    if result is None:
        raise AssertionError("deficit partition unexpectedly absent")
    return result


def maximum_rank_one_points_in_subspace(
    rows: int, columns: int, dimension: int
) -> dict[str, object]:
    """Exact Segre intersection maximum for one linear subspace.

    Transposition preserves matrix rank, so the smaller side is used as the
    simplex-product row dimension.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if not 0 <= dimension <= rows * columns:
        raise ValueError("subspace dimension is outside the matrix space")
    left, right = sorted((rows, columns))
    weighted, deficits = _maximum_weighted_deficit(left, right, dimension)
    maximum = weighted - ((1 << left) - 1)
    return {
        "rows": rows,
        "columns": columns,
        "normalized_rows": left,
        "normalized_columns": right,
        "subspace_dimension": dimension,
        "maximum_rank_one_points": maximum,
        "maximizing_deficits": list(deficits),
        "derivation": (
            "Schaathun product-simplex generalized-weight formula in "
            "deficit coordinates"
        ),
    }


def maximum_rank_one_points_closed_form(
    rows: int, columns: int, dimension: int
) -> int:
    """Closed product-simplex intersection formula.

    Normalize ``m=min(rows,columns)``, ``n=max(rows,columns)`` and write
    ``dimension = q*n + r`` with ``0 <= r < n``.  The chained simplex-product
    hierarchy gives

    ``(2**q-1)(2**n-1) + 2**q(2**r-1)``.

    The DP above is retained as an independent exact implementation for the
    registered theorem point and small controls.  The closed form makes finite
    scans over thousands of rectangle shapes cheap.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    ambient = rows * columns
    if not 0 <= dimension <= ambient:
        raise ValueError("subspace dimension is outside the matrix space")
    left, right = sorted((rows, columns))
    quotient, remainder = divmod(dimension, right)
    if quotient > left:
        raise AssertionError("normalized quotient exceeds the short side")
    maximum = ((1 << quotient) - 1) * ((1 << right) - 1)
    if remainder:
        maximum += (1 << quotient) * ((1 << remainder) - 1)
    return maximum


def nonadaptive_nonlinear_pair_cover_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
) -> dict[str, object]:
    """Necessary cover for arbitrary nonlinear cells with fixed query pairs.

    For one unordered pair of cells, all parity queries answered from that pair
    jointly factor through at most ``2*word_bits`` bits.  The joint parity map
    is linear in the checkpoint, so the coefficient vectors assigned to the
    pair have linear span dimension at most ``2*word_bits``.  Singleton reads
    are included by allowing equal addresses.
    """

    if cells <= 0 or word_bits <= 0:
        raise ValueError("cell count and word width must be positive")
    ambient = rows * columns
    subspace_dimension = min(ambient, 2 * word_bits)
    intersection = maximum_rank_one_points_in_subspace(
        rows, columns, subspace_dimension
    )
    address_pairs = cells + comb(cells, 2)
    maximum_covered = address_pairs * int(intersection["maximum_rank_one_points"])
    required = rank_one_mask_count(rows, columns)
    return {
        "model": "NONADAPTIVE_ARBITRARY_NONLINEAR_TWO_WORD",
        "cells": cells,
        "word_bits": word_bits,
        "address_pairs_including_singletons": address_pairs,
        "query_subspace_dimension_per_pair_at_most": subspace_dimension,
        "rank_one_intersection": intersection,
        "maximum_rank_one_queries_coverable_by_union_bound": maximum_covered,
        "required_rank_one_queries": required,
        "coverage_ratio": maximum_covered / required,
        "rejected": maximum_covered < required,
        "cell_values_may_be_arbitrary_nonlinear": True,
        "decoder_may_be_arbitrary": True,
        "second_address_is_value_adaptive": False,
    }


def affine_first_adaptive_cover_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
) -> dict[str, object]:
    """Necessary cover when first cells are affine and second cells arbitrary.

    Fix one first address ``a`` and any reachable first value ``s``.  The fiber
    of an affine word has a difference kernel ``K`` whose annihilator ``H`` has
    dimension at most ``word_bits``.  Partition the queries routed through
    ``a`` by their second address at value ``s``.  For one second address, the
    joint parity answers on the fiber factor through one arbitrary
    ``word_bits``-bit cell, so their restrictions to ``K`` have rank at most
    ``word_bits``.  Hence all query coefficient vectors in that group lie in a
    linear subspace of dimension at most ``2*word_bits``.

    There are at most ``cells`` first addresses and ``cells`` second addresses,
    so the complete query family would have to be covered by at most
    ``cells**2`` such subspaces.
    """

    if cells <= 0 or word_bits <= 0:
        raise ValueError("cell count and word width must be positive")
    ambient = rows * columns
    subspace_dimension = min(ambient, 2 * word_bits)
    intersection = maximum_rank_one_points_in_subspace(
        rows, columns, subspace_dimension
    )
    route_groups = cells * cells
    maximum_covered = route_groups * int(intersection["maximum_rank_one_points"])
    required = rank_one_mask_count(rows, columns)
    return {
        "model": "AFFINE_FIRST_WORD_ADAPTIVE_ARBITRARY_NONLINEAR_SECOND_WORD",
        "cells": cells,
        "word_bits": word_bits,
        "first_word_affine_rank_at_most": word_bits,
        "second_word_may_be_arbitrary_nonlinear": True,
        "second_address_may_depend_on_first_value": True,
        "route_groups_at_most": route_groups,
        "query_subspace_dimension_per_route_group_at_most": subspace_dimension,
        "rank_one_intersection": intersection,
        "maximum_rank_one_queries_coverable_by_union_bound": maximum_covered,
        "required_rank_one_queries": required,
        "coverage_ratio": maximum_covered / required,
        "rejected": maximum_covered < required,
    }


def adaptive_nonlinear_depth_cover_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
    probes: int,
) -> dict[str, object]:
    """Necessary cover for fully nonlinear deterministic adaptive routers.

    Stored cells may be *arbitrary* functions of the checkpoint.  At each
    non-final probe depth, fix one address group and choose the largest value
    fiber of the probed word inside the source set retained so far.  A
    ``word_bits``-bit word shrinks that set by at most ``2**word_bits``.  After
    ``probes-1`` such choices the retained set therefore has size at least

    ``2**D / 2**((probes-1)*word_bits)``.

    Partition the remaining queries by their final address.  Their complete
    joint parity vector factors through one final word and thus takes at most
    ``2**word_bits`` values on the retained source set.  If their coefficient
    span has dimension ``r``, a linear projection onto that span has global
    fibers of size exactly ``2**(D-r)``.  Hence

    ``2**(D-(probes-1)w) <= 2**w * 2**(D-r)``,

    so ``r <= probes*w``.  Repeating the query partition at each depth creates
    at most ``cells**probes`` final route groups.  Therefore every query family
    supported by such a decoder is contained in a union of that many linear
    subspaces, each of dimension at most ``probes*word_bits``.

    This argument needs no linearity, affinity, systematic storage, balanced
    fibers, nonadaptive addressing, or linear final decoder.
    """

    if cells <= 0 or word_bits <= 0 or probes <= 0:
        raise ValueError("cells, word width and probe count must be positive")
    ambient = rows * columns
    subspace_dimension = min(ambient, probes * word_bits)
    maximum_per_group = maximum_rank_one_points_closed_form(
        rows, columns, subspace_dimension
    )
    route_groups = cells**probes
    maximum_covered = route_groups * maximum_per_group
    required = rank_one_mask_count(rows, columns)
    ratio = Fraction(maximum_covered, required)
    return {
        "model": "FULLY_NONLINEAR_DETERMINISTIC_ADAPTIVE_WORD_ROUTER",
        "rows": rows,
        "columns": columns,
        "source_bits": ambient,
        "cells": cells,
        "word_bits": word_bits,
        "probes": probes,
        "route_groups_at_most": route_groups,
        "query_subspace_dimension_per_route_group_at_most": subspace_dimension,
        "maximum_rank_one_points_per_route_group": maximum_per_group,
        "maximum_rank_one_queries_coverable_by_union_bound": maximum_covered,
        "required_rank_one_queries": required,
        "coverage_ratio": str(ratio),
        "coverage_ratio_decimal": float(ratio),
        "rejected": maximum_covered < required,
        "stored_cells_may_be_arbitrary_nonlinear": True,
        "every_address_after_the_first_may_be_value_adaptive": True,
        "decoder_may_be_arbitrary_deterministic_logic": True,
    }


def nonadaptive_nonlinear_depth_cover_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
    probes: int,
) -> dict[str, object]:
    """Exact union-bound Gate for arbitrary nonlinear nonadaptive cells.

    Repeated reads carry no new information, so a nonadaptive query is grouped
    by the set of ``j<=probes`` distinct cells it reads.  All queries in one
    such group factor through ``j*word_bits`` stored bits on the complete source
    universe.  Their coefficient span therefore has dimension at most
    ``j*word_bits``.  Sum the exact Segre intersection bound separately over
    every possible support size instead of pessimistically replacing all groups
    by the largest ``probes*word_bits`` subspace.
    """

    if cells <= 0 or word_bits <= 0 or probes <= 0:
        raise ValueError("cells, word width and probe count must be positive")
    ambient = rows * columns
    maximum_distinct = min(probes, cells)
    terms: list[dict[str, object]] = []
    maximum_covered = 0
    for distinct in range(1, maximum_distinct + 1):
        groups = comb(cells, distinct)
        dimension = min(ambient, distinct * word_bits)
        per_group = maximum_rank_one_points_closed_form(rows, columns, dimension)
        contribution = groups * per_group
        maximum_covered += contribution
        terms.append(
            {
                "distinct_cells": distinct,
                "support_groups": groups,
                "query_subspace_dimension_at_most": dimension,
                "maximum_rank_one_points_per_group": per_group,
                "maximum_union_contribution": contribution,
            }
        )
    required = rank_one_mask_count(rows, columns)
    ratio = Fraction(maximum_covered, required)
    return {
        "model": "NONADAPTIVE_ARBITRARY_NONLINEAR_WORD_ROUTER",
        "rows": rows,
        "columns": columns,
        "source_bits": ambient,
        "cells": cells,
        "word_bits": word_bits,
        "probes": probes,
        "support_terms": terms,
        "maximum_rank_one_queries_coverable_by_union_bound": maximum_covered,
        "required_rank_one_queries": required,
        "coverage_ratio": str(ratio),
        "coverage_ratio_decimal": float(ratio),
        "rejected": maximum_covered < required,
        "stored_cells_may_be_arbitrary_nonlinear": True,
        "decoder_may_be_arbitrary_deterministic_logic": True,
        "value_adaptive_addresses": False,
    }


def one_value_stage_adaptive_cover_case(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
    probes: int,
) -> dict[str, object]:
    """Gate one adaptive routing stage followed by a fixed payload set.

    The first address is query-dependent but value-independent.  After reading
    its word, the algorithm may use that value and the query to choose *all*
    remaining payload addresses.  Those payload addresses may be read in any
    order, but none may depend on another payload value.

    For each first-address query group choose the largest first-word value
    fiber, retaining at least ``2**(D-w)`` sources.  If a subgroup then reads
    ``j`` distinct additional cells, its answers factor through ``j*w`` bits on
    that fiber, so its query span has dimension at most ``(j+1)w``.  There are
    at most ``S*C(S-1,j)`` such subgroups.  Summing exact Segre intersections
    yields a strictly stronger Gate than the unrestricted ``S**t`` route count.
    """

    if cells <= 0 or word_bits <= 0 or probes <= 0:
        raise ValueError("cells, word width and probe count must be positive")
    ambient = rows * columns
    maximum_payload = min(probes - 1, max(0, cells - 1))
    terms: list[dict[str, object]] = []
    maximum_covered = 0
    for payload_cells in range(0, maximum_payload + 1):
        groups = cells * comb(cells - 1, payload_cells)
        dimension = min(ambient, (payload_cells + 1) * word_bits)
        per_group = maximum_rank_one_points_closed_form(rows, columns, dimension)
        contribution = groups * per_group
        maximum_covered += contribution
        terms.append(
            {
                "additional_payload_cells": payload_cells,
                "route_groups": groups,
                "query_subspace_dimension_at_most": dimension,
                "maximum_rank_one_points_per_group": per_group,
                "maximum_union_contribution": contribution,
            }
        )
    required = rank_one_mask_count(rows, columns)
    ratio = Fraction(maximum_covered, required)
    return {
        "model": "ONE_VALUE_STAGE_ADAPTIVE_ARBITRARY_NONLINEAR_WORD_ROUTER",
        "rows": rows,
        "columns": columns,
        "source_bits": ambient,
        "cells": cells,
        "word_bits": word_bits,
        "probes": probes,
        "route_terms": terms,
        "maximum_rank_one_queries_coverable_by_union_bound": maximum_covered,
        "required_rank_one_queries": required,
        "coverage_ratio": str(ratio),
        "coverage_ratio_decimal": float(ratio),
        "rejected": maximum_covered < required,
        "stored_cells_may_be_arbitrary_nonlinear": True,
        "first_value_may_select_the_remaining_payload_set": True,
        "later_payload_values_may_change_addresses": False,
        "decoder_may_be_arbitrary_deterministic_logic": True,
    }


def all_query_union_descriptor_count(
    *, cells: int, word_bits: int, maximum_union_cells: int
) -> int:
    """Count favorable ``(active-address-set, active-values)`` descriptors.

    Fix one source ``x`` and let ``U(x)`` be the union of every cell address
    touched while answering the complete declared query family at ``x``.  The
    pair ``(U(x), E_U(x))`` uniquely determines ``x`` whenever the query family
    contains all matrix units: if another source agrees on those cells, every
    query follows the same transcript by induction and therefore every
    matrix-unit answer agrees.

    If all sources had ``|U(x)| <= u``, at most

    ``sum_(j<=u) C(S,j) 2**(w*j)``

    such descriptors would exist.  This count intentionally grants every
    address subset and every word-value pattern, whether reachable or not.
    """

    if cells <= 0 or word_bits <= 0:
        raise ValueError("cells and word width must be positive")
    if maximum_union_cells < 0 or maximum_union_cells > cells:
        raise ValueError("union bound must lie in [0, cells]")
    return sum(
        comb(cells, used) * (1 << (word_bits * used))
        for used in range(maximum_union_cells + 1)
    )


def minimum_worst_source_all_query_union_cells(
    *, source_bits: int, cells: int, word_bits: int
) -> dict[str, object]:
    """Minimum all-query union forced for at least one source."""

    if source_bits <= 0 or cells <= 0 or word_bits <= 0:
        raise ValueError("source and cell parameters must be positive")
    previous = 0
    source_count = 1 << source_bits
    for union_cells in range(cells + 1):
        current = all_query_union_descriptor_count(
            cells=cells,
            word_bits=word_bits,
            maximum_union_cells=union_cells,
        )
        if current >= source_count:
            return {
                "source_bits": source_bits,
                "cells": cells,
                "word_bits": word_bits,
                "forced_worst_source_all_query_union_cells": union_cells,
                "descriptor_count_through_previous_union": str(previous),
                "descriptor_count_through_forced_union": str(current),
                "source_count": str(source_count),
                "previous_union_is_insufficient": previous < source_count,
                "forced_union_count_is_capacity_feasible": current >= source_count,
            }
        previous = current
    raise AssertionError("reading all encoded cells must identify an injective source")


def independent_query_batch_union_floor(
    *,
    rows: int,
    columns: int,
    cells: int,
    word_bits: int,
    batch_queries: int = 32,
    favorable_value_lanes: int = FAVORABLE_Q4_LANES,
) -> dict[str, object]:
    """Worst-tuple encoded-word union under the strong all-query interface.

    Let one source have all-query union size ``L``.  If fewer than ``K`` chosen
    queries have covered fewer than ``min(K,L)`` cells, some query in the
    complete family touches a new union cell. Greedy selection therefore gives
    a tuple of at most ``K`` queries whose union contains at least
    ``min(K,L)`` cells. This is not a Transformer causal-reachability theorem;
    it is the strong independently selectable rank-one query interface used by
    earlier 32-query E0 physical-union audits.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if batch_queries <= 0 or favorable_value_lanes <= 0:
        raise ValueError("batch size and favorable lanes must be positive")
    source_bits = rows * columns
    union = minimum_worst_source_all_query_union_cells(
        source_bits=source_bits, cells=cells, word_bits=word_bits
    )
    forced_cells = min(
        batch_queries,
        int(union["forced_worst_source_all_query_union_cells"]),
    )
    forced_bits = forced_cells * word_bits
    source_payload_bits = favorable_value_lanes * source_bits
    fraction = Fraction(forced_bits, source_payload_bits)
    target_multiple = fraction / REGISTERED_P50_FRACTION
    return {
        "model": "STRONG_INDEPENDENT_QUERY_PHYSICAL_UNION_GATE",
        "rows": rows,
        "columns": columns,
        "source_bits": source_bits,
        "cells": cells,
        "word_bits": word_bits,
        "batch_queries": batch_queries,
        "all_query_union_capacity": union,
        "forced_distinct_cells_in_some_query_tuple": forced_cells,
        "forced_distinct_encoded_bits": forced_bits,
        "favorable_source_payload_bits": source_payload_bits,
        "forced_union_fraction": str(fraction),
        "forced_union_fraction_decimal": float(fraction),
        "registered_target_fraction": str(REGISTERED_P50_FRACTION),
        "target_multiple": str(target_multiple),
        "target_multiple_decimal": float(target_multiple),
        "target_rejected_under_strong_query_interface": (
            fraction > REGISTERED_P50_FRACTION
        ),
        "stored_cells_may_be_arbitrary_nonlinear": True,
        "addresses_may_be_fully_value_adaptive": True,
        "decoder_may_be_arbitrary_deterministic_logic": True,
        "query_family_contains_all_matrix_units": True,
        "ordinary_transformer_causal_reachability_of_hard_tuple": False,
        "physical_memory_tier_assignment_proved": False,
        "wall_clock_latency_proved": False,
    }


def proportional_word_router_case(
    rows: int, columns: int, *, word_bits: int = REGISTERED_WORD_BITS
) -> dict[str, object]:
    """Apply the complete favorable 64-bit target traffic allowance."""

    if rows <= 0 or columns <= 0 or word_bits <= 0:
        raise ValueError("shape and word width must be positive")
    source_bits = rows * columns
    granted_bits = proportional_cells(source_bits)
    cells = (granted_bits + word_bits - 1) // word_bits
    maximum_physical_bits = (
        REGISTERED_P50_FRACTION.numerator
        * FAVORABLE_Q4_LANES
        * source_bits
        // REGISTERED_P50_FRACTION.denominator
    )
    maximum_probes = maximum_physical_bits // word_bits
    if maximum_probes == 0:
        return {
            "rows": rows,
            "columns": columns,
            "source_bits": source_bits,
            "granted_encoded_bits": granted_bits,
            "padded_words": cells,
            "word_bits": word_bits,
            "maximum_target_probes": 0,
            "physical_target_rejected_before_cover": True,
            "cover_rejected_at_maximum_target_probes": True,
        }
    cover = adaptive_nonlinear_depth_cover_case(
        rows=rows,
        columns=columns,
        cells=cells,
        word_bits=word_bits,
        probes=maximum_probes,
    )
    traffic = Fraction(
        maximum_probes * word_bits, FAVORABLE_Q4_LANES * source_bits
    )
    return {
        "rows": rows,
        "columns": columns,
        "source_bits": source_bits,
        "granted_encoded_bits": granted_bits,
        "padded_words": cells,
        "word_bits": word_bits,
        "maximum_target_probes": maximum_probes,
        "maximum_probe_traffic_fraction": str(traffic),
        "maximum_probe_traffic_fraction_decimal": float(traffic),
        "registered_target_fraction": str(REGISTERED_P50_FRACTION),
        "physical_target_rejected_before_cover": False,
        "cover_rejected_at_maximum_target_probes": cover["rejected"],
        "cover": cover,
    }


def scan_side_bound(maximum_side: int = 128) -> dict[str, object]:
    """Scan every normalized rectangle up to ``maximum_side`` exactly."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
    total = 0
    zero_probe = 0
    cover_rejected = 0
    unclosed: list[list[int]] = []
    for rows in range(1, maximum_side + 1):
        for columns in range(rows, maximum_side + 1):
            total += 1
            case = proportional_word_router_case(rows, columns)
            if case["maximum_target_probes"] == 0:
                zero_probe += 1
            elif case["cover_rejected_at_maximum_target_probes"]:
                cover_rejected += 1
            else:
                unclosed.append([rows, columns])
    return {
        "maximum_side": maximum_side,
        "rectangles_checked": total,
        "zero_64bit_probe_budget": zero_probe,
        "adaptive_cover_rejected": cover_rejected,
        "unclosed_count": len(unclosed),
        "unclosed_shapes": unclosed,
    }


def first_unclosed_by_area(maximum_area: int = 5_400) -> dict[str, object]:
    """Find every minimum-area rectangle not rejected by the new cover gate."""

    if maximum_area <= 0:
        raise ValueError("maximum area must be positive")
    first_area: int | None = None
    first_cases: list[dict[str, object]] = []
    checked = 0
    short_side = 1
    while short_side * short_side <= maximum_area:
        for long_side in range(short_side, maximum_area // short_side + 1):
            area = short_side * long_side
            if first_area is not None and area > first_area:
                continue
            checked += 1
            case = proportional_word_router_case(short_side, long_side)
            if case["cover_rejected_at_maximum_target_probes"]:
                continue
            if first_area is None or area < first_area:
                first_area = area
                first_cases = [case]
            elif area == first_area:
                first_cases.append(case)
        short_side += 1
    return {
        "maximum_area_searched": maximum_area,
        "rectangles_checked_until_first_area": checked,
        "first_unclosed_area": first_area,
        "first_unclosed_cases": first_cases,
    }


def derive_audit() -> dict[str, object]:
    """Return the exact registered gates and claim boundary."""

    intersection = maximum_rank_one_points_in_subspace(
        REGISTERED_ROWS, REGISTERED_COLUMNS, 2 * REGISTERED_WORD_BITS
    )
    expected = (1 << 108) + (1 << 21) - 3
    if intersection["maximum_rank_one_points"] != expected:
        raise AssertionError("registered 25x108 Segre intersection drifted")
    deficits = list(intersection["maximizing_deficits"])
    if deficits[-2:] != [20, 108] or any(deficits[:-2]):
        raise AssertionError("registered maximizing deficit partition drifted")

    nonadaptive = nonadaptive_nonlinear_pair_cover_case(
        rows=REGISTERED_ROWS,
        columns=REGISTERED_COLUMNS,
        cells=REGISTERED_CELLS,
        word_bits=REGISTERED_WORD_BITS,
    )
    affine_adaptive = affine_first_adaptive_cover_case(
        rows=REGISTERED_ROWS,
        columns=REGISTERED_COLUMNS,
        cells=REGISTERED_CELLS,
        word_bits=REGISTERED_WORD_BITS,
    )
    if not nonadaptive["rejected"] or not affine_adaptive["rejected"]:
        raise AssertionError("registered router rejection unexpectedly failed")

    fully_nonlinear = adaptive_nonlinear_depth_cover_case(
        rows=REGISTERED_ROWS,
        columns=REGISTERED_COLUMNS,
        cells=REGISTERED_CELLS,
        word_bits=REGISTERED_WORD_BITS,
        probes=REGISTERED_PROBES,
    )
    if not fully_nonlinear["rejected"]:
        raise AssertionError("registered fully nonlinear router unexpectedly survived")
    if fully_nonlinear["coverage_ratio_decimal"] >= 0.000075:
        raise AssertionError("registered fully nonlinear coverage ratio drifted")

    side_scan = scan_side_bound(128)
    if side_scan["rectangles_checked"] != 8_256:
        raise AssertionError("side-128 rectangle count drifted")
    if side_scan["unclosed_count"] != 0:
        raise AssertionError("a side<=128 nonlinear word case unexpectedly survived")
    if side_scan["zero_64bit_probe_budget"] + side_scan["adaptive_cover_rejected"] != 8_256:
        raise AssertionError("side scan accounting drifted")

    next_frontier = first_unclosed_by_area(5_400)
    if next_frontier["first_unclosed_area"] != 5_400:
        raise AssertionError("minimum unclosed area drifted")
    first_shapes = sorted(
        (case["rows"], case["columns"])
        for case in next_frontier["first_unclosed_cases"]
    )
    if first_shapes != [(24, 225), (25, 216)]:
        raise AssertionError("minimum unclosed shapes drifted")
    for case in next_frontier["first_unclosed_cases"]:
        if case["padded_words"] != 99 or case["maximum_target_probes"] != 4:
            raise AssertionError("minimum unclosed word/probe parameters drifted")

    refined_frontier: list[dict[str, object]] = []
    for case in next_frontier["first_unclosed_cases"]:
        rows = int(case["rows"])
        columns = int(case["columns"])
        cells = int(case["padded_words"])
        probes = int(case["maximum_target_probes"])
        nonadaptive_four = nonadaptive_nonlinear_depth_cover_case(
            rows=rows,
            columns=columns,
            cells=cells,
            word_bits=REGISTERED_WORD_BITS,
            probes=probes,
        )
        one_stage = one_value_stage_adaptive_cover_case(
            rows=rows,
            columns=columns,
            cells=cells,
            word_bits=REGISTERED_WORD_BITS,
            probes=probes,
        )
        if not nonadaptive_four["rejected"] or not one_stage["rejected"]:
            raise AssertionError("registered four-probe refinement unexpectedly survived")
        refined_frontier.append(
            {
                "rows": rows,
                "columns": columns,
                "nonadaptive_four_probe_gate": nonadaptive_four,
                "one_value_stage_adaptive_gate": one_stage,
                "strong_32_query_union_gate": independent_query_batch_union_floor(
                    rows=rows,
                    columns=columns,
                    cells=cells,
                    word_bits=REGISTERED_WORD_BITS,
                    batch_queries=32,
                ),
            }
        )

    for case in refined_frontier:
        union_gate = case["strong_32_query_union_gate"]
        if not union_gate["target_rejected_under_strong_query_interface"]:
            raise AssertionError("registered 32-query union Gate unexpectedly survived")
        forced = union_gate["all_query_union_capacity"]
        if forced["forced_worst_source_all_query_union_cells"] != 84:
            raise AssertionError("registered area-5400 all-query union floor drifted")
        if union_gate["forced_distinct_cells_in_some_query_tuple"] != 32:
            raise AssertionError("registered 32-query distinct-cell floor drifted")
        if union_gate["target_multiple"] != "8":
            raise AssertionError("registered 32-query target multiple drifted")

    return {
        "classification": "E0_NONLINEAR_ROUTER_COVER_GATE",
        "contract": {
            "source": "arbitrary binary matrix",
            "rows": REGISTERED_ROWS,
            "columns": REGISTERED_COLUMNS,
            "source_bits": REGISTERED_ROWS * REGISTERED_COLUMNS,
            "stored_cells": REGISTERED_CELLS,
            "word_bits": REGISTERED_WORD_BITS,
            "stored_bits": REGISTERED_CELLS * REGISTERED_WORD_BITS,
            "query_family": "all nonzero binary rank-one masks",
            "probes": REGISTERED_PROBES,
            "native_numerical_lift": False,
        },
        "exact_rank_one_geometry": intersection,
        "nonadaptive_nonlinear_pair_gate": nonadaptive,
        "affine_first_adaptive_gate": affine_adaptive,
        "fully_nonlinear_adaptive_two_probe_gate": fully_nonlinear,
        "side_one_to_128_complete_target_scan": side_scan,
        "next_local_word_frontier": next_frontier,
        "next_local_word_frontier_refinement": refined_frontier,
        "claim_boundary": {
            "arbitrary_nonlinear_cell_values_in_nonadaptive_gate": True,
            "arbitrary_nonlinear_second_cells_in_affine_adaptive_gate": True,
            "arbitrary_final_boolean_decoders": True,
            "adaptive_second_addresses_in_affine_gate": True,
            "fully_nonlinear_adaptive_first_word_covered": True,
            "arbitrary_nonlinear_cells_at_all_probe_depths_covered": True,
            "side_one_to_128_target_word_router_closed": True,
            "first_unclosed_local_shapes": [[24, 225], [25, 216]],
            "first_unclosed_local_probe_count": 4,
            "first_unclosed_local_padded_words": 99,
            "first_unclosed_nonadaptive_four_probe_router": False,
            "first_unclosed_one_value_stage_adaptive_router": False,
            "first_unclosed_requires_multiple_successive_value_adaptive_routing_stages": True,
            "area5400_strong_independent_32_query_union_target_rejected": True,
            "area5400_forced_worst_source_all_query_union_cells": 84,
            "area5400_forced_32_query_distinct_cells": 32,
            "area5400_strong_query_union_target_multiple": "8",
            "ordinary_transformer_causal_reachability_of_hard_32_tuple": False,
            "global_cross_matrix_encoding_covered": False,
            "joint_32_query_physical_union_covered": (
                "STRONG_INDEPENDENT_QUERY_INTERFACE_ONLY"
            ),
            "native_q4_bf16_fp32_semantics_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
            "target_candidate_constructed": False,
        },
        "obligation_update": {
            "O1": "OPEN",
            "O2": "OPEN",
            "O3": "OPEN",
            "O4": "OPEN",
            "O5": "OPEN",
            "O6": "PARTIAL_E0_ONLY",
        },
        "decision": DECISION,
    }
