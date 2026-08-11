"""Biorthogonal-decomposition cancellation Gates for sparse Segre covers.

A weight-``t`` representative is fixed for every nonzero rank-one binary
matrix.  For one rank-``r`` matrix, the rank-one terms that can occur in a
minimal decomposition form the anti-flag graph of ``PG(r-1, 2)``.  Adjacent
vertices can occur together in one biorthogonal basis decomposition.

The graph spectrum forces overlaps among the supports of the representatives.
Those overlaps cancel in XOR, so the ordinary ``r*t`` rank-amplification
radius is not attainable for every rank-``r`` matrix.  Exact determinantal
counting can then be repeated at the smaller forced radius.

This module uses even ``r`` only.  In that case the least-eigenvalue magnitude
``2**(3*(r-2)/2)`` is an integer and every numerical comparison below remains
an exact integer or :class:`fractions.Fraction` calculation.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb

from vortex_runtime.determinantal_rank_amplification_gate import (
    _capacity_ordered_shapes,
    binary_matrix_rank_count,
    mixed_weight_representative_case,
    rank_amplification_case,
)
from vortex_runtime.segre_sparse_cover_fourier_gate import (
    hamming_ball_size,
    proportional_cells,
    target_radius,
)


DECISION = (
    "REJECT_30x40_31x39_AND_30x44_BY_BIORTHOGONAL_SUPPORT_"
    "CANCELLATION_KEEP_31x43_OPEN_UNCONSTRUCTED"
)


def anti_flag_graph_parameters(rank: int) -> dict[str, int]:
    """Return exact parameters needed from the binary anti-flag graph.

    Vertices are pairs ``(x, y)`` with ``x != 0`` and ``y*x = 1``.  Two
    vertices are adjacent when both cross pairings vanish.  For ``rank >= 2``
    the graph is regular.  Its least adjacency eigenvalue has magnitude
    ``2**(3*(rank-2)/2)``; this function restricts to even rank so that the
    value is integral.
    """

    if rank < 2 or rank % 2:
        raise ValueError("rank must be an even integer of at least two")
    vertices = ((1 << rank) - 1) * (1 << (rank - 1))
    degree = ((1 << (rank - 1)) - 1) * (1 << (rank - 2))
    least_eigenvalue_magnitude = 1 << (3 * (rank - 2) // 2)
    return {
        "rank": rank,
        "vertices": vertices,
        "degree": degree,
        "least_eigenvalue_magnitude": least_eigenvalue_magnitude,
    }


def minimum_distinct_nonzero_support_weight(
    *, length: int, count: int, radius: int
) -> dict[str, object]:
    """Minimize total weight of ``count`` distinct nonzero ball elements."""

    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if radius <= 0 or radius > length:
        raise ValueError("radius must be in [1, length]")

    remaining = count
    total_weight = 0
    shell_counts: list[dict[str, int]] = []
    for weight in range(1, radius + 1):
        available = comb(length, weight)
        selected = min(remaining, available)
        if selected:
            shell_counts.append(
                {
                    "weight": weight,
                    "available": available,
                    "selected": selected,
                }
            )
            total_weight += weight * selected
            remaining -= selected
        if remaining == 0:
            break
    if remaining:
        raise ValueError("the nonzero Hamming ball cannot hold the representatives")
    return {
        "length": length,
        "count": count,
        "radius": radius,
        "minimum_total_weight": total_weight,
        "minimum_average_weight": Fraction(total_weight, count),
        "shell_counts": shell_counts,
    }


def _minimum_spectral_edge_lower(
    *,
    atoms: int,
    vertices: int,
    degree: int,
    eigenvalue_magnitude: int,
    minimum_total_weight: int,
    maximum_total_weight: int,
) -> Fraction:
    """Minimize the summed Hoffman edge lower bound over possible activity."""

    if minimum_total_weight > maximum_total_weight:
        raise ValueError("minimum activity exceeds maximum activity")

    # F(L)=(d+s)L^2/(S*N)-sL is convex.  The actual integral L lies in the
    # supplied interval, so it is enough to inspect the interval endpoints and
    # the two integers surrounding the vertex.
    numerator = eigenvalue_magnitude * atoms * vertices
    denominator = 2 * (degree + eigenvalue_magnitude)
    floor_vertex = numerator // denominator
    candidates = {
        minimum_total_weight,
        maximum_total_weight,
        max(minimum_total_weight, min(maximum_total_weight, floor_vertex)),
        max(minimum_total_weight, min(maximum_total_weight, floor_vertex + 1)),
    }

    def bound(activity: int) -> Fraction:
        return Fraction(
            (degree + eigenvalue_magnitude) * activity * activity,
            atoms * vertices,
        ) - eigenvalue_magnitude * activity

    minimum = min(bound(activity) for activity in candidates)
    return max(Fraction(0), minimum)


def minimum_cancellation_from_pair_intersections(
    *, rank: int, pair_intersections: int, maximum_cancellation: int
) -> int:
    """Invert the exact multiplicity relation between overlap and XOR loss.

    If one atom coordinate occurs in ``m`` summands, it contributes
    ``C(m,2)`` pair intersections and ``m-(m mod 2)`` cancelled incidences.
    An unbounded knapsack deliberately overestimates the intersections
    possible for each cancellation budget, making the returned lower bound
    conservative.
    """

    if rank <= 0 or pair_intersections < 0 or maximum_cancellation < 0:
        raise ValueError("invalid cancellation parameters")
    if pair_intersections == 0:
        return 0

    unreachable = -1
    maximum_intersections = [unreachable] * (maximum_cancellation + 1)
    maximum_intersections[0] = 0
    for cancellation in range(maximum_cancellation + 1):
        current = maximum_intersections[cancellation]
        if current == unreachable:
            continue
        for multiplicity in range(2, rank + 1):
            cost = multiplicity - (multiplicity & 1)
            target = cancellation + cost
            if target <= maximum_cancellation:
                maximum_intersections[target] = max(
                    maximum_intersections[target],
                    current + comb(multiplicity, 2),
                )
    for cancellation, possible in enumerate(maximum_intersections):
        if possible >= pair_intersections:
            return cancellation
    raise AssertionError("pair-intersection lower bound exceeds physical maximum")


def biorthogonal_cancellation_rank_case(
    *,
    rows: int,
    columns: int,
    atoms: int,
    rank_one_radius: int,
    rank: int,
    determinantal_count: int | None = None,
) -> dict[str, object]:
    """Apply the support-overlap theorem at one even matrix rank."""

    if rows <= 0 or columns <= 0 or atoms < rows * columns:
        raise ValueError("invalid matrix or atom parameters")
    if rank_one_radius <= 0 or rank_one_radius > atoms:
        raise ValueError("invalid rank-one radius")
    if rank > min(rows, columns):
        raise ValueError("rank exceeds the matrix shape")

    graph = anti_flag_graph_parameters(rank)
    representatives = minimum_distinct_nonzero_support_weight(
        length=atoms,
        count=graph["vertices"],
        radius=rank_one_radius,
    )
    minimum_total_weight = int(representatives["minimum_total_weight"])
    edge_lower = _minimum_spectral_edge_lower(
        atoms=atoms,
        vertices=graph["vertices"],
        degree=graph["degree"],
        eigenvalue_magnitude=graph["least_eigenvalue_magnitude"],
        minimum_total_weight=minimum_total_weight,
        maximum_total_weight=graph["vertices"] * rank_one_radius,
    )
    expected_pair_intersections = (
        comb(rank, 2)
        * edge_lower
        / (graph["vertices"] * graph["degree"])
    )
    guaranteed_pair_intersections = (
        expected_pair_intersections.numerator
        + expected_pair_intersections.denominator
        - 1
    ) // expected_pair_intersections.denominator
    forced_cancellation = minimum_cancellation_from_pair_intersections(
        rank=rank,
        pair_intersections=guaranteed_pair_intersections,
        maximum_cancellation=rank * rank_one_radius,
    )
    forced_radius_exact_rank = rank * rank_one_radius - forced_cancellation
    forced_radius_lower_ranks = (rank - 1) * rank_one_radius
    forced_radius_at_most_rank = max(
        forced_radius_exact_rank, forced_radius_lower_ranks
    )

    if determinantal_count is None:
        determinantal_count = sum(
            binary_matrix_rank_count(rows=rows, columns=columns, rank=current)
            for current in range(rank + 1)
        )
    ball = hamming_ball_size(
        length=atoms, radius=min(atoms, forced_radius_at_most_rank)
    )
    ratio = Fraction(ball, determinantal_count)
    hoffman_chromatic_lower_exceeds_atoms = (
        graph["degree"] * graph["degree"]
        > (atoms - 1)
        * (atoms - 1)
        * (1 << (3 * (rank - 2)))
    )
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": rows * columns,
        "atoms": atoms,
        "rank_one_radius": rank_one_radius,
        "rank": rank,
        "anti_flag_graph": graph,
        "minimum_distinct_representative_activity": {
            **representatives,
            "minimum_average_weight": str(
                representatives["minimum_average_weight"]
            ),
            "minimum_average_weight_decimal": float(
                representatives["minimum_average_weight"]
            ),
        },
        "summed_ordered_internal_edge_lower": str(edge_lower),
        "expected_pair_intersections_lower": str(expected_pair_intersections),
        "expected_pair_intersections_lower_decimal": float(
            expected_pair_intersections
        ),
        "guaranteed_pair_intersections": guaranteed_pair_intersections,
        "forced_cancellation": forced_cancellation,
        "forced_radius_exact_rank": forced_radius_exact_rank,
        "forced_radius_lower_ranks": forced_radius_lower_ranks,
        "forced_radius_at_most_rank": forced_radius_at_most_rank,
        "determinantal_count": str(determinantal_count),
        "hamming_ball_size": str(ball),
        "ball_to_determinantal_ratio": str(ratio),
        "ball_to_determinantal_ratio_decimal": float(ratio),
        "hoffman_chromatic_lower_exceeds_atoms": (
            hoffman_chromatic_lower_exceeds_atoms
        ),
        "biorthogonal_cancellation_rejects": ball < determinantal_count,
    }


def biorthogonal_cancellation_case(
    *, rows: int, columns: int, atoms: int, rank_one_radius: int
) -> dict[str, object]:
    """Scan every admissible even rank and return the strongest exact case."""

    cumulative = 1
    checked: list[dict[str, object]] = []
    closest: tuple[Fraction, int] | None = None
    first_violation: dict[str, object] | None = None
    for rank in range(1, min(rows, columns) + 1):
        cumulative += binary_matrix_rank_count(
            rows=rows, columns=columns, rank=rank
        )
        if rank < 2 or rank % 2:
            continue
        case = biorthogonal_cancellation_rank_case(
            rows=rows,
            columns=columns,
            atoms=atoms,
            rank_one_radius=rank_one_radius,
            rank=rank,
            determinantal_count=cumulative,
        )
        checked.append(case)
        ratio = Fraction(case["ball_to_determinantal_ratio"])
        if closest is None or ratio < closest[0]:
            closest = (ratio, len(checked) - 1)
        if first_violation is None and case["biorthogonal_cancellation_rejects"]:
            first_violation = case
            break
    if closest is None:
        raise ValueError("the matrix shape has no even rank to check")
    return {
        "rows": rows,
        "columns": columns,
        "atoms": atoms,
        "rank_one_radius": rank_one_radius,
        "biorthogonal_cancellation_rejects": first_violation is not None,
        "first_violation": first_violation,
        "closest_checked_case": checked[closest[1]],
        "checked_cases": checked,
    }


def first_unclosed_by_all_sparse_cover_gates(
    maximum_side: int = 128,
) -> dict[str, object]:
    """Continue the exact capacity-ordered scan through the new Gate."""

    ordered = _capacity_ordered_shapes(maximum_side)
    rejection_counts: Counter[str] = Counter()
    for index, (ordering_ratio, rows, columns) in enumerate(ordered, start=1):
        dimension = rows * columns
        atoms = proportional_cells(dimension)
        radius = target_radius(dimension)
        amplified = rank_amplification_case(
            rows=rows,
            columns=columns,
            atoms=atoms,
            rank_one_radius=radius,
        )
        if amplified["rank_amplification_rejects"]:
            rejection_counts["RANK_AMPLIFICATION"] += 1
            continue
        averaged = mixed_weight_representative_case(
            rows=rows, columns=columns, atoms=atoms, radius=radius
        )
        if averaged["mixed_weight_averaging_rejects"]:
            rejection_counts["MIXED_WEIGHT_AVERAGING"] += 1
            continue
        cancellation = biorthogonal_cancellation_case(
            rows=rows,
            columns=columns,
            atoms=atoms,
            rank_one_radius=radius,
        )
        if cancellation["biorthogonal_cancellation_rejects"]:
            rejection_counts["BIORTHOGONAL_CANCELLATION"] += 1
            continue
        return {
            "maximum_side": maximum_side,
            "capacity_feasible_shape_count": len(ordered),
            "exact_subset_slack_rank": index,
            "preceding_rejection_counts": dict(sorted(rejection_counts.items())),
            "first_unclosed_shape": [rows, columns],
            "first_unclosed_atoms": atoms,
            "first_unclosed_radius": radius,
            "ordering_ratio": str(ordering_ratio),
            "rank_amplification_case": amplified,
            "mixed_weight_representative_case": averaged,
            "biorthogonal_cancellation_case": cancellation,
        }
    raise ValueError("every capacity-feasible rectangle in the scan was rejected")


def derive_audit() -> dict[str, object]:
    prior_frontier = biorthogonal_cancellation_case(
        rows=30, columns=40, atoms=1404, rank_one_radius=14
    )
    if not prior_frontier["biorthogonal_cancellation_rejects"]:
        raise AssertionError("the registered 30x40 rejection drifted")
    prior_violation = dict(prior_frontier["first_violation"])
    if prior_violation["rank"] != 22:
        raise AssertionError("the registered 30x40 witness rank drifted")

    next_frontier = biorthogonal_cancellation_case(
        rows=31, columns=39, atoms=1414, rank_one_radius=14
    )
    if not next_frontier["biorthogonal_cancellation_rejects"]:
        raise AssertionError("the registered 31x39 rejection drifted")

    scan = first_unclosed_by_all_sparse_cover_gates()
    if scan["first_unclosed_shape"] != [31, 43]:
        raise AssertionError("the registered first survivor drifted")
    survivor = dict(scan["biorthogonal_cancellation_case"])
    if survivor["biorthogonal_cancellation_rejects"]:
        raise AssertionError("the registered survivor failed the new Gate")

    return {
        "classification": "E0_BIORTHOGONAL_DECOMPOSITION_CANCELLATION_GATE",
        "contract": {
            "fixed_binary_linear_atom_map": True,
            "arbitrary_atom_values": True,
            "distinct_nonzero_representative_per_rank_one_query": True,
            "every_rank_one_query_has_radius_t_representative": True,
            "even_rank_spectral_cases_only": True,
            "registered_side_scan_maximum": 128,
        },
        "proof_equations": {
            "anti_flag_vertices": "N_r=(2^r-1)2^(r-1)",
            "anti_flag_degree": "d_r=(2^(r-1)-1)2^(r-2)",
            "least_eigenvalue": "lambda_min=-2^(3(r-2)/2)",
            "summed_internal_edges": (
                "sum_c e(A_c) >= (d+s)L^2/(S*N)-sL"
            ),
            "pair_intersections": (
                "E[I] >= C(r,2) sum_c e(A_c)/(N*d)"
            ),
            "xor_cancellation": (
                "Delta=sum_c(m_c-(m_c mod 2)); I=sum_c C(m_c,2)"
            ),
        },
        "thirty_by_forty_rejection": prior_frontier,
        "thirty_one_by_thirty_nine_rejection": next_frontier,
        "combined_subset_slack_scan": scan,
        "claim_boundary": {
            "first_846_capacity_ordered_shapes_rejected": True,
            "all_rectangular_shapes_rejected": False,
            "thirty_one_by_forty_three_constructed": False,
            "adaptive_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_numerical_lift_covered": False,
            "joint_32_query_union_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_BIORTHOGONAL_CANCELLATION_AS_MANDATORY_FIXED_LINEAR_GATE",
            "REJECT_30x40_AT_RANK_22",
            "REJECT_31x39_AT_RANK_22",
            "REJECT_30x44_AT_RANK_22",
            "PROMOTE_31x43_AS_FIRST_COMBINED_GATE_CONSTRUCTION_TARGET",
            "KEEP_ADAPTIVE_FINITE_WORD_MODEL_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
