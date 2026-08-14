"""Recursive support-cancellation Gates for fixed binary atom dictionaries.

The one-shot biorthogonal Gate proves that, in some minimal rank-one
decomposition of a fixed rank-``r`` matrix, representative supports must
overlap.  A positive internal-edge lower bound proves something stronger:
there is one *adjacent pair* whose representatives overlap.  Remove that
pair.  The residual matrix has rank ``r-2`` and satisfies exactly the same
uniform dictionary contract, so the argument can be applied again.

This module records that induction and combines it with the stronger
one-shot overlap-to-cancellation knapsack whenever the latter wins.  All
comparisons are exact integer or :class:`fractions.Fraction` calculations.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from vortex_runtime.biorthogonal_cancellation_gate import (
    biorthogonal_cancellation_rank_case,
)
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
    "REJECT_31x43_AND_32x42_BY_RECURSIVE_BIORTHOGONAL_PAIR_PEELING_"
    "KEEP_31x42_OPEN_UNCONSTRUCTED"
)


def recursive_biorthogonal_cancellation_case(
    *, rows: int, columns: int, atoms: int, rank_one_radius: int
) -> dict[str, object]:
    """Apply pair peeling and exact determinantal counting at every rank.

    Let ``Delta_k`` be a cancellation lower bound for every rank-``k``
    matrix.  Odd rank can peel one arbitrary term, hence
    ``Delta_k >= Delta_(k-1)``.  At even rank the one-shot spectral Gate gives
    a direct lower bound.  In addition, whenever its summed internal-edge
    lower bound is positive, an adjacent pair shares a support coordinate.
    Peeling that pair loses two incidences and leaves rank ``k-2``:

    ``Delta_k >= 2 + Delta_(k-2)``.

    If the edge lower bound is zero, an arbitrary adjacent pair still gives
    ``Delta_k >= Delta_(k-2)``.  The maximum of these valid constructions is
    retained.  The resulting exact-rank radii are accumulated so the final
    Hamming ball covers the complete rank-at-most-``k`` determinantal set.
    """

    if rows <= 0 or columns <= 0 or atoms < rows * columns:
        raise ValueError("invalid matrix or atom parameters")
    if rank_one_radius <= 0 or rank_one_radius > atoms:
        raise ValueError("invalid rank-one radius")

    maximum_rank = min(rows, columns)
    cumulative_determinantal_count = 1
    cancellation_by_rank: dict[int, int] = {0: 0}
    exact_radius_by_rank: dict[int, int] = {0: 0}
    maximum_radius_so_far = 0
    checked: list[dict[str, object]] = []
    first_violation: dict[str, object] | None = None
    closest: tuple[Fraction, int] | None = None

    for rank in range(1, maximum_rank + 1):
        cumulative_determinantal_count += binary_matrix_rank_count(
            rows=rows, columns=columns, rank=rank
        )

        if rank % 2:
            forced_cancellation = cancellation_by_rank[rank - 1]
            proof_source = "PEEL_ONE_TERM_THEN_USE_PREDECESSOR"
            one_shot: dict[str, object] | None = None
            overlap_pair_forced = False
            recursive_candidate = forced_cancellation
        else:
            one_shot = biorthogonal_cancellation_rank_case(
                rows=rows,
                columns=columns,
                atoms=atoms,
                rank_one_radius=rank_one_radius,
                rank=rank,
                determinantal_count=cumulative_determinantal_count,
            )
            overlap_pair_forced = (
                Fraction(one_shot["summed_ordered_internal_edge_lower"]) > 0
            )
            recursive_candidate = cancellation_by_rank[rank - 2] + (
                2 if overlap_pair_forced else 0
            )
            one_shot_candidate = int(one_shot["forced_cancellation"])
            if recursive_candidate > one_shot_candidate:
                forced_cancellation = recursive_candidate
                proof_source = "RECURSIVE_ADJACENT_PAIR_PEELING"
            else:
                forced_cancellation = one_shot_candidate
                proof_source = "ONE_SHOT_PAIR_INTERSECTION_KNAPSACK"

        cancellation_by_rank[rank] = forced_cancellation
        exact_radius = rank * rank_one_radius - forced_cancellation
        exact_radius_by_rank[rank] = exact_radius
        maximum_radius_so_far = max(maximum_radius_so_far, exact_radius)
        ball = hamming_ball_size(
            length=atoms, radius=min(atoms, maximum_radius_so_far)
        )
        ratio = Fraction(ball, cumulative_determinantal_count)
        case: dict[str, object] = {
            "rows": rows,
            "columns": columns,
            "atoms": atoms,
            "rank_one_radius": rank_one_radius,
            "rank": rank,
            "forced_cancellation": forced_cancellation,
            "forced_radius_exact_rank": exact_radius,
            "forced_radius_at_most_rank": maximum_radius_so_far,
            "proof_source": proof_source,
            "overlap_pair_forced": overlap_pair_forced,
            "recursive_candidate_cancellation": recursive_candidate,
            "one_shot_case": one_shot,
            "determinantal_count": str(cumulative_determinantal_count),
            "hamming_ball_size": str(ball),
            "ball_to_determinantal_ratio": str(ratio),
            "ball_to_determinantal_ratio_decimal": float(ratio),
            "recursive_cancellation_rejects": ratio < 1,
        }
        checked.append(case)
        if closest is None or ratio < closest[0]:
            closest = (ratio, len(checked) - 1)
        if first_violation is None and ratio < 1:
            first_violation = case
            break

    if closest is None:
        raise AssertionError("at least one matrix rank must be checked")
    return {
        "rows": rows,
        "columns": columns,
        "atoms": atoms,
        "rank_one_radius": rank_one_radius,
        "recursive_cancellation_rejects": first_violation is not None,
        "first_violation": first_violation,
        "closest_checked_case": checked[closest[1]],
        "checked_cases": checked,
    }


def first_unclosed_by_recursive_cancellation_gates(
    maximum_side: int = 128,
) -> dict[str, object]:
    """Continue the registered capacity-order scan through recursion."""

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
        recursive = recursive_biorthogonal_cancellation_case(
            rows=rows,
            columns=columns,
            atoms=atoms,
            rank_one_radius=radius,
        )
        if recursive["recursive_cancellation_rejects"]:
            rejection_counts["RECURSIVE_BIORTHOGONAL_CANCELLATION"] += 1
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
            "recursive_biorthogonal_cancellation_case": recursive,
        }
    raise ValueError("every capacity-feasible rectangle in the scan was rejected")


def derive_audit() -> dict[str, object]:
    """Return the registered exact witnesses and claim boundary."""

    prior_frontier = recursive_biorthogonal_cancellation_case(
        rows=31, columns=43, atoms=1559, rank_one_radius=15
    )
    if not prior_frontier["recursive_cancellation_rejects"]:
        raise AssertionError("the registered 31x43 rejection drifted")
    violation = dict(prior_frontier["first_violation"])
    if violation["rank"] != 22 or violation["forced_cancellation"] != 6:
        raise AssertionError("the registered recursive witness drifted")

    next_balanced = recursive_biorthogonal_cancellation_case(
        rows=32, columns=42, atoms=1572, rank_one_radius=15
    )
    if not next_balanced["recursive_cancellation_rejects"]:
        raise AssertionError("the registered 32x42 rejection drifted")

    scan = first_unclosed_by_recursive_cancellation_gates()
    if scan["exact_subset_slack_rank"] != 857:
        raise AssertionError("the registered recursive scan rank drifted")
    if scan["first_unclosed_shape"] != [31, 42]:
        raise AssertionError("the registered recursive scan shape drifted")

    return {
        "classification": "E0_RECURSIVE_BIORTHOGONAL_CANCELLATION_GATE",
        "contract": {
            "fixed_binary_linear_atom_map": True,
            "arbitrary_atom_values": True,
            "distinct_nonzero_representative_per_rank_one_query": True,
            "every_rank_one_query_has_radius_t_representative": True,
            "positive_spectral_edge_bound_implies_overlap_pair": True,
            "residual_rank_drops_by_two": True,
            "registered_side_scan_maximum": 128,
        },
        "proof_equations": {
            "positive_edge_witness": "sum_c e(A_c)>0",
            "pair_peel": "Delta_r >= 2 + Delta_(r-2)",
            "arbitrary_pair_peel": "Delta_r >= Delta_(r-2)",
            "odd_rank_peel": "Delta_r >= Delta_(r-1)",
            "exact_rank_radius": "R_r=r*t-Delta_r",
            "rank_at_most_radius": "R_<=r=max_(0<=j<=r) R_j",
        },
        "thirty_one_by_forty_three_rejection": prior_frontier,
        "thirty_two_by_forty_two_rejection": next_balanced,
        "combined_subset_slack_scan": scan,
        "claim_boundary": {
            "first_856_capacity_ordered_shapes_rejected": True,
            "all_rectangular_shapes_rejected": False,
            "thirty_one_by_forty_two_constructed": False,
            "adaptive_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_numerical_lift_covered": False,
            "joint_32_query_union_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_RECURSIVE_PAIR_PEELING_AS_MANDATORY_FIXED_LINEAR_GATE",
            "REJECT_31x43_AT_RANK_22",
            "REJECT_32x42",
            "PROMOTE_31x42_AS_FIRST_COMBINED_GATE_CONSTRUCTION_TARGET",
            "KEEP_ADAPTIVE_FINITE_WORD_MODEL_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }

