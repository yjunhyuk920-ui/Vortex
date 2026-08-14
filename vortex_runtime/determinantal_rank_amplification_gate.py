"""Rank-amplification Gates for sparse covers of binary rank-one masks.

If every rank-one matrix is the XOR of at most ``t`` fixed atoms, every
rank-at-most-``r`` matrix is the XOR of at most ``r*t`` atoms: decompose the
matrix into ``r`` rank-one matrices and concatenate their representatives.
The Hamming ball of that amplified radius must therefore be at least as large
as the complete binary determinantal set.

A second exact Gate chooses one distinct representative for every rank-one
matrix and samples ``s`` atom positions.  Every representative of weight at
most ``t`` is contained with probability at least C(s,t)/C(S,t), while all
contained answers lie in an at-most-s-dimensional matrix subspace.  The exact
Segre/product-simplex weight hierarchy bounds that subspace intersection.

Together these Gates reject the first 770 capacity-feasible rectangles in the
registered side-1..128 scan.  The first unclosed shape is 30 x 40 with 1,404
atoms and radius 14.  Survival is not a construction.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb

from vortex_runtime.segre_ruling_preimage_sphere_gate import (
    strongest_restricted_tensor_case,
)
from vortex_runtime.segre_sparse_cover_fourier_gate import (
    distinct_rectangular_rank_one_masks,
    fourier_cover_case,
    hamming_ball_size,
    proportional_cells,
    target_radius,
)


DECISION = (
    "REJECT_FIRST_770_SPARSE_SEGRE_COVERS_BY_DETERMINANTAL_RANK_"
    "AMPLIFICATION_AND_MIXED_WEIGHT_AVERAGING_KEEP_30x40_OPEN_UNCONSTRUCTED"
)


def binary_matrix_rank_count(*, rows: int, columns: int, rank: int) -> int:
    """Number of ``rows x columns`` binary matrices of exact rank ``rank``."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if rank < 0 or rank > min(rows, columns):
        raise ValueError("rank is outside the matrix shape")
    result = 1
    for index in range(rank):
        result = (
            result
            * ((1 << rows) - (1 << index))
            * ((1 << columns) - (1 << index))
            // ((1 << rank) - (1 << index))
        )
    return result


def binary_determinantal_count(
    *, rows: int, columns: int, maximum_rank: int
) -> int:
    """Number of binary matrices of rank at most ``maximum_rank``."""

    if maximum_rank < 0 or maximum_rank > min(rows, columns):
        raise ValueError("maximum rank is outside the matrix shape")
    return sum(
        binary_matrix_rank_count(rows=rows, columns=columns, rank=rank)
        for rank in range(maximum_rank + 1)
    )


def maximum_rank_one_points(
    *, rows: int, columns: int, subspace_dimension: int
) -> int:
    """Exact largest nonzero rank-one intersection of a linear subspace.

    The sides are normalized so ``rows <= columns``.  Write ``s=q*columns+r``.
    The product-simplex generalized-weight formula is attained by ``q`` full
    row strips and one length-``r`` strip and gives the expression below.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if rows > columns:
        rows, columns = columns, rows
    if not 0 <= subspace_dimension <= rows * columns:
        raise ValueError("subspace dimension is outside the matrix shape")
    full_rows, remainder = divmod(subspace_dimension, columns)
    return (
        ((1 << full_rows) - 1) * ((1 << columns) - 1)
        + (1 << full_rows) * ((1 << remainder) - 1)
    )


def rank_amplification_case(
    *, rows: int, columns: int, atoms: int, rank_one_radius: int
) -> dict[str, object]:
    """Apply all determinantal Hamming-ball capacity necessities."""

    if rows <= 0 or columns <= 0 or atoms < rows * columns:
        raise ValueError("invalid matrix or atom parameters")
    if rank_one_radius < 0 or rank_one_radius > atoms:
        raise ValueError("invalid rank-one radius")

    cumulative = 1
    cases: list[dict[str, object]] = []
    closest: tuple[Fraction, int] | None = None
    first_violation: dict[str, object] | None = None
    for rank in range(1, min(rows, columns) + 1):
        cumulative += binary_matrix_rank_count(
            rows=rows, columns=columns, rank=rank
        )
        amplified_radius = min(atoms, rank * rank_one_radius)
        ball = hamming_ball_size(length=atoms, radius=amplified_radius)
        ratio = Fraction(ball, cumulative)
        case = {
            "maximum_rank": rank,
            "amplified_radius": amplified_radius,
            "hamming_ball_size": str(ball),
            "determinantal_count": str(cumulative),
            "ball_to_determinantal_ratio": str(ratio),
            "ball_to_determinantal_ratio_decimal": float(ratio),
            "capacity_passes": ball >= cumulative,
        }
        cases.append(case)
        if closest is None or ratio < closest[0]:
            closest = (ratio, len(cases) - 1)
        if first_violation is None and ball < cumulative:
            first_violation = case
            break

    if closest is None:
        raise AssertionError("rank-amplification scan was unexpectedly empty")
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": rows * columns,
        "atoms": atoms,
        "rank_one_radius": rank_one_radius,
        "rank_amplification_rejects": first_violation is not None,
        "first_violation": first_violation,
        "closest_checked_case": cases[closest[1]],
        "checked_cases": cases,
    }


def mixed_weight_representative_case(
    *, rows: int, columns: int, atoms: int, radius: int
) -> dict[str, object]:
    """Average all representative weights through one sampling constraint.

    For ``s >= t`` and every representative weight ``1 <= k <= t``,

        C(s,k)/C(S,k) >= C(s,t)/C(S,t).

    Thus sampling an ``s``-subset contains at least the latter fraction of all
    nonzero canonical representatives in expectation.  Their distinct images
    lie in an at-most-``s``-dimensional span, whose rank-one intersection is
    bounded exactly by ``maximum_rank_one_points``.
    """

    if rows <= 0 or columns <= 0 or atoms < rows * columns:
        raise ValueError("invalid matrix or atom parameters")
    if radius <= 0 or radius > atoms:
        raise ValueError("radius must be in [1, atoms]")
    dimension = rows * columns
    nonzero_queries = distinct_rectangular_rank_one_masks(rows, columns) - 1
    full_radius_subsets = comb(atoms, radius)

    best_ratio: Fraction | None = None
    witness_dimension: int | None = None
    witness_intersection: int | None = None
    for sampled_atoms in range(radius, dimension + 1):
        intersection = maximum_rank_one_points(
            rows=rows,
            columns=columns,
            subspace_dimension=sampled_atoms,
        )
        upper_ratio = Fraction(
            intersection * full_radius_subsets,
            comb(sampled_atoms, radius) * nonzero_queries,
        )
        if best_ratio is None or upper_ratio < best_ratio:
            best_ratio = upper_ratio
            witness_dimension = sampled_atoms
            witness_intersection = intersection

    if best_ratio is None or witness_dimension is None or witness_intersection is None:
        raise AssertionError("representative sampling scan was unexpectedly empty")
    integer_upper = (
        witness_intersection
        * full_radius_subsets
        // comb(witness_dimension, radius)
    )
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": dimension,
        "atoms": atoms,
        "radius": radius,
        "nonzero_rank_one_queries": str(nonzero_queries),
        "witness_sampled_atoms": witness_dimension,
        "witness_maximum_rank_one_intersection": str(witness_intersection),
        "canonical_representative_upper": str(integer_upper),
        "upper_to_required_ratio": str(best_ratio),
        "upper_to_required_ratio_decimal": float(best_ratio),
        "mixed_weight_averaging_rejects": integer_upper < nonzero_queries,
    }


def _capacity_ordered_shapes(maximum_side: int) -> list[tuple[Fraction, int, int]]:
    ordered: list[tuple[Fraction, int, int]] = []
    for rows in range(1, maximum_side + 1):
        for columns in range(rows, maximum_side + 1):
            dimension = rows * columns
            atoms = proportional_cells(dimension)
            radius = target_radius(dimension)
            if radius <= 0:
                continue
            queries = distinct_rectangular_rank_one_masks(rows, columns)
            ball = hamming_ball_size(length=atoms, radius=radius)
            if ball >= queries:
                ordered.append((Fraction(ball, queries), rows, columns))
    ordered.sort()
    return ordered


def first_unclosed_by_amplification_and_averaging(
    maximum_side: int = 128,
) -> dict[str, object]:
    """Find the first exact subset-slack case surviving both new Gates."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
    ordered = _capacity_ordered_shapes(maximum_side)
    rejection_counts: Counter[str] = Counter()
    mixed_rejections: list[dict[str, object]] = []
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
            violation = dict(amplified["first_violation"])
            rejection_counts[f"RANK_{violation['maximum_rank']}"] += 1
            continue
        averaged = mixed_weight_representative_case(
            rows=rows, columns=columns, atoms=atoms, radius=radius
        )
        if averaged["mixed_weight_averaging_rejects"]:
            rejection_counts["MIXED_WEIGHT_AVERAGING"] += 1
            mixed_rejections.append(averaged)
            continue
        return {
            "maximum_side": maximum_side,
            "capacity_feasible_shape_count": len(ordered),
            "exact_subset_slack_rank": index,
            "preceding_rejection_counts": dict(sorted(rejection_counts.items())),
            "mixed_weight_rejections": mixed_rejections,
            "first_unclosed_shape": [rows, columns],
            "first_unclosed_atoms": atoms,
            "first_unclosed_radius": radius,
            "ordering_ratio": str(ordering_ratio),
            "rank_amplification_case": amplified,
            "mixed_weight_representative_case": averaged,
        }
    raise ValueError("every capacity-feasible rectangle in the scan was rejected")


def derive_audit() -> dict[str, object]:
    rank_two = rank_amplification_case(
        rows=17, columns=43, atoms=855, rank_one_radius=8
    )
    violation = dict(rank_two["first_violation"])
    if violation["maximum_rank"] != 2:
        raise AssertionError("the registered rank-two rejection drifted")

    prior_survivor = mixed_weight_representative_case(
        rows=32, columns=39, atoms=1460, radius=14
    )
    if not prior_survivor["mixed_weight_averaging_rejects"]:
        raise AssertionError("the registered mixed-weight rejection drifted")

    scan = first_unclosed_by_amplification_and_averaging()
    if scan["first_unclosed_shape"] != [30, 40]:
        raise AssertionError("the registered first survivor drifted")
    survivor_amplified = dict(scan["rank_amplification_case"])
    survivor_averaged = dict(scan["mixed_weight_representative_case"])
    if survivor_amplified["rank_amplification_rejects"]:
        raise AssertionError("the registered survivor failed rank amplification")
    if survivor_averaged["mixed_weight_averaging_rejects"]:
        raise AssertionError("the registered survivor failed representative averaging")

    survivor_preimage = strongest_restricted_tensor_case(rows=30, columns=40)
    survivor_fourier = fourier_cover_case(rows=30, columns=40)
    if survivor_preimage["preimage_sphere_rejects"]:
        raise AssertionError("the registered survivor failed the prior preimage Gate")
    if survivor_fourier["fourier_second_moment_rejects"]:
        raise AssertionError("the registered survivor failed the prior Fourier Gate")

    return {
        "classification": "E0_DETERMINANTAL_RANK_AMPLIFICATION_GATE",
        "contract": {
            "fixed_binary_linear_atom_map": True,
            "arbitrary_atom_values": True,
            "every_rank_one_query_has_radius_t_representative": True,
            "registered_side_scan_maximum": 128,
        },
        "proof_equations": {
            "rank_amplification": "RankLeq(r) subset phi(Ball(S,r*t))",
            "determinantal_capacity": (
                "sum(i=0..r,N_rank_i(a,b)) <= sum(j=0..r*t,C(S,j))"
            ),
            "mixed_weight_sampling": (
                "(|Segre|-1) C(s,t)/C(S,t) <= M_rank1(a,b,s)"
            ),
            "rank_one_subspace_maximum": (
                "s=qb+r => M=(2^q-1)(2^b-1)+2^q(2^r-1), a<=b"
            ),
        },
        "rank_two_rejection_example": rank_two,
        "mixed_weight_rejection_example": prior_survivor,
        "combined_subset_slack_scan": scan,
        "survivor_prior_preimage_case": survivor_preimage,
        "survivor_prior_fourier_case": survivor_fourier,
        "claim_boundary": {
            "first_770_capacity_ordered_shapes_rejected": True,
            "all_rectangular_shapes_rejected": False,
            "thirty_by_forty_constructed": False,
            "adaptive_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_numerical_lift_covered": False,
            "joint_32_query_union_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_DETERMINANTAL_RANK_AMPLIFICATION_AS_MANDATORY_GATE",
            "PROMOTE_MIXED_WEIGHT_REPRESENTATIVE_AVERAGING_AS_MANDATORY_GATE",
            "REJECT_17x43_BY_RANK_TWO_CAPACITY",
            "REJECT_32x39_BY_MIXED_WEIGHT_AVERAGING",
            "PROMOTE_30x40_AS_FIRST_COMBINED_GATE_CONSTRUCTION_TARGET",
            "KEEP_ADAPTIVE_FINITE_WORD_MODEL_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
