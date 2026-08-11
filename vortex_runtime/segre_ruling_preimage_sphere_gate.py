"""Preimage-sphere Gate for sparse binary covers of rank-one masks.

The Fourier Gate leaves some higher-slack rectangular dictionaries open.  A
different exact obstruction comes from looking inside linear subspaces of the
binary Segre set.  If ``phi: F_2^S -> F_2^(a x b)`` covers every rank-one mask
with a word of weight at most ``t``, then for every ``x``-by-``y`` tensor
subspace its preimage has dimension ``S-ab+xy`` and must contain

    1 + (2^x - 1)(2^y - 1)

distinct ambient words of weight at most ``t``.  A ``k``-dimensional binary
subspace contains at most ``sum(i=0..t, C(k,i))`` such words: project onto an
information set and observe that projection cannot increase Hamming weight.

This rejects the prior first Fourier survivor, 13 x 89.  The first case that
survives both exact Gates in the registered finite scan is 18 x 36.  Survival
is not a construction.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from vortex_runtime.segre_sparse_cover_fourier_gate import (
    distinct_rectangular_rank_one_masks,
    fourier_cover_case,
    hamming_ball_size,
    proportional_cells,
    target_radius,
)


DECISION = (
    "REJECT_13x89_BY_SEGRE_RULING_PREIMAGE_SPHERE_"
    "KEEP_18x36_SIMULTANEOUS_COVER_OPEN_UNCONSTRUCTED"
)


def low_weight_subspace_upper(*, dimension: int, radius: int) -> int:
    """Maximum possible number of weight-at-most-radius words in a subspace.

    For a binary subspace ``P <= F_2^S`` of dimension ``k``, choose ``k``
    independent coordinate columns of a generator matrix.  Projection to
    those coordinates is a bijection from ``P`` to ``F_2^k`` and never has
    greater weight than the original word.  Therefore the requested count is
    at most the size of the radius-``t`` ball in ``F_2^k``.
    """

    if dimension < 0 or radius < 0:
        raise ValueError("dimension and radius must be nonnegative")
    return sum(comb(dimension, weight) for weight in range(min(radius, dimension) + 1))


def restricted_tensor_preimage_case(
    *,
    rows: int,
    columns: int,
    atoms: int,
    radius: int,
    left_dimension: int,
    right_dimension: int,
) -> dict[str, object]:
    """Apply the exact preimage-sphere necessity to ``A tensor B``.

    ``A`` and ``B`` may be arbitrary factor subspaces with the requested
    dimensions.  Their tensor product has dimension ``xy`` and contains
    exactly ``1+(2^x-1)(2^y-1)`` distinct rank-at-most-one matrices.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if not 1 <= left_dimension <= rows:
        raise ValueError("left dimension is outside the matrix shape")
    if not 1 <= right_dimension <= columns:
        raise ValueError("right dimension is outside the matrix shape")
    ambient_dimension = rows * columns
    if atoms < ambient_dimension:
        raise ValueError("atoms cannot span the ambient matrix space")
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    kernel_dimension = atoms - ambient_dimension
    tensor_dimension = left_dimension * right_dimension
    preimage_dimension = kernel_dimension + tensor_dimension
    required_low_weight_words = 1 + (
        ((1 << left_dimension) - 1) * ((1 << right_dimension) - 1)
    )
    low_weight_upper = low_weight_subspace_upper(
        dimension=preimage_dimension, radius=radius
    )
    ratio = Fraction(low_weight_upper, required_low_weight_words)
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": ambient_dimension,
        "atoms": atoms,
        "radius": radius,
        "kernel_dimension": kernel_dimension,
        "left_dimension": left_dimension,
        "right_dimension": right_dimension,
        "tensor_dimension": tensor_dimension,
        "preimage_dimension": preimage_dimension,
        "required_low_weight_words": str(required_low_weight_words),
        "low_weight_subspace_upper": str(low_weight_upper),
        "upper_to_required_ratio": str(ratio),
        "upper_to_required_ratio_decimal": float(ratio),
        "preimage_sphere_rejects": low_weight_upper < required_low_weight_words,
    }


def strongest_restricted_tensor_case(
    *, rows: int, columns: int, atoms: int | None = None, radius: int | None = None
) -> dict[str, object]:
    """Return the minimum exact sphere/query ratio over all factor dimensions."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    dimension = rows * columns
    cell_count = proportional_cells(dimension) if atoms is None else atoms
    query_radius = target_radius(dimension) if radius is None else radius
    cases = (
        restricted_tensor_preimage_case(
            rows=rows,
            columns=columns,
            atoms=cell_count,
            radius=query_radius,
            left_dimension=left_dimension,
            right_dimension=right_dimension,
        )
        for left_dimension in range(1, rows + 1)
        for right_dimension in range(1, columns + 1)
    )
    return min(
        cases,
        key=lambda case: Fraction(str(case["upper_to_required_ratio"])),
    )


def projected_ruling_cover_requirement(
    *, rows: int, columns: int, atoms: int, radius: int
) -> dict[str, object]:
    """Necessary ordinary covering-code parameters for the larger ruling.

    The preimage of a ruling has dimension ``kernel_dimension+ruling_dimension``.
    Taking an information-set projection turns the shared kernel into a binary
    linear code of length equal to that preimage dimension, dimension equal to
    the kernel dimension, and covering radius at most ``t``.  This is only a
    necessary condition; one such code does not construct simultaneous Segre
    coverage.
    """

    if rows <= 0 or columns <= 0 or atoms < rows * columns or radius < 0:
        raise ValueError("invalid projected-cover parameters")
    kernel_dimension = atoms - rows * columns
    ruling_dimension = max(rows, columns)
    length = kernel_dimension + ruling_dimension
    ball = low_weight_subspace_upper(dimension=length, radius=radius)
    syndromes = 1 << ruling_dimension
    density = Fraction(ball, syndromes)
    return {
        "ruling_dimension": ruling_dimension,
        "projected_code_length": length,
        "projected_code_dimension": kernel_dimension,
        "projected_code_codimension": ruling_dimension,
        "required_covering_radius_at_most": radius,
        "sphere_volume": str(ball),
        "syndrome_count": str(syndromes),
        "sphere_covering_density": str(density),
        "sphere_covering_density_decimal": float(density),
        "ordinary_covering_code_requirement_passes_sphere_bound": ball >= syndromes,
        "ordinary_covering_code_constructed": False,
        "simultaneous_ruling_cover_constructed": False,
    }


def first_unclosed_by_combined_gates(maximum_side: int = 128) -> dict[str, object]:
    """Scan exact subset-slack order through preimage and Fourier Gates."""

    if maximum_side <= 0:
        raise ValueError("maximum side must be positive")
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

    rejected: list[dict[str, object]] = []
    for ratio, rows, columns in ordered:
        restricted = strongest_restricted_tensor_case(rows=rows, columns=columns)
        if restricted["preimage_sphere_rejects"]:
            rejected.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "ordering_ratio": str(ratio),
                    "rejected_by": "PREIMAGE_SPHERE",
                    "witness": restricted,
                }
            )
            continue
        fourier = fourier_cover_case(rows=rows, columns=columns)
        if fourier["fourier_second_moment_rejects"]:
            rejected.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "ordering_ratio": str(ratio),
                    "rejected_by": "FOURIER_SECOND_MOMENT",
                    "witness": fourier,
                }
            )
            continue
        return {
            "maximum_side": maximum_side,
            "capacity_feasible_shape_count": len(ordered),
            "exact_subset_slack_rank": len(rejected) + 1,
            "preceding_rejected_shapes": rejected,
            "first_unclosed_shape": [rows, columns],
            "ordering_ratio": str(ratio),
            "strongest_restricted_case": restricted,
            "fourier_case": fourier,
        }
    raise ValueError("every capacity-feasible rectangle in the scan was rejected")


def derive_audit() -> dict[str, object]:
    prior_survivor = strongest_restricted_tensor_case(rows=13, columns=89)
    if not prior_survivor["preimage_sphere_rejects"]:
        raise AssertionError("the registered 13x89 ruling rejection drifted")
    if (prior_survivor["left_dimension"], prior_survivor["right_dimension"]) != (
        1,
        89,
    ):
        raise AssertionError("the registered 13x89 strongest witness drifted")

    scan = first_unclosed_by_combined_gates()
    if scan["first_unclosed_shape"] != [18, 36]:
        raise AssertionError("the first combined-Gate survivor drifted")
    survivor = dict(scan["strongest_restricted_case"])
    if survivor["preimage_sphere_rejects"]:
        raise AssertionError("the registered survivor unexpectedly closed")
    cover_requirement = projected_ruling_cover_requirement(
        rows=18, columns=36, atoms=758, radius=7
    )

    return {
        "classification": "E0_SEGRE_RULING_PREIMAGE_SPHERE_GATE",
        "contract": {
            "fixed_binary_linear_atom_map": True,
            "arbitrary_atom_values": True,
            "all_rank_one_queries_must_be_covered": True,
            "registered_side_scan_maximum": 128,
        },
        "proof_equations": {
            "preimage_dimension": "dim(phi^-1(A tensor B))=S-ab+xy",
            "required_rank_one_words": "1+(2^x-1)(2^y-1)",
            "information_set_bound": (
                "|P intersect Ball(S,t)| <= sum(i=0..t,C(dim(P),i))"
            ),
        },
        "rejected_prior_survivor": prior_survivor,
        "combined_subset_slack_scan": scan,
        "first_survivor_projected_ruling_cover_requirement": cover_requirement,
        "claim_boundary": {
            "thirteen_by_eighty_nine_rejected": True,
            "all_rectangular_shapes_rejected": False,
            "eighteen_by_thirty_six_constructed": False,
            "ordinary_projected_cover_code_constructed": False,
            "simultaneous_rulings_constructed": False,
            "adaptive_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_numerical_lift_covered": False,
            "joint_32_query_union_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_13x89_BY_FIXED_RULING_INFORMATION_SET_BOUND",
            "PROMOTE_18x36_AS_FIRST_COMBINED_GATE_CONSTRUCTION_TARGET",
            "REQUIRE_PROJECTED_BINARY_LINEAR_146_110_RADIUS_7_COVER",
            "DO_NOT_EQUATE_ONE_ORDINARY_COVER_WITH_SIMULTANEOUS_SEGRE_COVER",
            "KEEP_ADAPTIVE_FINITE_WORD_MODEL_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
