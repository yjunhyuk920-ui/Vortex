"""Fourier Gate for near-capacity sparse covers of binary rank-one masks.

Let ``g_1,...,g_S`` be arbitrary binary matrix atoms and let ``phi`` map a
subset of atoms to its XOR.  If every rank-at-most-one matrix has a preimage
in the Hamming ball of radius ``t``, then the Fourier transform of that whole
ball must approximate the explicitly known Fourier transform of the binary
Segre set.

For the tight registered block cases this forces every rank-one dual codeword
to have extreme Hamming bias.  Averaging the squared bias gives a conflicting
upper bound: two unequal atoms have rank at least one, whose correlation with
a uniform nonzero rank-one matrix is strictly below one half.  Surjectivity
also limits how many atom pairs can be equal.

The Gate is scoped.  It rejects the listed near-capacity block parameters; it
does not reject sparse covers with much larger subset-count slack, adaptive
word decoders, or native numerical schemes.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
GLOBAL_ADVICE_BITS = 8 * 8 * (1 << 30)
REGISTERED_P50_FRACTION = Fraction(8, 675)

DECISION = (
    "REJECT_NEAR_CAPACITY_SEGRE_SPARSE_COVERS_BY_FOURIER_SECOND_MOMENT_"
    "KEEP_SLACK_AND_ADAPTIVE_MODELS_OPEN"
)


def distinct_rectangular_rank_one_masks(rows: int, columns: int) -> int:
    """Number of rank-at-most-one ``rows x columns`` matrices over GF(2)."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    return 1 + ((1 << rows) - 1) * ((1 << columns) - 1)


def krawtchouk(order: int, weight: int, length: int) -> int:
    """Binary Krawtchouk polynomial ``K_order(weight; length)``."""

    if length <= 0:
        raise ValueError("length must be positive")
    if order < 0 or order > length or weight < 0 or weight > length:
        raise ValueError("order and weight must lie in [0, length]")
    return sum(
        (-1) ** intersection
        * comb(weight, intersection)
        * comb(length - weight, order - intersection)
        for intersection in range(
            max(0, order - (length - weight)), min(order, weight) + 1
        )
    )


def hamming_ball_character_sum(
    *, length: int, radius: int, dual_weight: int
) -> int:
    """Character sum of a radius-``radius`` Hamming ball."""

    if radius < 0 or radius > length:
        raise ValueError("radius must lie in [0, length]")
    return sum(
        krawtchouk(order, dual_weight, length)
        for order in range(radius + 1)
    )


def hamming_ball_size(*, length: int, radius: int) -> int:
    if length <= 0 or radius < 0 or radius > length:
        raise ValueError("invalid Hamming-ball parameters")
    return sum(comb(length, weight) for weight in range(radius + 1))


def rank_one_fourier_coefficient(
    *, rows: int, columns: int, dual_rank: int, include_zero: bool
) -> int:
    """Character sum of the rectangular binary rank-one set.

    For a dual matrix of rank ``r``, summing over nonzero factor pairs gives

        2**(rows+columns-r) - 2**rows - 2**columns + 1.

    Adding the unique zero matrix contributes one more.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    if dual_rank < 0 or dual_rank > min(rows, columns):
        raise ValueError("dual rank is outside the matrix shape")
    coefficient = (
        (1 << (rows + columns - dual_rank))
        - (1 << rows)
        - (1 << columns)
        + 1
    )
    return coefficient + int(include_zero)


def maximum_equal_ordered_pairs(*, atoms: int, ambient_dimension: int) -> int:
    """Maximize equal ordered atom pairs subject to spanning the ambient space.

    Spanning dimension ``D`` requires at least ``D`` distinct nonzero atoms.
    Convexity puts every one of the ``S-D`` extra copies on one of those atoms,
    giving ``(S-D+1)^2 + D-1`` equal ordered pairs.
    """

    if ambient_dimension <= 0 or atoms < ambient_dimension:
        raise ValueError("atoms must be at least the ambient dimension")
    return (atoms - ambient_dimension + 1) ** 2 + ambient_dimension - 1


def proportional_cells(ambient_dimension: int) -> int:
    if ambient_dimension <= 0:
        raise ValueError("ambient dimension must be positive")
    return (
        ambient_dimension
        * (REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_ADVICE_BITS)
        // REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )


def target_radius(ambient_dimension: int) -> int:
    if ambient_dimension <= 0:
        raise ValueError("ambient dimension must be positive")
    return (
        REGISTERED_P50_FRACTION.numerator * ambient_dimension
        // REGISTERED_P50_FRACTION.denominator
    )


def _integer_ranges(values: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for value in values:
        if not ranges or value != ranges[-1][1] + 1:
            ranges.append([value, value])
        else:
            ranges[-1][1] = value
    return ranges


def fourier_cover_case(
    *, rows: int, columns: int, atoms: int | None = None, radius: int | None = None
) -> dict[str, object]:
    """Apply the exact Fourier/second-moment necessary conditions."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix sides must be positive")
    dimension = rows * columns
    cell_count = proportional_cells(dimension) if atoms is None else atoms
    query_radius = target_radius(dimension) if radius is None else radius
    if cell_count < dimension:
        raise ValueError("atoms cannot span the ambient matrix space")
    if query_radius < 0 or query_radius > cell_count:
        raise ValueError("invalid query radius")

    queries = distinct_rectangular_rank_one_masks(rows, columns)
    ball = hamming_ball_size(length=cell_count, radius=query_radius)
    if ball < queries:
        return {
            "rows": rows,
            "columns": columns,
            "ambient_dimension": dimension,
            "atoms": cell_count,
            "radius": query_radius,
            "query_count": str(queries),
            "hamming_ball_size": str(ball),
            "hamming_ball_to_query_ratio": float(Fraction(ball, queries)),
            "capacity_passes": False,
            "valid_rank_one_dual_weight_ranges": [],
            "fourier_second_moment_rejects": True,
            "rejection_reason": "RAW_SUBSET_CAPACITY_BELOW_QUERY_COUNT",
        }

    slack = ball - queries
    rank_one_coefficient = rank_one_fourier_coefficient(
        rows=rows, columns=columns, dual_rank=1, include_zero=True
    )
    valid_weights = [
        weight
        for weight in range(cell_count + 1)
        if abs(
            hamming_ball_character_sum(
                length=cell_count,
                radius=query_radius,
                dual_weight=weight,
            )
            - rank_one_coefficient
        )
        <= slack
    ]
    if not valid_weights:
        return {
            "rows": rows,
            "columns": columns,
            "ambient_dimension": dimension,
            "atoms": cell_count,
            "radius": query_radius,
            "query_count": str(queries),
            "hamming_ball_size": str(ball),
            "hamming_ball_to_query_ratio": float(Fraction(ball, queries)),
            "capacity_passes": True,
            "fourier_slack": str(slack),
            "valid_rank_one_dual_weight_ranges": [],
            "fourier_second_moment_rejects": True,
            "rejection_reason": "NO_DUAL_WEIGHT_SATISFIES_POINTWISE_FOURIER_NECESSITY",
        }

    minimum_absolute_bias = min(
        abs(cell_count - 2 * weight) for weight in valid_weights
    )
    nonzero_rank_one_count = queries - 1
    rank_one_bias = Fraction(
        rank_one_fourier_coefficient(
            rows=rows,
            columns=columns,
            dual_rank=1,
            include_zero=False,
        ),
        nonzero_rank_one_count,
    )
    if rank_one_bias >= Fraction(1, 2):
        raise AssertionError("nonzero rank-one correlation must be below one half")
    equal_pairs = maximum_equal_ordered_pairs(
        atoms=cell_count, ambient_dimension=dimension
    )
    second_moment_upper = Fraction(equal_pairs, 1) + Fraction(
        cell_count * cell_count - equal_pairs, 1
    ) * rank_one_bias
    second_moment_lower = minimum_absolute_bias**2
    rejected = Fraction(second_moment_lower, 1) > second_moment_upper
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": dimension,
        "atoms": cell_count,
        "radius": query_radius,
        "radius_fraction": str(Fraction(query_radius, dimension)),
        "query_count": str(queries),
        "hamming_ball_size": str(ball),
        "hamming_ball_to_query_ratio": float(Fraction(ball, queries)),
        "capacity_passes": True,
        "fourier_slack": str(slack),
        "rank_one_fourier_coefficient_including_zero": str(
            rank_one_coefficient
        ),
        "valid_rank_one_dual_weight_ranges": _integer_ranges(valid_weights),
        "minimum_absolute_character_bias": minimum_absolute_bias,
        "minimum_squared_character_bias": second_moment_lower,
        "maximum_equal_ordered_atom_pairs": equal_pairs,
        "maximum_nonzero_difference_rank_one_correlation": str(rank_one_bias),
        "second_moment_upper": str(second_moment_upper),
        "second_moment_upper_decimal": float(second_moment_upper),
        "second_moment_gap": str(
            Fraction(second_moment_lower, 1) - second_moment_upper
        ),
        "fourier_second_moment_rejects": rejected,
        "rejection_reason": (
            "EXTREME_DUAL_BIAS_EXCEEDS_SPANNING_ATOM_SECOND_MOMENT"
            if rejected
            else "SECOND_MOMENT_NOT_DECISIVE"
        ),
    }


def kernel_defect_requirement(
    *, rows: int, columns: int, atoms: int, radius: int
) -> dict[str, object]:
    """Necessary low-weight kernel relation for any sparse rank-one cover.

    Fix the larger Segre ruling, choose a radius-``t`` representative for every
    point in that linear subspace, and compare representatives for ``x``,
    ``y``, and ``x+y``.  If the atom map's kernel distance exceeded ``3t``,
    every defect would vanish and the section would be linear.  A binary
    linear space whose every vector has weight at most ``t`` has support below
    ``2t`` and hence dimension below ``2t``.
    """

    if rows <= 0 or columns <= 0 or atoms < rows * columns or radius < 0:
        raise ValueError("invalid kernel-defect parameters")
    ruling_dimension = max(rows, columns)
    condition_applies = ruling_dimension >= 2 * radius
    return {
        "atoms": atoms,
        "ruling_dimension": ruling_dimension,
        "radius": radius,
        "linear_section_maximum_support_strictly_below": 2 * radius,
        "condition_applies": condition_applies,
        "required_kernel_minimum_distance_at_most": (
            3 * radius if condition_applies else None
        ),
        "random_high_girth_dictionary_can_be_promoted": False,
    }


def first_unclosed_by_subset_slack(maximum_side: int = 128) -> dict[str, object]:
    """Scan capacity-feasible rectangles in exact ``|Ball|/|Segre|`` order."""

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

    preceding: list[dict[str, object]] = []
    for ratio, rows, columns in ordered:
        case = fourier_cover_case(rows=rows, columns=columns)
        if not case["fourier_second_moment_rejects"]:
            return {
                "maximum_side": maximum_side,
                "capacity_feasible_shape_count": len(ordered),
                "exact_subset_slack_rank": len(preceding) + 1,
                "preceding_rejected_shapes": preceding,
                "first_unclosed_case": case,
                "ordering_ratio": str(ratio),
            }
        preceding.append(case)
    raise ValueError("every capacity-feasible rectangle in the scan was rejected")


def derive_audit() -> dict[str, object]:
    scan = first_unclosed_by_subset_slack()
    rejected = list(scan["preceding_rejected_shapes"])
    if not all(case["fourier_second_moment_rejects"] for case in rejected):
        raise AssertionError("a registered near-capacity rejection drifted")
    first_unclosed = dict(scan["first_unclosed_case"])
    if (first_unclosed["rows"], first_unclosed["columns"]) != (13, 89):
        raise AssertionError("the first unclosed subset-slack case drifted")
    if len(rejected) != 9 or first_unclosed["fourier_second_moment_rejects"]:
        raise AssertionError("the declared open slack case unexpectedly closed")
    square = next(case for case in rejected if case["rows"] == 23)
    defect = kernel_defect_requirement(rows=13, columns=89, atoms=1353, radius=13)
    return {
        "classification": "E0_SEGRE_SPARSE_COVER_FOURIER_GATE",
        "contract": {
            "registered_non_embedding_binary_bits": (
                REGISTERED_NON_EMBEDDING_COEFFICIENTS
            ),
            "global_advice_bits": GLOBAL_ADVICE_BITS,
            "registered_target_fraction": str(REGISTERED_P50_FRACTION),
            "arbitrary_binary_matrix_atoms": True,
            "binary_single_query_screen_only": True,
        },
        "proof_equations": {
            "coverage": "Segre(a,b) subset phi(Ball(S,t))",
            "pointwise_fourier": (
                "|sum(j=0..t,K_j(k_M;S))-Rhat_rank(M)| <= |Ball|-|Segre|"
            ),
            "rank_one_transform": (
                "Rhat_r=2^(a+b-r)-2^a-2^b+2 including zero"
            ),
            "second_moment": (
                "E_M(sum_j (-1)^<M,g_j>)^2 = sum_jk beta_rank(g_j+g_k)"
            ),
            "spanning_duplicate_bound": "equal ordered pairs <= (S-D+1)^2+D-1",
        },
        "near_capacity_rejections": rejected,
        "registered_square_detail": square,
        "subset_slack_order_scan": {
            "maximum_side": scan["maximum_side"],
            "capacity_feasible_shape_count": scan[
                "capacity_feasible_shape_count"
            ],
            "exact_subset_slack_rank": scan["exact_subset_slack_rank"],
            "ordering_ratio": scan["ordering_ratio"],
            "preceding_rejected_shapes": [
                {
                    "rows": case["rows"],
                    "columns": case["columns"],
                    "hamming_ball_to_query_ratio": case[
                        "hamming_ball_to_query_ratio"
                    ],
                }
                for case in rejected
            ],
            "first_unclosed_shape": [
                first_unclosed["rows"], first_unclosed["columns"]
            ],
        },
        "survivor_kernel_defect_requirement": defect,
        "claim_boundary": {
            "all_atom_values_may_be_arbitrary_matrices": True,
            "all_atom_duplicates_charged": True,
            "near_capacity_shapes_rejected": True,
            "all_rectangular_shapes_rejected": False,
            "thirteen_by_eighty_nine_constructed": False,
            "adaptive_addresses_covered": False,
            "arbitrary_word_decoder_covered": False,
            "native_numerical_lift_covered": False,
            "joint_32_query_union_covered": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_FIRST_NINE_SUBSET_SLACK_ORDER_NEAR_CAPACITY_COVERS",
            "REJECT_RAW_SUBSET_CAPACITY_AS_CONSTRUCTION",
            "REQUIRE_LOW_WEIGHT_KERNEL_DEFECTS_IN_ANY_SLACK_SURVIVOR",
            "KEEP_13x89_SLACK_CASE_OPEN_UNCONSTRUCTED",
            "KEEP_ADAPTIVE_FINITE_WORD_MODEL_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
