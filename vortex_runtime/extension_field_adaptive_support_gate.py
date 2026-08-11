"""Adaptive support Gate for extension-field linear checkpoint cells.

An ``m x k`` binary matrix can be packed as ``k`` symbols of
``E = GF(2**m)``.  A tempting exact data structure stores ``S`` arbitrary
``E``-linear forms of those symbols and, for each rank-one query, adaptively
reads at most ``t`` forms.  The coefficients, addresses, and final Boolean
post-processing may all depend on the query and on previously read values.

Adaptivity does not remove a sparse-support necessity.  Run the decoder on
the all-zero source.  Every observed linear cell is zero, fixing one support
``J``.  Every source perturbation in the common kernel of those cells follows
the same zero transcript, so exactness forces the requested linear functional
into the ``E``-span of ``J``.  A ``j``-dimensional ``E``-span intersects the
binary query space in at most ``2**j`` vectors.  Consequently

    sum(comb(S, j) * 2**j for j in range(t + 1)) >= 2**k

is necessary even for deterministic adaptive addressing and arbitrary exact
post-processing.  This module evaluates that condition and a block-local
aggregate storage/traffic relaxation.  It does not cover nonlinear stored
cells or cells that mix independent matrix blocks.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import product


getcontext().prec = 80

REGISTERED_NON_EMBEDDING_COEFFICIENTS = 403_747_897_344
GLOBAL_HOT_BITS = 8 * 8 * (1 << 30)
REGISTERED_DFLOAT_BYTES = 551_220_000_000

# Logical coefficient-work allowance.  The four-lane value grants the whole
# packed-Q4 traffic budget to one varying binary plane, which is strictly more
# favorable than the actual complete native calculation.
REGISTERED_P50_FRACTION = Fraction(8, 675)
FOUR_Q4_LANE_TRAFFIC_FRACTION = 4 * REGISTERED_P50_FRACTION

ONE_PLANE_PROPORTIONAL_EXPANSION = Fraction(
    REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_HOT_BITS,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)
ALL_Q4_STORAGE_TO_ONE_PLANE_EXPANSION = Fraction(
    4 * REGISTERED_NON_EMBEDDING_COEFFICIENTS + GLOBAL_HOT_BITS,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)
ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION = Fraction(
    8 * REGISTERED_DFLOAT_BYTES + GLOBAL_HOT_BITS,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
)

DECISION = (
    "REJECT_BLOCK_LOCAL_EXTENSION_FIELD_LINEAR_CELLS_EVEN_WITH_ADAPTIVE_"
    "ADDRESSES_KEEP_NONLINEAR_AND_CROSS_BLOCK_CELLS_OPEN"
)


def sparse_support_union_bound(*, cells: int, probes: int) -> int:
    """Return ``sum_{j<=probes} C(cells,j) 2**j`` exactly."""

    if cells <= 0 or probes < 0 or probes > cells:
        raise ValueError("invalid cell/probe parameters")
    term = 1
    total = 1
    for weight in range(1, probes + 1):
        term = term * 2 * (cells - weight + 1) // weight
        total += term
    return total


def minimum_sparse_support_radius(*, directions: int, cells: int) -> int:
    """Return the first radius whose support union can count all directions."""

    if directions <= 0 or cells < directions:
        raise ValueError("cells must span the positive direction dimension")
    threshold = 1 << directions
    term = 1
    total = 1
    for radius in range(1, cells + 1):
        term = term * 2 * (cells - radius + 1) // radius
        total += term
        if total >= threshold:
            return radius
    raise AssertionError("the full support family must cover every direction")


def _support_snapshot(*, directions: int, cells: int, probes: int) -> dict[str, object]:
    bound = sparse_support_union_bound(cells=cells, probes=probes)
    bit_length = bound.bit_length()
    return {
        "probes": probes,
        "probe_fraction": str(Fraction(probes, directions)),
        "probe_fraction_decimal": float(Fraction(probes, directions)),
        "support_union_upper_bound_bit_length": bit_length,
        "support_union_upper_bound_lt": f"2^{bit_length}",
        "required_binary_direction_count": f"2^{directions}",
        "exponent_gap_at_least": max(0, directions - bit_length),
        "count_condition_passes": bound >= (1 << directions),
    }


def proportional_block_case(direction_dimension: int) -> dict[str, object]:
    """Evaluate one proportionally allocated extension-field block exactly."""

    if direction_dimension <= 0:
        raise ValueError("direction dimension must be positive")
    cells = (
        direction_dimension * ONE_PLANE_PROPORTIONAL_EXPANSION.numerator
        // ONE_PLANE_PROPORTIONAL_EXPANSION.denominator
    )
    minimum = minimum_sparse_support_radius(
        directions=direction_dimension, cells=cells
    )
    logical_target = (
        direction_dimension * REGISTERED_P50_FRACTION.numerator
        // REGISTERED_P50_FRACTION.denominator
    )
    traffic_target = (
        direction_dimension * FOUR_Q4_LANE_TRAFFIC_FRACTION.numerator
        // FOUR_Q4_LANE_TRAFFIC_FRACTION.denominator
    )
    previous = sparse_support_union_bound(cells=cells, probes=minimum - 1)
    attained = sparse_support_union_bound(cells=cells, probes=minimum)
    return {
        "binary_direction_dimension": direction_dimension,
        "field_cells": cells,
        "storage_expansion": str(Fraction(cells, direction_dimension)),
        "minimum_probes": minimum,
        "minimum_probe_fraction": str(Fraction(minimum, direction_dimension)),
        "minimum_probe_fraction_decimal": float(
            Fraction(minimum, direction_dimension)
        ),
        "minimum_to_logical_target_multiplier": (
            None
            if logical_target == 0
            else float(Fraction(minimum, logical_target))
        ),
        "minimum_to_four_lane_traffic_target_multiplier": (
            None
            if traffic_target == 0
            else float(Fraction(minimum, traffic_target))
        ),
        "logical_target": _support_snapshot(
            directions=direction_dimension,
            cells=cells,
            probes=logical_target,
        ),
        "four_lane_traffic_target": _support_snapshot(
            directions=direction_dimension,
            cells=cells,
            probes=traffic_target,
        ),
        "minimum_minus_one_union_bit_length": previous.bit_length(),
        "minimum_union_bit_length": attained.bit_length(),
        "minimum_is_exact": previous < (1 << direction_dimension) <= attained,
    }


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _coverage_exponent(*, probe_fraction: Decimal, storage_expansion: Decimal) -> Decimal:
    """Return ``p log2(2 e lambda / p)`` from the aggregate relaxation."""

    if probe_fraction <= 0 or storage_expansion <= 0:
        raise ValueError("fractions must be positive")
    two = Decimal(2)
    return probe_fraction * (
        (two * Decimal(1).exp() * storage_expansion / probe_fraction).ln()
        / two.ln()
    )


def _minimum_aggregate_probe_fraction(storage_expansion: Fraction) -> Decimal:
    """Solve ``p log2(2 e lambda/p)=1`` by monotone bisection."""

    expansion = _decimal(storage_expansion)
    low = Decimal(0)
    high = Decimal(1)
    for _ in range(280):
        middle = (low + high) / 2
        if _coverage_exponent(
            probe_fraction=middle, storage_expansion=expansion
        ) < 1:
            low = middle
        else:
            high = middle
    return high


def required_storage_expansion(probe_fraction: Fraction) -> Decimal:
    """Invert the aggregate necessity for a requested probe fraction."""

    p = _decimal(probe_fraction)
    two = Decimal(2)
    e = Decimal(1).exp()
    return p * ((two.ln() / p).exp()) / (two * e)


def aggregate_block_local_case(
    *, name: str, storage_expansion: Fraction, target_fraction: Fraction
) -> dict[str, object]:
    """Evaluate arbitrary storage allocation among block-local linear cells.

    For block ``i``, the exact count and the standard binomial bound give

        N_i <= T_i log2(2 e A_i/T_i),

    where ``N_i=m_i*k_i`` source bits, ``A_i=m_i*S_i`` stored bits, and
    ``T_i=m_i*t_i`` probed bits.  Concavity of ``T log(A/T)`` then gives the
    same inequality for totals, so allocation across blocks cannot evade it.
    """

    p = _decimal(target_fraction)
    expansion = _decimal(storage_expansion)
    exponent = _coverage_exponent(
        probe_fraction=p, storage_expansion=expansion
    )
    minimum = _minimum_aggregate_probe_fraction(storage_expansion)
    required = required_storage_expansion(target_fraction)
    return {
        "name": name,
        "storage_expansion": str(storage_expansion),
        "storage_expansion_decimal": str(expansion),
        "target_fraction": str(target_fraction),
        "target_fraction_decimal": str(p),
        "target_coverage_exponent_upper": str(exponent),
        "target_rejected": exponent < 1,
        "minimum_probe_fraction_relaxation": str(minimum),
        "minimum_to_target_multiplier": str(minimum / p),
        "required_storage_expansion_at_target": str(required),
        "available_to_required_storage_ratio": str(expansion / required),
    }


def rational_dfloat_rejection_witness(
    target_fraction: Fraction,
) -> dict[str, object]:
    """Give an integer-only rejection using ``e<3`` and ``lambda<12``."""

    if ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION >= 12:
        raise AssertionError("the registered DFloat grant no longer fits lambda<12")
    argument_upper = Fraction(72, 1) / target_fraction
    exponent = 0
    while Fraction(1 << exponent, 1) <= argument_upper:
        exponent += 1
    capacity_upper = target_fraction * exponent
    return {
        "e_upper": 3,
        "storage_expansion_upper": 12,
        "log_argument_strict_upper": f"2^{exponent}",
        "capacity_exponent_strict_upper": str(capacity_upper),
        "capacity_exponent_strict_upper_decimal": float(capacity_upper),
        "rejects": capacity_upper < 1,
    }


def _gf4_multiply(left: int, right: int) -> int:
    """Multiply two polynomial-basis GF(4) elements modulo x^2+x+1."""

    if not 0 <= left < 4 or not 0 <= right < 4:
        raise ValueError("GF(4) elements must be in [0,3]")
    raw = 0
    for bit in range(2):
        if (right >> bit) & 1:
            raw ^= left << bit
    if raw & 0b1000:
        raw ^= 0b1110
    if raw & 0b0100:
        raw ^= 0b0111
    return raw


def _gf4_span(generators: tuple[tuple[int, ...], ...]) -> set[tuple[int, ...]]:
    width = len(generators[0]) if generators else 3
    result: set[tuple[int, ...]] = set()
    for coefficients in product(range(4), repeat=len(generators)):
        vector = [0] * width
        for coefficient, generator in zip(coefficients, generators):
            for index, value in enumerate(generator):
                vector[index] ^= _gf4_multiply(coefficient, value)
        result.add(tuple(vector))
    return result


def exhaustive_gf4_intersection_control() -> dict[str, object]:
    """Exhaustively check the intersection lemma in ``GF(4)^3``."""

    vectors = tuple(product(range(4), repeat=3))
    binary = set(product((0, 1), repeat=3))
    maxima: dict[int, int] = {0: 1}
    witnesses: dict[int, list[list[int]]] = {0: []}
    for generator_count in (1, 2):
        best = 0
        witness: tuple[tuple[int, ...], ...] = ()
        for generators in product(vectors, repeat=generator_count):
            intersection = len(_gf4_span(generators) & binary)
            if intersection > best:
                best = intersection
                witness = generators
        maxima[generator_count] = best
        witnesses[generator_count] = [list(vector) for vector in witness]
    return {
        "ambient": "GF(4)^3",
        "maximum_binary_intersection_by_generator_count": maxima,
        "expected_upper_bounds": {0: 1, 1: 2, 2: 4},
        "witnesses": witnesses,
        "passes": maxima == {0: 1, 1: 2, 2: 4},
    }


def derive_audit() -> dict[str, object]:
    """Return the exact registered Gate and explicit claim boundary."""

    dimensions = (1_024, 16_384, 53_248, 128_256)
    blocks = [proportional_block_case(value) for value in dimensions]
    if any(case["logical_target"]["count_condition_passes"] for case in blocks):
        raise AssertionError("a registered logical target unexpectedly passed")
    if any(
        case["four_lane_traffic_target"]["count_condition_passes"]
        for case in blocks
    ):
        raise AssertionError("a registered four-lane target unexpectedly passed")

    q4_traffic = aggregate_block_local_case(
        name="ALL_PACKED_Q4_BITS_PLUS_8GIB_GRANTED_TO_ONE_BINARY_PLANE",
        storage_expansion=ALL_Q4_STORAGE_TO_ONE_PLANE_EXPANSION,
        target_fraction=FOUR_Q4_LANE_TRAFFIC_FRACTION,
    )
    dfloat_traffic = aggregate_block_local_case(
        name="ALL_DFLOAT11_BITS_PLUS_8GIB_GRANTED_TO_ONE_BINARY_PLANE",
        storage_expansion=ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION,
        target_fraction=FOUR_Q4_LANE_TRAFFIC_FRACTION,
    )
    dfloat_work = aggregate_block_local_case(
        name="ALL_DFLOAT11_BITS_PLUS_8GIB_LOGICAL_WORK_GATE",
        storage_expansion=ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION,
        target_fraction=REGISTERED_P50_FRACTION,
    )
    rational_witness = rational_dfloat_rejection_witness(
        FOUR_Q4_LANE_TRAFFIC_FRACTION
    )
    small_control = exhaustive_gf4_intersection_control()
    if not all(
        case["target_rejected"]
        for case in (q4_traffic, dfloat_traffic, dfloat_work)
    ):
        raise AssertionError("the aggregate registered rejection drifted")
    if not rational_witness["rejects"] or not small_control["passes"]:
        raise AssertionError("a proof control failed")

    required = required_storage_expansion(FOUR_Q4_LANE_TRAFFIC_FRACTION)
    required_tib = (
        required
        * Decimal(REGISTERED_NON_EMBEDDING_COEFFICIENTS)
        / Decimal(8)
        / (Decimal(1024) ** 4)
    )
    return {
        "classification": "E0_EXTENSION_FIELD_ADAPTIVE_SUPPORT_GATE",
        "contract": {
            "source_per_block": "beta in GF(2^m)^k",
            "stored_cells": "arbitrary GF(2^m)-linear forms of beta",
            "query": "Trace(alpha * <u,beta>), alpha!=0, u in GF(2)^k",
            "deterministic_query_dependent_addresses": True,
            "addresses_may_depend_on_previous_cell_values": True,
            "arbitrary_exact_postprocessing": True,
            "block_local_cells": True,
        },
        "proof_equations": {
            "zero_transcript": "ker(G_J) subset ker(q) => q in span_GF(2)(G_J bits)",
            "field_support": "alpha*u in span_E{g_j:j in J} => u in span_E{g_j:j in J}",
            "intersection": "dim_GF(2)(span_E(J) intersect GF(2)^k) <= |J|",
            "support_union": "sum_(j=0)^t C(S,j) 2^j >= 2^k",
            "binomial_relaxation": "k <= t log2(2eS/t)",
            "aggregate": "1 <= p log2(2e lambda/p)",
        },
        "registered_constants": {
            "non_embedding_coefficients": REGISTERED_NON_EMBEDDING_COEFFICIENTS,
            "global_hot_bits": GLOBAL_HOT_BITS,
            "logical_target_fraction": str(REGISTERED_P50_FRACTION),
            "four_q4_lane_traffic_fraction": str(
                FOUR_Q4_LANE_TRAFFIC_FRACTION
            ),
            "one_plane_proportional_expansion": str(
                ONE_PLANE_PROPORTIONAL_EXPANSION
            ),
            "all_q4_to_one_plane_expansion": str(
                ALL_Q4_STORAGE_TO_ONE_PLANE_EXPANSION
            ),
            "all_dfloat_to_one_plane_expansion": str(
                ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION
            ),
        },
        "small_field_control": small_control,
        "proportional_registered_blocks": blocks,
        "aggregate_block_local_cases": [q4_traffic, dfloat_traffic, dfloat_work],
        "integer_only_dfloat_traffic_rejection": rational_witness,
        "storage_required_at_four_lane_target": {
            "expansion": str(required),
            "one_plane_encoded_tib": str(required_tib),
        },
        "claim_boundary": {
            "fixed_extension_field_linear_cells_rejected": True,
            "deterministic_adaptive_addresses_rejected": True,
            "arbitrary_exact_postprocessing_rejected": True,
            "arbitrary_block_local_storage_allocation_rejected": True,
            "nonlinear_stored_cells_covered": False,
            "cross_matrix_mixed_cells_covered": False,
            "source_dependent_nonlinear_hot_advice_covered": False,
            "randomized_error_decoder_covered": False,
            "native_q4_bf16_fp32_semantics_covered": False,
            "surviving_runtime_candidate": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_EXTENSION_FIELD_LINEAR_SUMMARY_CONSTRUCTOR",
            "REJECT_ADAPTIVE_ZERO_TRANSCRIPT_ESCAPE",
            "REQUIRE_NONLINEAR_STORED_CELLS_OR_CROSS_BLOCK_STRUCTURE_FOR_NEXT_ROUTE",
            "KEEP_GENERAL_FINITE_WORD_TICKET_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
