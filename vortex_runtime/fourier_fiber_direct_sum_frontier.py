"""Finite Fourier-fiber audit for nonlinear global advice.

Fixing one advice value leaves a fiber of possible databases.  If every
binary linear query is answered with at most ``t`` adaptive raw-bit probes,
each restricted Walsh character has a real multilinear representation of
degree at most ``t``.  Walsh characters span every function on the fiber, so
the fiber cannot be larger than a Hamming ball of radius ``t``.

The resulting theorem genuinely charges cross-block probes and permits
arbitrary nonlinear global advice.  This module also records why the proof
does not resolve the much smaller rank-one query family.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb


REGISTERED_LAYERS = 126
REGISTERED_HIDDEN = 16_384
REGISTERED_KV = 1_024
REGISTERED_INTERMEDIATE = 53_248
REGISTERED_VOCAB = 128_256
REGISTERED_PARAMETERS = 405_849_243_648
GLOBAL_ADVICE_BITS = 8 * (8 * (1 << 30))
DFLOAT11_REPORTED_BITS = 8 * 551_220_000_000
REQUESTED_BLOCK_BITS = Fraction(DFLOAT11_REPORTED_BITS, 40)
REGISTERED_P50_FRACTION = Fraction(8, 675)
BINARY_RADIUS_WITNESS = Fraction(131, 500)
CERTIFIED_QUERIES = 32

DECISION = (
    "KEEP_FOURIER_FIBER_BOUND_FOR_ALL_LINEAR_TUPLES_"
    "REJECT_AS_GENERAL_RANK_ONE_TARGET_RESOLUTION"
)


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def binary_entropy_decimal(value: Fraction) -> Decimal:
    """Return binary entropy with high deterministic decimal precision."""

    if value <= 0 or value >= 1:
        raise ValueError("entropy input must lie strictly between zero and one")
    with localcontext() as context:
        context.prec = 80
        x = _decimal_fraction(value)
        one_minus_x = Decimal(1) - x
        log_two = Decimal(2).ln()
        return -(
            x * x.ln() + one_minus_x * one_minus_x.ln()
        ) / log_two


def hamming_ball_size(dimension: int, radius: int) -> int:
    if dimension < 0 or radius < 0 or radius > dimension:
        raise ValueError("invalid Hamming-ball parameters")
    return sum(comb(dimension, index) for index in range(radius + 1))


def all_linear_entropy_witness(
    *, data_bits: int, advice_bits: int, radius: Fraction
) -> dict[str, object]:
    """Certify a worst-case adaptive bit-probe floor for all parities."""

    if data_bits <= 0 or advice_bits < 0 or advice_bits >= data_bits:
        raise ValueError("require 0 <= advice_bits < data_bits")
    if radius <= 0 or radius >= Fraction(1, 2):
        raise ValueError("binary entropy witness must lie in (0, 1/2)")
    entropy = binary_entropy_decimal(radius)
    with localcontext() as context:
        context.prec = 80
        required_entropy = Decimal(1) - (
            Decimal(advice_bits) / Decimal(data_bits)
        )
        margin = required_entropy - entropy
    if margin <= 0:
        raise ValueError("radius does not certify the fiber-volume inequality")
    forced_probes = (radius.numerator * data_bits) // radius.denominator + 1
    return {
        "data_bits": data_bits,
        "global_nonlinear_advice_bits": advice_bits,
        "largest_fiber_log2_size_lower_bound": data_bits - advice_bits,
        "radius_witness": str(radius),
        "radius_witness_decimal": float(radius),
        "binary_entropy_at_witness": str(entropy),
        "required_normalized_fiber_log_size": str(required_entropy),
        "strict_entropy_margin": str(margin),
        "minimum_worst_case_raw_bit_probes": forced_probes,
        "minimum_probe_fraction": str(Fraction(forced_probes, data_bits)),
        "minimum_probe_percent": 100 * forced_probes / data_bits,
        "advice_may_be_arbitrary_nonlinear": True,
        "raw_probes_may_be_adaptive_and_cross_block": True,
        "query_computation_is_free": True,
        "answer_error": "ZERO",
    }


def registered_rank_one_query_log2_upper_bound() -> dict[str, int]:
    """Count a favorable binary rank-one pair description for every matrix."""

    per_layer = (
        2 * (REGISTERED_HIDDEN + REGISTERED_HIDDEN)
        + 2 * (REGISTERED_KV + REGISTERED_HIDDEN)
        + 3 * (REGISTERED_INTERMEDIATE + REGISTERED_HIDDEN)
    )
    embedding_and_head = 2 * (REGISTERED_VOCAB + REGISTERED_HIDDEN)
    total = REGISTERED_LAYERS * per_layer + embedding_and_head
    per_layer_parameters = (
        2 * REGISTERED_HIDDEN * REGISTERED_HIDDEN
        + 2 * REGISTERED_KV * REGISTERED_HIDDEN
        + 3 * REGISTERED_INTERMEDIATE * REGISTERED_HIDDEN
    )
    embedding_and_head_parameters = (
        2 * REGISTERED_VOCAB * REGISTERED_HIDDEN
    )
    parameters_from_shapes = (
        REGISTERED_LAYERS * per_layer_parameters
        + embedding_and_head_parameters
    )
    if parameters_from_shapes != REGISTERED_PARAMETERS:
        raise AssertionError("registered shape population drifted")
    return {
        "parameters_reconstructed_from_shapes": parameters_from_shapes,
        "registered_parameter_constant": REGISTERED_PARAMETERS,
        "per_layer_rank_one_pair_bits_upper_bound": per_layer,
        "layers": REGISTERED_LAYERS,
        "embedding_and_lm_head_pair_bits_upper_bound": embedding_and_head,
        "one_independent_tuple_log2_query_count_upper_bound": total,
        "thirty_two_tuple_log2_query_count_upper_bound": (
            CERTIFIED_QUERIES * total
        ),
    }


def character_dimension_method_ceiling(
    *, data_bits: int, query_log2_count_upper_bound: int
) -> dict[str, object]:
    """Give a finite ceiling on the restricted-character dimension method.

    The method can use at most one character column per query.  We find an
    integer ``t`` for which the elementary bound

        C(data_bits, t) >= (data_bits / t)^t >= 2^query_log2_count_upper_bound

    is certified by a power-of-two floor.  At that point the low-degree space
    is already large enough to hold every possible query character, so query
    cardinality alone cannot force a larger depth.  This is a limitation of
    the proof method, not an algorithmic upper bound.
    """

    if data_bits <= 0 or query_log2_count_upper_bound <= 0:
        raise ValueError("positive data and query dimensions required")
    candidates: list[tuple[int, int]] = []
    for power in range(1, data_bits.bit_length()):
        probes = (
            query_log2_count_upper_bound + power - 1
        ) // power
        if probes <= data_bits // (1 << power):
            candidates.append((probes, power))
    if not candidates:
        raise ValueError("no finite binomial-dimension witness found")
    probes, power = min(candidates)
    return {
        "data_bits": data_bits,
        "query_log2_count_upper_bound": query_log2_count_upper_bound,
        "method_ceiling_probe_count": probes,
        "power_of_two_ratio_witness": power,
        "certified_log2_binomial_lower_bound": power * probes,
        "method_ceiling_probe_fraction": str(Fraction(probes, data_bits)),
        "method_ceiling_probe_percent": 100 * probes / data_bits,
        "is_algorithmic_query_upper_bound": False,
        "is_target_lower_bound": False,
        "meaning": "QUERY_CARDINALITY_ONLY_CANNOT_FORCE_MORE",
    }


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _decision_tree_depth_binary(
    states: tuple[int, ...], labels: tuple[int, ...], dimension: int
) -> int:
    paired = tuple(sorted(zip(states, labels)))

    @lru_cache(maxsize=None)
    def solve(current: tuple[tuple[int, int], ...]) -> int:
        current_labels = {label for _, label in current}
        if len(current_labels) <= 1:
            return 0
        best = dimension + 1
        for coordinate in range(dimension):
            zero = tuple(
                pair for pair in current if not ((pair[0] >> coordinate) & 1)
            )
            one = tuple(
                pair for pair in current if (pair[0] >> coordinate) & 1
            )
            if not zero or not one:
                continue
            best = min(best, 1 + max(solve(zero), solve(one)))
        if best == dimension + 1:
            raise AssertionError("nonconstant labels were not separable")
        return best

    return solve(paired)


def exhaustive_binary_fiber_control(dimension: int = 3) -> dict[str, object]:
    """Exhaust all nonempty small fibers and every parity query."""

    if dimension <= 0 or dimension > 4:
        raise ValueError("control dimension must be in [1, 4]")
    universe = tuple(range(1 << dimension))
    fibers = 0
    query_checks = 0
    violations = 0
    maximum_depth = 0
    for mask in range(1, 1 << len(universe)):
        fiber = tuple(
            state for state in universe if (mask >> state) & 1
        )
        fibers += 1
        fiber_maximum = 0
        for query in universe:
            labels = tuple(_parity(query & state) for state in fiber)
            depth = _decision_tree_depth_binary(
                fiber, labels, dimension
            )
            fiber_maximum = max(fiber_maximum, depth)
            maximum_depth = max(maximum_depth, depth)
            query_checks += 1
        if len(fiber) > hamming_ball_size(dimension, fiber_maximum):
            violations += 1
    return {
        "dimension": dimension,
        "nonempty_fibers": fibers,
        "parity_query_checks": query_checks,
        "maximum_observed_decision_tree_depth": maximum_depth,
        "fiber_volume_violations": violations,
        "all_fibers_satisfy_bound": violations == 0,
    }


def walsh_row_orthogonality_control(dimension: int = 4) -> dict[str, object]:
    if dimension <= 0 or dimension > 10:
        raise ValueError("control dimension must be in [1, 10]")
    states = range(1 << dimension)
    queries = range(1 << dimension)
    checks = 0
    failures = 0
    for left, right in combinations(states, 2):
        correlation = sum(
            (-1) ** (_parity(query & left) ^ _parity(query & right))
            for query in queries
        )
        checks += 1
        failures += int(correlation != 0)
    return {
        "dimension": dimension,
        "distinct_row_pairs": checks,
        "orthogonality_failures": failures,
        "restricted_rows_are_independent": failures == 0,
    }


def derive_audit() -> dict[str, object]:
    all_linear = all_linear_entropy_witness(
        data_bits=REGISTERED_PARAMETERS,
        advice_bits=GLOBAL_ADVICE_BITS,
        radius=BINARY_RADIUS_WITNESS,
    )
    forced = all_linear["minimum_worst_case_raw_bit_probes"]
    assert isinstance(forced, int)
    query_counts = registered_rank_one_query_log2_upper_bound()
    one_log = query_counts[
        "one_independent_tuple_log2_query_count_upper_bound"
    ]
    thirty_two_log = query_counts[
        "thirty_two_tuple_log2_query_count_upper_bound"
    ]
    assert isinstance(one_log, int) and isinstance(thirty_two_log, int)
    one_ceiling = character_dimension_method_ceiling(
        data_bits=REGISTERED_PARAMETERS,
        query_log2_count_upper_bound=one_log,
    )
    thirty_two_ceiling = character_dimension_method_ceiling(
        data_bits=REGISTERED_PARAMETERS,
        query_log2_count_upper_bound=thirty_two_log,
    )
    method_32 = thirty_two_ceiling["method_ceiling_probe_count"]
    assert isinstance(method_32, int)
    registered_budget = REGISTERED_P50_FRACTION * REGISTERED_PARAMETERS
    return {
        "classification": "E0_FOURIER_FIBER_DIRECT_SUM_FRONTIER",
        "decision": DECISION,
        "theorem": {
            "fixed_advice_fiber": "F_a={x:R(x)=a}",
            "decision_tree_to_polynomial": "DEPTH_t_IMPLIES_REAL_DEGREE_AT_MOST_t",
            "character_span": "ALL_WALSH_CHARACTERS_RESTRICTED_TO_F_a_SPAN_R_TO_THE_F_a",
            "finite_inequality": "2^(D-r)<=|F_a|<=sum(j=0..t,C(D,j))",
            "cross_block_probes": "CHARGED_IN_THE_SAME_DEPTH_t",
            "advice": "ARBITRARY_NONLINEAR_AND_FREE_TO_READ",
            "computation": "FREE_FAVORABLE_GRANT",
            "randomization": "DETERMINISTIC_ZERO_ERROR",
        },
        "registered_all_linear_tuple": {
            **all_linear,
            "requested_dfloat_block_bits": str(REQUESTED_BLOCK_BITS),
            "forced_over_requested_dfloat_block": float(
                Fraction(forced, 1) / REQUESTED_BLOCK_BITS
            ),
            "reaches_requested_dfloat_block_under_one_bit_per_probe": (
                forced >= REQUESTED_BLOCK_BITS
            ),
            "registered_p50_coefficient_budget": str(registered_budget),
            "forced_over_registered_p50_coefficient_budget": float(
                Fraction(forced, 1) / registered_budget
            ),
            "independent_query_description_bits": REGISTERED_PARAMETERS,
            "query_description_over_requested_dfloat_block": float(
                Fraction(REGISTERED_PARAMETERS, 1) / REQUESTED_BLOCK_BITS
            ),
            "query_description_is_free_in_theorem_only": True,
        },
        "rank_one_character_dimension_boundary": {
            **query_counts,
            "one_tuple_method_ceiling": one_ceiling,
            "thirty_two_tuple_method_ceiling": thirty_two_ceiling,
            "requested_dfloat_block_over_32_tuple_method_ceiling": float(
                REQUESTED_BLOCK_BITS / method_32
            ),
            "registered_p50_budget_over_32_tuple_method_ceiling": float(
                registered_budget / method_32
            ),
            "rank_one_character_lower_bound_for_advice_fiber": "NOT_SUPPLIED",
            "general_rank_one_ticket_resolved": False,
        },
        "controls": {
            "exhaustive_binary_fibers": exhaustive_binary_fiber_control(),
            "walsh_orthogonality": walsh_row_orthogonality_control(),
        },
        "claim_boundary": {
            "all_linear_simultaneous_tuple_theorem": True,
            "charges_global_advice_synergy": True,
            "standard_transformer_queries_are_all_linear_tuples": False,
            "rank_one_query_family_theorem": False,
            "target_scale_rank_one_lower_bound": False,
            "universal_2_5_percent_feasibility": "NOT_ESTABLISHED",
            "universal_2_5_percent_impossibility": "NOT_PROVED",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
    }
