"""Finite audit of global-advice division and CKL direct-sum shortcuts.

The single-matrix CKL rank-one theorem cannot be applied model-wide by
dividing global nonlinear advice by a matrix or tile count.  This module gives
an exact XOR-synergy counterexample and also evaluates an intentionally invalid
favorable tile sum, so neither failure is hidden behind asymptotic notation.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import ceil, log2


REGISTERED_LAYERS = 126
REGISTERED_HIDDEN = 16_384
REGISTERED_INTERMEDIATE = 53_248
REGISTERED_NON_EMBEDDING_PARAMETERS = 403_747_897_344
GLOBAL_ADVICE_BITS = 8 * (8 * (1 << 30))
DFLOAT11_REPORTED_BITS = 8 * 551_220_000_000
REQUESTED_BLOCK_BITS = Fraction(DFLOAT11_REPORTED_BITS, 40)
CKL_PROOF_DENOMINATOR = 8_192

DECISION = (
    "REJECT_NAIVE_GLOBAL_ADVICE_DIVISION_AND_CKL_TILE_SUM_"
    "AS_TARGET_BOUND"
)


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def _minimum_square_dimension_for_average_advice(
    *, data_bits: int, advice_bits: int
) -> int:
    if data_bits <= 0 or advice_bits < 0:
        raise ValueError("data_bits must be positive and advice nonnegative")
    dimension = 1
    while dimension * dimension <= data_bits:
        tile_count = data_bits // (dimension * dimension)
        if Fraction(advice_bits, tile_count) >= dimension:
            return dimension
        dimension += 1
    raise ValueError("no complete square tile satisfies the lower endpoint")


def separate_tile_injectivity_probe_floor(
    *, tile_bits: int, local_advice_cap_bits: int
) -> int:
    """Return the elementary worst-case raw-probe floor for one tile.

    If fewer than ``tile_bits`` advice bits describe an arbitrary binary tile,
    two tiles share an advice string.  A basis rank-one query selects a bit on
    which those tiles differ, so an exact systematic query algorithm cannot
    answer every query with zero raw probes.  This proves only one separately
    chosen hard query for one tile.  It is not a simultaneous direct-sum bound.
    """

    if tile_bits <= 0 or local_advice_cap_bits < 0:
        raise ValueError("tile_bits must be positive and advice nonnegative")
    return int(local_advice_cap_bits < tile_bits)


def registered_full_hidden_square_tiles() -> dict[str, int]:
    """Count complete 16,384-square tiles in the registered dense layers."""

    expansion_tiles = REGISTERED_INTERMEDIATE // REGISTERED_HIDDEN
    per_layer = 2 + 3 * expansion_tiles
    return {
        "q_and_o_tiles_per_layer": 2,
        "full_tiles_per_gate_up_down_matrix": expansion_tiles,
        "gate_up_down_tiles_per_layer": 3 * expansion_tiles,
        "full_tiles_per_layer": per_layer,
        "layers": REGISTERED_LAYERS,
        "full_tiles_total": per_layer * REGISTERED_LAYERS,
    }


def xor_advice(words: tuple[int, ...], *, symbol_bits: int) -> int:
    if len(words) < 2 or symbol_bits <= 0:
        raise ValueError("at least two words and positive symbol_bits required")
    limit = 1 << symbol_bits
    if any(word < 0 or word >= limit for word in words):
        raise ValueError("word is outside the declared finite alphabet")
    answer = 0
    for word in words:
        answer ^= word
    return answer


def recover_from_xor_advice(
    advice: int,
    known_words: tuple[int, ...],
    *,
    symbol_bits: int,
) -> int:
    return xor_advice((advice, *known_words), symbol_bits=symbol_bits)


def exhaustive_xor_synergy_control(
    *, instances: int = 3, symbol_bits: int = 3
) -> dict[str, object]:
    """Exhaustively verify the conditional-information XOR counterexample."""

    if instances < 2 or symbol_bits <= 0:
        raise ValueError("instances >= 2 and symbol_bits > 0 required")
    alphabet = range(1 << symbol_bits)
    advice_counts = [0] * (1 << symbol_bits)
    cases = 0
    recovery_checks = 0
    for words in product(alphabet, repeat=instances):
        words_tuple = tuple(words)
        advice = xor_advice(words_tuple, symbol_bits=symbol_bits)
        advice_counts[advice] += 1
        cases += 1
        for target in range(instances):
            known = words_tuple[:target] + words_tuple[target + 1 :]
            recovered = recover_from_xor_advice(
                advice, known, symbol_bits=symbol_bits
            )
            if recovered != words_tuple[target]:
                raise AssertionError("XOR conditional reconstruction failed")
            recovery_checks += 1

    uniform_count = (1 << symbol_bits) ** (instances - 1)
    return {
        "instances": instances,
        "symbol_bits_per_instance": symbol_bits,
        "cases": cases,
        "recovery_checks": recovery_checks,
        "advice_histogram": advice_counts,
        "advice_is_uniform": all(
            count == uniform_count for count in advice_counts
        ),
        "advice_entropy_bits": symbol_bits,
        "conditional_information_per_instance_bits": symbol_bits,
        "sum_conditional_information_bits": instances * symbol_bits,
        "conditional_information_over_advice_entropy": instances,
        "every_instance_recoverable_given_advice_and_all_others": True,
    }


def favorable_ckl_tile_sum_screen() -> dict[str, object]:
    """Evaluate an invalidly favorable per-tile CKL direct-sum substitution.

    The screen divides advice evenly, grants independently selectable hard
    queries for every tile, sums all per-tile probes, and packs probe bits
    perfectly.  No cited theorem supplies those direct-sum steps.  The point
    is that even this overgrant does not yield the registered finite bound
    when only the elementary separate-tile injectivity consequence is
    retained.  That one-probe consequence does not come from an asymptotic
    CKL constant and cannot legally be summed without a direct-sum theorem.
    """

    data_bits = REGISTERED_NON_EMBEDDING_PARAMETERS
    advice_rate = Fraction(GLOBAL_ADVICE_BITS, data_bits)
    minimum_dimension = _minimum_square_dimension_for_average_advice(
        data_bits=data_bits,
        advice_bits=GLOBAL_ADVICE_BITS,
    )
    tile_bits = minimum_dimension * minimum_dimension
    favorable_tile_count = data_bits // tile_bits
    average_redundancy = Fraction(GLOBAL_ADVICE_BITS, favorable_tile_count)
    granted_local_advice_cap = _ceil_fraction(average_redundancy)
    separate_tile_probe_floor = separate_tile_injectivity_probe_floor(
        tile_bits=tile_bits,
        local_advice_cap_bits=granted_local_advice_cap,
    )
    invalid_summed_probe_floor = (
        favorable_tile_count * separate_tile_probe_floor
    )

    leading_monomial_per_tile = (
        minimum_dimension**3
        / (float(average_redundancy) * log2(minimum_dimension))
    )
    counterfactual_proof_coefficient_per_tile = (
        leading_monomial_per_tile / CKL_PROOF_DENOMINATOR
    )

    return {
        "binary_data_bits": data_bits,
        "global_advice_bits": GLOBAL_ADVICE_BITS,
        "global_advice_rate": str(advice_rate),
        "global_advice_rate_percent": 100 * float(advice_rate),
        "minimum_dimension_with_average_r_at_least_n": minimum_dimension,
        "tile_bits": tile_bits,
        "favorable_global_reshape_tile_count": favorable_tile_count,
        "unused_binary_bits": data_bits - favorable_tile_count * tile_bits,
        "invalid_even_average_redundancy_bits": str(average_redundancy),
        "granted_ceiling_local_advice_cap_bits": granted_local_advice_cap,
        "ckl_theorem_statement_upper_range_n_squared_over_4": tile_bits // 4,
        "ckl_theorem_statement_lower_range_n": minimum_dimension,
        "average_redundancy_inside_theorem_statement_range": (
            minimum_dimension
            <= average_redundancy
            <= Fraction(tile_bits, 4)
        ),
        "ckl_displayed_proof_regime_n_squared_over_64": str(
            Fraction(tile_bits, 64)
        ),
        "average_redundancy_inside_displayed_proof_regime": (
            average_redundancy <= Fraction(tile_bits, 64)
        ),
        "leading_monomial_per_tile_if_hidden_constant_were_one": (
            leading_monomial_per_tile
        ),
        "leading_monomial_uses_exact_invalid_average": True,
        "counterfactual_displayed_proof_coefficient_outside_regime": (
            counterfactual_proof_coefficient_per_tile
        ),
        "separate_tile_injectivity_probe_floor": separate_tile_probe_floor,
        "separate_tile_floor_source": (
            "FINITE_PIGEONHOLE_PLUS_BASIS_RANK_ONE_QUERY"
        ),
        "simultaneous_hard_query_composition_proved": False,
        "invalid_sum_of_separate_tile_probe_floors": (
            invalid_summed_probe_floor
        ),
        "requested_dfloat_block_bits": str(REQUESTED_BLOCK_BITS),
        "invalid_floor_sum_fraction_of_requested_block": str(
            Fraction(invalid_summed_probe_floor, 1) / REQUESTED_BLOCK_BITS
        ),
        "invalid_floor_sum_percent_of_requested_block": (
            100
            * float(
                Fraction(invalid_summed_probe_floor, 1)
                / REQUESTED_BLOCK_BITS
            )
        ),
        "requested_block_over_invalid_floor_sum": (
            float(REQUESTED_BLOCK_BITS / invalid_summed_probe_floor)
        ),
        "even_invalid_sum_of_separate_floors_reaches_target": (
            invalid_summed_probe_floor >= REQUESTED_BLOCK_BITS
        ),
        "invalid_grants": [
            "DIVIDE_NONLINEAR_GLOBAL_ADVICE_EVENLY",
            "ROUND_EACH_TILE_ADVICE_CAP_UP_INDEPENDENTLY",
            "RESHAPE_ALL_COEFFICIENTS_INTO_INDEPENDENT_SQUARE_TILES",
            "CHOOSE_EACH_TILE_HARD_QUERY_INDEPENDENTLY",
            "SUM_SEPARATELY_CHOSEN_SINGLE_TILE_WORST_CASES",
            "PACK_ALL_FORCED_BITS_WITH_ZERO_ADDRESS_OVERHEAD",
            "TREAT_GF2_QUERY_AS_NATIVE_NUMERICAL_QUERY",
        ],
    }


def registered_square_tile_average_screen() -> dict[str, object]:
    layout = registered_full_hidden_square_tiles()
    tile_count = layout["full_tiles_total"]
    tile_bits = REGISTERED_HIDDEN * REGISTERED_HIDDEN
    average_redundancy = Fraction(GLOBAL_ADVICE_BITS, tile_count)
    leading_t = Fraction(REGISTERED_HIDDEN**3, 1) / (
        average_redundancy * 14
    )
    return {
        **layout,
        "tile_bits": tile_bits,
        "invalid_even_average_redundancy_bits": str(average_redundancy),
        "invalid_even_average_fraction_of_tile": str(
            average_redundancy / tile_bits
        ),
        "invalid_even_average_percent_of_tile": (
            100 * float(average_redundancy / tile_bits)
        ),
        "ckl_theorem_statement_upper_range_n_squared_over_4": tile_bits // 4,
        "average_inside_theorem_statement_upper_range": (
            average_redundancy <= Fraction(tile_bits, 4)
        ),
        "ckl_displayed_proof_regime_n_squared_over_64": tile_bits // 64,
        "average_inside_displayed_proof_regime": (
            average_redundancy <= Fraction(tile_bits, 64)
        ),
        "leading_monomial_t_if_hidden_constant_were_one": str(leading_t),
        "leading_monomial_t_decimal": float(leading_t),
        "hidden_asymptotic_constant_set_to_one_counterfactually": True,
        "leading_monomial_is_certified_finite_lower_bound": False,
        "sum_of_leading_monomials_across_tiles": str(leading_t * tile_count),
        "invalid_probe_to_payload_grant": (
            "ONE_PROBE_EQUALS_ONE_PERFECTLY_PACKED_PHYSICAL_BIT"
        ),
        "requested_block_over_summed_leading_monomials": (
            float(REQUESTED_BLOCK_BITS / (leading_t * tile_count))
        ),
        "direct_sum_theorem_supplied": False,
    }


def derive_audit() -> dict[str, object]:
    synergy = exhaustive_xor_synergy_control()
    favorable_sum = favorable_ckl_tile_sum_screen()
    registered_tiles = registered_square_tile_average_screen()
    return {
        "classification": "E0_GLOBAL_ADVICE_SYNERGY_FRONTIER",
        "decision": DECISION,
        "registered_square_tiles": registered_tiles,
        "xor_synergy_control": synergy,
        "invalid_favorable_ckl_tile_sum": favorable_sum,
        "claim_boundary": {
            "naive_advice_division_is_valid": False,
            "reason": "CONDITIONAL_INFORMATION_CAN_BE_SYNERGISTIC",
            "single_matrix_ckl_theorem_is_valid": True,
            "published_cross_matrix_direct_sum": False,
            "published_cross_matrix_probe_charge_for_synergy": False,
            "target_scale_lower_bound_established": False,
            "general_nonlinear_rank_one_gap": "OPEN",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "primary_source": (
            "https://cs.au.dk/~larsen/papers/BooleanMatrixVectorLB.pdf"
        ),
    }
