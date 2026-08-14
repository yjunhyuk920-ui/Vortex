"""Finite E0 audit of the general nonlinear rank-one probe gap.

The requested universal 2.5% statement quantifies over arbitrary checkpoint
data, arbitrary nonlinear global advice, and adaptive probes. This module
does not turn an asymptotic theorem into a target-scale lower bound. It
substitutes the registered 405B numbers into the strongest relevant theorem
premises and records exactly what they do and do not certify.
"""

from __future__ import annotations

from fractions import Fraction
from math import floor, isqrt, log2


REQUESTED_BLOCK_FRACTION = Fraction(1, 40)
REGISTERED_P50_FRACTION = Fraction(8, 675)
CERTIFIED_TOKENS_PER_BLOCK = 32
GLOBAL_ADVICE_BYTES = 8 * (1 << 30)
GLOBAL_ADVICE_BITS = 8 * GLOBAL_ADVICE_BYTES
LLAMA_405B_PARAMETERS = 405_849_243_648
Q4_BITS = 4 * LLAMA_405B_PARAMETERS
DFLOAT11_REPORTED_BYTES = 551_220_000_000
DFLOAT11_REPORTED_BITS = 8 * DFLOAT11_REPORTED_BYTES
NATURAL_WORD_BITS = 64
LARGEST_REGISTERED_SQUARE_DIMENSION = 16_384
KO_ENTROPY_CONSTANT = 10_000
KO_QUERY_EXPONENT_CONSTANT = 5

DECISION = (
    "UNIVERSAL_2_5_PERCENT_NOT_ESTABLISHED_"
    "GENERAL_NONLINEAR_RANK_ONE_GAP_REMAINS_OPEN"
)


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def nrs_help_bit_substitution(total_input_bits: int) -> dict[str, object]:
    """Substitute a database size into the Nisan--Rudich--Saks theorem.

    With `l` arbitrary help bits and `l + 1` disjoint parity instances of
    `k` bits, at least one instance retains decision-tree depth `k`. We
    partition the entire input as favorably as possible for this theorem.
    """

    if total_input_bits <= 0:
        raise ValueError("total_input_bits must be positive")
    instances = GLOBAL_ADVICE_BITS + 1
    parity_instance_bits = total_input_bits // instances
    covered_bits = parity_instance_bits * instances
    bit_probe_lower_bound = parity_instance_bits
    word_probe_lower_bound = _ceil_fraction(
        Fraction(bit_probe_lower_bound, NATURAL_WORD_BITS)
    )
    word_bytes_lower_bound = word_probe_lower_bound * (
        NATURAL_WORD_BITS // 8
    )
    requested_budget_bits = REQUESTED_BLOCK_FRACTION * total_input_bits
    return {
        "total_input_bits": total_input_bits,
        "arbitrary_help_bits": GLOBAL_ADVICE_BITS,
        "required_disjoint_instances": instances,
        "parity_bits_per_instance": parity_instance_bits,
        "covered_input_bits": covered_bits,
        "unused_input_bits": total_input_bits - covered_bits,
        "forced_bit_probes_for_one_instance": bit_probe_lower_bound,
        "forced_64_bit_word_probes_for_one_instance": (
            word_probe_lower_bound
        ),
        "forced_word_bytes_for_one_instance": word_bytes_lower_bound,
        "requested_2_5_percent_budget_bits": str(requested_budget_bits),
        "requested_budget_to_forced_bit_probe_ratio": (
            float(requested_budget_bits / bit_probe_lower_bound)
            if bit_probe_lower_bound
            else None
        ),
        "covers_arbitrary_nonlinear_help_bits": True,
        "covers_cross_instance_input_probes": True,
        "target_scale_rejection": False,
    }


def rank_one_query_log2_upper_bound(
    dimension: int, *, field_symbol_bits: int = 1
) -> int:
    """Return a favorable upper bound on log2 of the rank-one query set.

    A pair of length-`dimension` vectors over an alphabet with at most
    `2**field_symbol_bits` values has at most
    `2**(2 * dimension * field_symbol_bits)` possible pairs. Counting
    duplicate outer products only makes the true query set smaller.
    """

    if dimension <= 0 or field_symbol_bits <= 0:
        raise ValueError("dimension and field_symbol_bits must be positive")
    return 2 * dimension * field_symbol_bits


def ko_theorem_substitution(
    *,
    data_bits: int,
    query_collection_log2_upper_bound: int,
    target_probe_bits: int,
) -> dict[str, object]:
    """Evaluate the explicit premises of Ko's 2025 Theorem 1.3.

    The theorem requires `log |S| >= 10^4 * t * log s` and an asymptotic
    query-count condition `m = omega(s * 2^(5t))`. The latter has no finite
    threshold, so this function never promotes it to a concrete certificate.
    """

    if data_bits <= 0:
        raise ValueError("data_bits must be positive")
    if query_collection_log2_upper_bound <= 0:
        raise ValueError("query collection must be nonempty")
    if target_probe_bits <= 0:
        raise ValueError("target_probe_bits must be positive")
    systematic_cells = data_bits + GLOBAL_ADVICE_BITS
    log2_space = log2(systematic_cells)
    maximum_t_from_collection_premise = floor(
        query_collection_log2_upper_bound
        / (KO_ENTROPY_CONSTANT * log2_space)
    )
    entropy_rhs_for_target = (
        KO_ENTROPY_CONSTANT * target_probe_bits * log2_space
    )
    minimum_log2_m_without_omega = (
        log2_space + KO_QUERY_EXPONENT_CONSTANT * target_probe_bits
    )
    return {
        "data_bits": data_bits,
        "systematic_one_bit_cells_including_global_advice": (
            systematic_cells
        ),
        "log2_systematic_cells": log2_space,
        "query_collection_log2_upper_bound": (
            query_collection_log2_upper_bound
        ),
        "target_probe_bits": target_probe_bits,
        "maximum_integer_t_allowed_by_collection_entropy_premise": (
            maximum_t_from_collection_premise
        ),
        "entropy_premise_rhs_at_target_t": entropy_rhs_for_target,
        "entropy_premise_holds_at_target_t": (
            query_collection_log2_upper_bound >= entropy_rhs_for_target
        ),
        "minimum_log2_query_count_before_asymptotic_omega": (
            minimum_log2_m_without_omega
        ),
        "query_universe_can_meet_even_non_omega_count_at_target_t": (
            query_collection_log2_upper_bound
            >= minimum_log2_m_without_omega
        ),
        "finite_omega_threshold_supplied_by_theorem": False,
        "target_scale_rejection": False,
    }


def ckl_single_matrix_boundary() -> dict[str, object]:
    """Record why the single-matrix succinct uMv theorem is inapplicable."""

    dimension = LARGEST_REGISTERED_SQUARE_DIMENSION
    matrix_bits = dimension * dimension
    theorem_redundancy_ceiling = matrix_bits // 4
    return {
        "matrix_dimension": dimension,
        "matrix_bits": matrix_bits,
        "published_redundancy_range_lower_bits": dimension,
        "published_redundancy_range_upper_bits": (
            theorem_redundancy_ceiling
        ),
        "global_advice_bits": GLOBAL_ADVICE_BITS,
        "global_advice_over_range_ceiling": str(
            Fraction(GLOBAL_ADVICE_BITS, theorem_redundancy_ceiling)
        ),
        "global_advice_over_range_ceiling_decimal": (
            GLOBAL_ADVICE_BITS / theorem_redundancy_ceiling
        ),
        "single_matrix_theorem_applicable_with_global_advice": False,
        "published_model": (
            "SYSTEMATIC_GF2_EXACT_RANK_ONE_BIT_OUTPUT_ADAPTIVE_PROBES"
        ),
        "published_tradeoff": "t*r = Omega(n^3/log n)",
        "model_wide_direct_sum_supplied": False,
        "target_scale_rejection": False,
    }


def derive_audit() -> dict[str, object]:
    requested_block_bits = (
        REQUESTED_BLOCK_FRACTION * DFLOAT11_REPORTED_BITS
    )
    registered_p50_block_bits = (
        REGISTERED_P50_FRACTION * DFLOAT11_REPORTED_BITS
    )
    requested_per_token_bits = (
        requested_block_bits / CERTIFIED_TOKENS_PER_BLOCK
    )
    registered_per_token_bits = (
        registered_p50_block_bits / CERTIFIED_TOKENS_PER_BLOCK
    )

    optimistic_dimension = isqrt(DFLOAT11_REPORTED_BITS)
    optimistic_rank_one_log2 = rank_one_query_log2_upper_bound(
        optimistic_dimension
    )
    optimistic_ko = ko_theorem_substitution(
        data_bits=DFLOAT11_REPORTED_BITS,
        query_collection_log2_upper_bound=optimistic_rank_one_log2,
        target_probe_bits=int(requested_block_bits),
    )

    largest_dimension = LARGEST_REGISTERED_SQUARE_DIMENSION
    largest_matrix_bits = largest_dimension * largest_dimension
    largest_rank_one_log2 = rank_one_query_log2_upper_bound(
        largest_dimension
    )
    largest_matrix_ko = ko_theorem_substitution(
        data_bits=largest_matrix_bits,
        query_collection_log2_upper_bound=largest_rank_one_log2,
        target_probe_bits=largest_matrix_bits // 40,
    )

    all_linear_ko = ko_theorem_substitution(
        data_bits=DFLOAT11_REPORTED_BITS,
        query_collection_log2_upper_bound=DFLOAT11_REPORTED_BITS,
        target_probe_bits=int(requested_block_bits),
    )
    all_linear_t_ceiling = all_linear_ko[
        "maximum_integer_t_allowed_by_collection_entropy_premise"
    ]
    assert isinstance(all_linear_t_ceiling, int)

    return {
        "classification": "E0_GENERAL_FINITE_WORD_RANK_ONE_PROBE_GAP",
        "decision": DECISION,
        "frozen_model": {
            "cold_input": "UNCHANGED_BOUNDED_WORD_CHECKPOINT",
            "global_advice_bytes": GLOBAL_ADVICE_BYTES,
            "global_advice_read_charge": "FREE_FAVORABLE_GRANT",
            "preprocessing": "ARBITRARY_CHECKPOINT_ONLY",
            "query_probes": "ADAPTIVE_AND_DATA_DEPENDENT",
            "query_computation": "FREE_IN_LOWER_BOUND_MODEL",
            "required_answer": "ZERO_ERROR_EXACT_RANK_ONE_SCALAR",
            "metadata_kv_state_scheduler_reprop_cost": (
                "ZERO_FAVORABLE_GRANT"
            ),
            "forbidden": [
                "UNBOUNDED_WORDS",
                "FUTURE_TRACE_ADVICE",
                "APPROXIMATE_ANSWERS",
                "EXHAUSTIVE_QUERY_ANSWER_TABLE",
                "EXTERNAL_PROVER",
            ],
        },
        "registered_target": {
            "llama_405b_parameters": LLAMA_405B_PARAMETERS,
            "q4_bits": Q4_BITS,
            "dfloat11_reported_bytes": DFLOAT11_REPORTED_BYTES,
            "dfloat11_reported_bits": DFLOAT11_REPORTED_BITS,
            "requested_block_fraction": str(REQUESTED_BLOCK_FRACTION),
            "registered_p50_block_fraction": str(
                REGISTERED_P50_FRACTION
            ),
            "certified_tokens_per_block": CERTIFIED_TOKENS_PER_BLOCK,
            "requested_block_bits": str(requested_block_bits),
            "requested_block_bytes": str(requested_block_bits / 8),
            "requested_per_certified_token_bits": str(
                requested_per_token_bits
            ),
            "requested_per_certified_token_bytes": str(
                requested_per_token_bits / 8
            ),
            "registered_p50_block_bits": str(
                registered_p50_block_bits
            ),
            "registered_p50_per_certified_token_bits": str(
                registered_per_token_bits
            ),
            "global_advice_fraction_of_dfloat_bits": str(
                Fraction(GLOBAL_ADVICE_BITS, DFLOAT11_REPORTED_BITS)
            ),
            "global_advice_fraction_of_dfloat_bits_decimal": (
                GLOBAL_ADVICE_BITS / DFLOAT11_REPORTED_BITS
            ),
        },
        "nrs_help_bits": {
            "theorem_scope": (
                "ARBITRARY_NONLINEAR_HELP_BITS_AND_CROSS_INSTANCE_PROBES"
            ),
            "one_bit_per_parameter": nrs_help_bit_substitution(
                LLAMA_405B_PARAMETERS
            ),
            "q4_physical_bits_favorable_ceiling": (
                nrs_help_bit_substitution(Q4_BITS)
            ),
            "dfloat11_reported_physical_bits_favorable_ceiling": (
                nrs_help_bit_substitution(DFLOAT11_REPORTED_BITS)
            ),
            "q4_native_query_caveat": (
                "NUMERICAL_RANK_ONE_DOES_NOT_EXPOSE_FOUR_INDEPENDENT_PARITIES"
            ),
            "dfloat_independence_caveat": (
                "LOSSLESS_CODE_BITS_ARE_NOT_PROVEN_INDEPENDENT_PARITY_INPUTS"
            ),
            "conclusion": (
                "VALID_GENERAL_HELP_BIT_BOUND_BUT_BILLIONS_OF_TIMES_"
                "TOO_SMALL_FOR_TARGET_REJECTION"
            ),
        },
        "ko_2025": {
            "theorem": (
                "log|S|>=10000*t*log(s) AND "
                "m=omega(s*2^(5t))"
            ),
            "probe_accounting_mismatch": (
                "KO_CHARGES_PROBES_TO_ALL_PREPROCESSED_S_CELLS"
            ),
            "free_global_advice_checkpoint_probe_bound": (
                "NOT SUPPLIED"
            ),
            "largest_registered_square_rank_one": largest_matrix_ko,
            "illegal_optimistic_global_bit_reshape_rank_one": optimistic_ko,
            "all_linear_queries_not_rank_one": all_linear_ko,
            "all_linear_entropy_premise_t_ceiling_bits": (
                all_linear_t_ceiling
            ),
            "requested_budget_over_all_linear_t_ceiling": (
                float(requested_block_bits / all_linear_t_ceiling)
            ),
            "conclusion": (
                "NO_FINITE_TARGET_CERTIFICATE_AND_QUERY_FAMILY_MISMATCH"
            ),
        },
        "ckl_2018": ckl_single_matrix_boundary(),
        "reduction_boundary": {
            "systematic_nonlinear_common_bits_problem": (
                "THE_REMAINING_ABSTRACT_INTERFACE"
            ),
            "systematic_linear_special_case": (
                "CLOSED_BUT_TARGET_SCALE_BOUND_TOO_WEAK"
            ),
            "arbitrary_nonlinear_rank_one_advice": "OPEN",
            "native_q4_bf16_equivalence_from_gf2": "NOT_AUTOMATIC",
            "global_direct_sum_across_matrices_and_bit_planes": (
                "NOT SUPPLIED"
            ),
        },
        "sound_theorem_that_is_currently_available": {
            "statement": (
                "IF a sound certificate closes within the byte budget, "
                "the emitted token is exact; otherwise refinement must continue"
            ),
            "universal_exactness": "CONDITIONAL_ON_TERMINAL_STATE_EXACTIZATION",
            "universal_2_5_percent_performance": "DOES_NOT_FOLLOW",
        },
        "claim_boundary": {
            "general_nonlinear_rank_one_constructor": "NOT FOUND",
            "general_nonlinear_rank_one_target_lower_bound": "NOT PROVED",
            "universal_2_5_percent_guarantee": "NOT ESTABLISHED",
            "universal_2_5_percent_impossibility": "NOT PROVED",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "primary_sources": {
            "nrs_help_bits": (
                "https://www.brics.dk/NS/94/3/BRICS-NS-94-3.pdf"
            ),
            "ko_2025": (
                "https://eccc.weizmann.ac.il/report/2025/155/download"
            ),
            "ckl_2018": (
                "https://cs.au.dk/~larsen/papers/BooleanMatrixVectorLB.pdf"
            ),
            "rigidity_boundary": "https://arxiv.org/abs/1910.11921",
        },
    }
