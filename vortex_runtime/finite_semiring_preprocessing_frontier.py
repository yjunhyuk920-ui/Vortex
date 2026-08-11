"""Finite audit of Williams' preprocessed finite-semiring MatVec graph.

Williams (2007) gives a genuine subquadratic query algorithm after
superquadratic preprocessing.  Its big-O query count can look compatible with
VORTEX if table representation and pointer payload are omitted.  This module
reconstructs the favorable physical information counts of the published
two-layer graph and separates finite-semiring algebra from native rounded
Transformer arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil, log2
import struct


REGISTERED_DIMENSION = 16_384
REGISTERED_PARAMETERS = 405_849_243_648
DFLOAT11_REPORTED_BYTES = 551_220_000_000
DFLOAT11_REPORTED_BITS = 8 * DFLOAT11_REPORTED_BYTES
GLOBAL_ADVICE_BYTES = 8 * (1 << 30)
REQUESTED_BLOCK_FRACTION = Fraction(1, 40)
REGISTERED_CERTIFIED_TOKENS = 32
MAXIMUM_THEOREM_PARAMETER_BLOCK_SYMBOLS = 14

DECISION = (
    "REJECT_WILLIAMS_FINITE_SEMIRING_GRAPH_AS_"
    "REFERENCE_EXACT_2_5_PERCENT_CORE"
)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _ceil_fraction(value: Fraction) -> int:
    return _ceil_div(value.numerator, value.denominator)


def williams_graph_layout(
    *, dimension: int, alphabet_bits: int, block_symbols: int
) -> dict[str, object]:
    """Count the minimum explicit pointer payload in the published graph.

    There are ceil(n/b) input groups.  Each group has K**b possible input
    patterns, and every such first-layer node stores one neighbor for every
    output group.  The output group is known from list position, so a neighbor
    identifier is favorably charged only b*log2(K) bits.  Object headers,
    addresses, counts, mutable state, and allocator rounding are all omitted.
    """

    if dimension <= 0 or alphabet_bits <= 0 or block_symbols <= 0:
        raise ValueError("dimension, alphabet_bits, and block_symbols must be positive")

    groups = _ceil_div(dimension, block_symbols)
    patterns_per_group = 1 << (alphabet_bits * block_symbols)
    pointer_payload_bits = alphabet_bits * block_symbols
    raw_matrix_bits = dimension * dimension * alphabet_bits

    first_layer_nodes = groups * patterns_per_group
    sidecar_neighbor_pointers = first_layer_nodes * groups
    sidecar_pointer_payload_bits = (
        sidecar_neighbor_pointers * pointer_payload_bits
    )

    query_neighbor_pointers = groups * groups
    query_pointer_payload_bits = query_neighbor_pointers * pointer_payload_bits

    return {
        "dimension": dimension,
        "alphabet_bits": alphabet_bits,
        "alphabet_cardinality": 1 << alphabet_bits,
        "block_symbols": block_symbols,
        "groups": groups,
        "patterns_per_group": patterns_per_group,
        "minimum_bits_per_neighbor_value": pointer_payload_bits,
        "raw_matrix_bits": raw_matrix_bits,
        "first_layer_nodes": first_layer_nodes,
        "sidecar_neighbor_pointers": sidecar_neighbor_pointers,
        "sidecar_pointer_payload_bits": sidecar_pointer_payload_bits,
        "sidecar_pointer_payload_bytes": _ceil_div(
            sidecar_pointer_payload_bits, 8
        ),
        "sidecar_pointer_payload_gib": (
            sidecar_pointer_payload_bits / (8 * (1 << 30))
        ),
        "sidecar_pointer_payload_fraction_of_raw": str(
            Fraction(sidecar_pointer_payload_bits, raw_matrix_bits)
        ),
        "query_neighbor_pointers": query_neighbor_pointers,
        "query_pointer_payload_bits": query_pointer_payload_bits,
        "query_pointer_payload_bytes": _ceil_div(
            query_pointer_payload_bits, 8
        ),
        "query_pointer_payload_fraction_of_raw": str(
            Fraction(query_pointer_payload_bits, raw_matrix_bits)
        ),
        "query_pointer_payload_percent_of_raw": (
            100 * query_pointer_payload_bits / raw_matrix_bits
        ),
        "omitted_favorable_costs": [
            "POINTER_OBJECT_HEADERS",
            "GLOBAL_NODE_ADDRESSES",
            "LIST_LENGTHS_AND_OFFSETS",
            "SECOND_LAYER_VALUES",
            "MUTABLE_COUNTER_STATE",
            "QUERY_INPUT_AND_OUTPUT",
            "SEMIRING_OPERATIONS",
            "CACHE_LINE_AND_PAGE_ROUNDING",
        ],
    }


def ideal_model_wide_substitution(
    *, alphabet_bits: int, block_symbols: int
) -> dict[str, object]:
    """Apply the graph's continuous favorable ratios to all 405B symbols.

    Ceilings and every non-pointer cost are removed.  This gives the graph a
    lower payload than its explicit finite layout and is therefore suitable
    only as a rejection screen.
    """

    if alphabet_bits <= 0 or block_symbols <= 0:
        raise ValueError("alphabet_bits and block_symbols must be positive")

    semantic_raw_bits = REGISTERED_PARAMETERS * alphabet_bits
    ideal_query_bits = Fraction(semantic_raw_bits, block_symbols)
    ideal_query_bytes = ideal_query_bits / 8
    ideal_sidecar_factor = Fraction(
        1 << (alphabet_bits * block_symbols), block_symbols
    )
    ideal_sidecar_bits = semantic_raw_bits * ideal_sidecar_factor
    target_bits = REQUESTED_BLOCK_FRACTION * DFLOAT11_REPORTED_BITS
    direct_no_reuse_block_bits = ideal_query_bits * REGISTERED_CERTIFIED_TOKENS

    return {
        "alphabet_bits": alphabet_bits,
        "block_symbols": block_symbols,
        "semantic_raw_bits": semantic_raw_bits,
        "ideal_query_payload_bits": str(ideal_query_bits),
        "ideal_query_payload_bytes": str(ideal_query_bytes),
        "ideal_query_fraction_of_semantic_raw": str(
            Fraction(1, block_symbols)
        ),
        "ideal_query_fraction_of_dfloat": str(
            ideal_query_bits / DFLOAT11_REPORTED_BITS
        ),
        "ideal_query_percent_of_dfloat": (
            100 * float(ideal_query_bits / DFLOAT11_REPORTED_BITS)
        ),
        "requested_block_bits": str(target_bits),
        "query_over_requested_block_budget": str(
            ideal_query_bits / target_bits
        ),
        "minimum_block_symbols_for_requested_dfloat_block": (
            minimum_block_symbols_for_dfloat_budget(alphabet_bits=alphabet_bits)
        ),
        "direct_32_query_no_reuse_payload_bits": str(
            direct_no_reuse_block_bits
        ),
        "direct_32_query_no_reuse_fraction_of_dfloat": str(
            direct_no_reuse_block_bits / DFLOAT11_REPORTED_BITS
        ),
        "direct_32_query_no_reuse_percent_of_dfloat": (
            100
            * float(direct_no_reuse_block_bits / DFLOAT11_REPORTED_BITS)
        ),
        "direct_32_query_no_reuse_over_requested_block_budget": str(
            direct_no_reuse_block_bits / target_bits
        ),
        "direct_32_query_no_reuse_fits_requested_block_budget": (
            direct_no_reuse_block_bits <= target_bits
        ),
        "ideal_sidecar_factor_over_semantic_raw": str(ideal_sidecar_factor),
        "ideal_sidecar_bits": str(ideal_sidecar_bits),
        "ideal_sidecar_gib": float(ideal_sidecar_bits / (8 * (1 << 30))),
        "sidecar_over_global_8_gib": float(
            ideal_sidecar_bits / (8 * GLOBAL_ADVICE_BYTES)
        ),
        "query_payload_fits_requested_block_budget": (
            ideal_query_bits <= target_bits
        ),
        "sidecar_fits_global_advice": (
            ideal_sidecar_bits <= 8 * GLOBAL_ADVICE_BYTES
        ),
        "direct_32_query_no_reuse_scope": (
            "DIRECT_CONSTRUCTOR_EXECUTION_NOT_A_GENERAL_MULTIQUERY_LOWER_BOUND"
        ),
    }


def minimum_block_symbols_for_fraction(fraction: Fraction) -> int:
    """Return the ideal b needed for a 1/b pointer-payload fraction."""

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must lie in (0, 1]")
    return _ceil_fraction(1 / fraction)


def minimum_block_symbols_for_dfloat_budget(*, alphabet_bits: int) -> int:
    """Return favorable b needed against the registered DFloat block budget."""

    if alphabet_bits <= 0:
        raise ValueError("alphabet_bits must be positive")
    semantic_raw_bits = REGISTERED_PARAMETERS * alphabet_bits
    target_bits = REQUESTED_BLOCK_FRACTION * DFLOAT11_REPORTED_BITS
    return _ceil_fraction(Fraction(semantic_raw_bits, 1) / target_bits)


def round_to_bfloat16(value: float) -> float:
    """Round a finite float32-representable value to BF16, ties to even."""

    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    exponent = bits & 0x7F80_0000
    if exponent == 0x7F80_0000:
        raise ValueError("non-finite values are outside this control")
    least_retained_bit = (bits >> 16) & 1
    rounded = bits + 0x7FFF + least_retained_bit
    bf16_bits = (rounded >> 16) & 0xFFFF
    return struct.unpack(">f", struct.pack(">I", bf16_bits << 16))[0]


def bf16_add(left: float, right: float) -> float:
    return round_to_bfloat16(left + right)


def round_to_float32(value: float) -> float:
    """Round a finite float64-host value to IEEE binary32."""

    rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    exponent = struct.unpack(">I", struct.pack(">f", rounded))[0] & 0x7F80_0000
    if exponent == 0x7F80_0000:
        raise ValueError("non-finite values are outside this control")
    return rounded


def fp32_add(left: float, right: float) -> float:
    return round_to_float32(left + right)


def native_rounding_semiring_counterexample() -> dict[str, object]:
    """Give an exact BF16 non-associativity witness."""

    one = 1.0
    half_ulp_at_one = 2.0**-8
    left_associated = bf16_add(bf16_add(one, half_ulp_at_one), half_ulp_at_one)
    right_associated = bf16_add(one, bf16_add(half_ulp_at_one, half_ulp_at_one))
    return {
        "a": one,
        "b": half_ulp_at_one,
        "c": half_ulp_at_one,
        "rounded_a_plus_b": bf16_add(one, half_ulp_at_one),
        "left_associated": left_associated,
        "right_associated": right_associated,
        "addition_is_associative": left_associated == right_associated,
        "finite_semiring_premise_holds_for_native_bf16_addition": False,
        "consequence": (
            "SEMIRING_REORDERING_DOES_NOT_PRESERVE_NATIVE_ROUNDED_REFERENCE"
        ),
    }


def native_fp32_accumulation_counterexample() -> dict[str, object]:
    """Show that a common FP32 accumulation carrier is not a semiring either."""

    a = 2.0**24
    b = 1.0
    c = -(2.0**24)
    left_associated = fp32_add(fp32_add(a, b), c)
    right_associated = fp32_add(a, fp32_add(b, c))
    return {
        "a": a,
        "b": b,
        "c": c,
        "left_associated": left_associated,
        "right_associated": right_associated,
        "addition_is_associative": left_associated == right_associated,
        "finite_semiring_premise_holds_for_native_fp32_addition": False,
        "consequence": (
            "SEMIRING_REORDERING_DOES_NOT_PRESERVE_NATIVE_FP32_ACCUMULATION"
        ),
    }


def derive_audit() -> dict[str, object]:
    maximum_b = MAXIMUM_THEOREM_PARAMETER_BLOCK_SYMBOLS
    layout_boolean = williams_graph_layout(
        dimension=REGISTERED_DIMENSION,
        alphabet_bits=1,
        block_symbols=maximum_b,
    )
    model_boolean = ideal_model_wide_substitution(
        alphabet_bits=1, block_symbols=maximum_b
    )
    model_q4 = ideal_model_wide_substitution(
        alphabet_bits=4, block_symbols=maximum_b
    )
    model_bf16 = ideal_model_wide_substitution(
        alphabet_bits=16, block_symbols=maximum_b
    )
    rounding = native_rounding_semiring_counterexample()
    fp32_rounding = native_fp32_accumulation_counterexample()

    return {
        "classification": "E0_FINITE_SEMIRING_PREPROCESSING_FRONTIER",
        "decision": DECISION,
        "published_constructor": {
            "source": "WILLIAMS_MATRIX_VECTOR_SUBQUADRATIC_PREPROCESSING",
            "query_steps": "O(n^2 / (epsilon*log n)^2)",
            "preprocessing_steps": "O(n^(2 + epsilon*log2(K)))",
            "machine": "POINTER_MACHINE_OR_LOG_N_WORD_RAM",
            "algebra": "FINITE_SEMIRING_WITH_CONSTANT_TIME_OPERATIONS",
            "block_symbols": "ceil(epsilon*log2(n))",
            "epsilon_range": "0 < epsilon < 1",
        },
        "registered_target": {
            "dimension": REGISTERED_DIMENSION,
            "log2_dimension": log2(REGISTERED_DIMENSION),
            "maximum_theorem_parameter_block_symbols": maximum_b,
            "physical_block_symbols_can_exceed_theorem_parameter": True,
            "requested_block_fraction": str(REQUESTED_BLOCK_FRACTION),
            "registered_certified_tokens": REGISTERED_CERTIFIED_TOKENS,
            "minimum_block_symbols_for_2_5_percent_of_semantic_raw": (
                minimum_block_symbols_for_fraction(REQUESTED_BLOCK_FRACTION)
            ),
            "minimum_block_symbols_for_requested_dfloat_block": {
                "one_bit_semantics": minimum_block_symbols_for_dfloat_budget(
                    alphabet_bits=1
                ),
                "q4_semantics": minimum_block_symbols_for_dfloat_budget(
                    alphabet_bits=4
                ),
                "bf16_symbol_semantics": minimum_block_symbols_for_dfloat_budget(
                    alphabet_bits=16
                ),
            },
            "dfloat11_reported_bytes": DFLOAT11_REPORTED_BYTES,
            "global_advice_bytes": GLOBAL_ADVICE_BYTES,
        },
        "largest_square_boolean_layout": layout_boolean,
        "model_wide_favorable_substitutions": {
            "one_bit_semantics": model_boolean,
            "q4_semantics": model_q4,
            "bf16_symbol_semantics": model_bf16,
        },
        "native_rounding_boundary": rounding,
        "native_fp32_accumulation_boundary": fp32_rounding,
        "claim_boundary": {
            "published_boolean_or_finite_semiring_algorithm": "VALID",
            "published_explicit_adjacency_catalog_charged": True,
            "compressed_or_implicit_edge_catalog": "NOT_COVERED",
            "physical_pointer_payload_charged": True,
            "q4_query_payload_fits_2_5_percent": model_q4[
                "query_payload_fits_requested_block_budget"
            ],
            "bf16_query_payload_fits_2_5_percent": model_bf16[
                "query_payload_fits_requested_block_budget"
            ],
            "boolean_sidecar_fits_8_gib": model_boolean[
                "sidecar_fits_global_advice"
            ],
            "native_bf16_is_published_semiring": False,
            "native_fp32_accumulation_is_published_semiring": False,
            "published_constructor_returns_full_matvec": True,
            "direct_constructor_is_scalar_rank_one_query": False,
            "published_multiquery_32_token_bound": False,
            "direct_32_query_no_reuse_boolean_fits_2_5_percent": model_boolean[
                "direct_32_query_no_reuse_fits_requested_block_budget"
            ],
            "general_nonlinear_numerical_rank_one_gap": "OPEN",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "primary_source": "https://people.csail.mit.edu/rrw/mat-vec3.pdf",
    }
