"""Finite activity lower bound for fixed linear functional decoders.

Let a cold dictionary have atom matrix ``G`` and let a fixed linear decoder
``H`` satisfy ``G H = I``.  The exact query coefficients are ``H q``.  For an
independently sampled tuple of binary rank-one masks, every nonzero row of H is
active with probability at least one quarter.  Across a 32-query batch, almost
all nonzero rows are active at least once.

The theorem is deliberately scoped to a fixed linear decoder and independently
selectable rank-one tuples.  It does not cover nonlinear syndrome decoding or
prove that a causal Transformer trace realizes the hard tuple.
"""

from __future__ import annotations

from fractions import Fraction


REGISTERED_NON_EMBEDDING_BITS = 403_747_897_344
HOT_BITS = 8 * 8 * (1 << 30)
BATCH_QUERIES = 32
WORD_BITS = 64
FAVORABLE_LINK_BYTES_PER_SECOND = 32_000_000_000
DFLOAT11_REPORTED_BYTES = 551_220_000_000
REQUESTED_BLOCK_BYTES = Fraction(DFLOAT11_REPORTED_BYTES, 40)

DECISION = (
    "REJECT_FIXED_LINEAR_FUNCTIONAL_DECODER_FOR_32_INDEPENDENT_"
    "RANK_ONE_TUPLES_KEEP_NONLINEAR_SYNDROME_DECODER_OPEN"
)


def bilinear_one_probability(rank: int) -> Fraction:
    """Exact ``Pr[r.T A u = 1]`` for uniform binary r,u and rank(A)."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank == 0:
        return Fraction(0)
    return Fraction((1 << rank) - 1, 1 << (rank + 1))


def minimum_nonzero_decoder_row_activity() -> Fraction:
    """Minimum activity probability of one nonzero global decoder row."""

    return bilinear_one_probability(1)


def batch_union_activity_lower(batch_queries: int) -> Fraction:
    """Minimum probability a nonzero decoder row fires at least once."""

    if batch_queries <= 0:
        raise ValueError("batch query count must be positive")
    inactive_upper = (Fraction(3, 4)) ** batch_queries
    return Fraction(1) - inactive_upper


def ceil_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("cannot ceil a negative resource")
    return (value.numerator + value.denominator - 1) // value.denominator


def fixed_linear_decoder_cold_floor(
    *,
    data_bits: int = REGISTERED_NON_EMBEDDING_BITS,
    hot_bits: int = HOT_BITS,
    batch_queries: int = BATCH_QUERIES,
    word_bits: int = WORD_BITS,
) -> dict[str, object]:
    """Worst-case cold payload under maximally favorable bit packing.

    If ``G H = I_D``, then ``rank(H) >= D``, so H has at least D nonzero
    rows.  Put as many of their one-bit stored cells in hot memory as possible.
    Every remaining nonzero row fires in a batch with probability at least
    ``1-(3/4)^K``.  Averaging supplies one batch whose union is at least the
    ceiling of that expectation.
    """

    if data_bits <= 0 or hot_bits < 0 or hot_bits >= data_bits:
        raise ValueError("require 0 <= hot bits < data bits")
    if batch_queries <= 0 or word_bits <= 0 or word_bits % 8:
        raise ValueError("invalid batch or word size")
    cold_nonzero_rows = data_bits - hot_bits
    union_probability = batch_union_activity_lower(batch_queries)
    expected_active = cold_nonzero_rows * union_probability
    forced_active_bits = ceil_fraction(expected_active)
    forced_words = (forced_active_bits + word_bits - 1) // word_bits
    forced_bytes = forced_words * (word_bits // 8)
    return {
        "data_dimension_bits": data_bits,
        "minimum_nonzero_decoder_rows": data_bits,
        "hot_one_bit_cells_granted": hot_bits,
        "minimum_cold_nonzero_decoder_rows": cold_nonzero_rows,
        "batch_queries": batch_queries,
        "minimum_single_query_row_activity": str(
            minimum_nonzero_decoder_row_activity()
        ),
        "batch_union_activity_lower": str(union_probability),
        "batch_union_activity_lower_decimal": float(union_probability),
        "expected_cold_active_cells_lower": str(expected_active),
        "forced_worst_case_cold_active_cell_bits": forced_active_bits,
        "favorable_word_bits": word_bits,
        "forced_worst_case_cold_words": forced_words,
        "forced_worst_case_cold_bytes": forced_bytes,
    }


def derive_audit() -> dict[str, object]:
    floor = fixed_linear_decoder_cold_floor()
    cold_bytes = int(floor["forced_worst_case_cold_bytes"])
    block_seconds = Fraction(cold_bytes, FAVORABLE_LINK_BYTES_PER_SECOND)
    per_token_seconds = block_seconds / BATCH_QUERIES
    return {
        "classification": "E0_FIXED_LINEAR_FUNCTIONAL_DECODER_ACTIVITY_BOUND",
        "theorem": {
            "dictionary_atoms": "columns of G in GF(2)^D",
            "fixed_linear_decoder": "H with G H = I_D",
            "query_coefficients": "a(q)=H q",
            "nonzero_row_activity": (
                "Pr[sum_i r_i^T A_i u_i=1]=(1-2^-sum_i rank(A_i))/2"
            ),
            "independent_batch_union": "1-(3/4)^K",
            "hot_cells_and_query_compute": "FREE_FAVORABLE_GRANT",
        },
        "registered_floor": floor,
        "io_comparison": {
            "requested_block_bytes": str(REQUESTED_BLOCK_BYTES),
            "requested_block_bytes_decimal": float(REQUESTED_BLOCK_BYTES),
            "cold_over_requested_block": float(
                Fraction(cold_bytes, 1) / REQUESTED_BLOCK_BYTES
            ),
            "favorable_link_bytes_per_second": (
                FAVORABLE_LINK_BYTES_PER_SECOND
            ),
            "zero_compute_block_ms": float(block_seconds * 1_000),
            "zero_compute_ms_per_token": float(per_token_seconds * 1_000),
            "exceeds_20ms_per_token_before_other_costs": (
                per_token_seconds > Fraction(1, 50)
            ),
        },
        "claim_boundary": {
            "fixed_linear_decoder": "REJECTED",
            "invertible_basis_change": "COVERED",
            "fixed_fft_or_tensor_transform_inverse": "COVERED",
            "linear_syndrome_representative": "COVERED",
            "nonlinear_minimum_weight_syndrome_decoder": "NOT_COVERED",
            "causal_transformer_reachability_of_independent_batch": (
                "NOT_ESTABLISHED"
            ),
            "native_numerical_lift": "NOT_SUPPLIED",
            "arbitrary_nonlinear_hot_advice": "NOT_COVERED",
            "general_nonlinear_data_structure": "NOT_COVERED",
            "target_candidate": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_FIXED_LINEAR_FUNCTIONAL_DECODER",
            "KEEP_NONLINEAR_SYNDROME_DECODER_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
