"""Causal-utility Gate for Cartesian bilinear batching.

For matrices ``R``, ``W``, and ``U``, one streamed use of ``W`` can produce
all entries of ``R.T @ W @ U``.  With ``k`` left and right directions this is
``k**2`` exact scalar measurements.  The measurements, however, refer to only
the ``k`` forward states represented by the columns of ``U``.  Varying the
left direction asks another question about the same state; it does not create
another causally usable successor state.

This module freezes that distinction and charges the optimistic full-sweep
floor.  It is not a lower bound against partial-evidence or nonlinear cold
sources.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, localcontext


REGISTERED_PARAMETERS = 405_849_243_648
REGISTERED_BATCH_STATES = 32
REGISTERED_EFFECTIVE_BITS_PER_WEIGHT = Decimal("10.6")
REGISTERED_DFLOAT_BYTES = Decimal("551220000000")
EFFECTIVE_LINK_BYTES_PER_SECOND = Decimal("32000000000")
TARGET_SECONDS_PER_TOKEN = Decimal("0.020")

DECISION = (
    "REJECT_CARTESIAN_SCALAR_COUNT_AS_CAUSAL_TOKEN_AMPLIFICATION_"
    "KEEP_CARTESIAN_BILINEAR_BATCHING_AS_AUXILIARY_"
    "REQUIRE_PARTIAL_EXACT_INFORMATION_SOURCE"
)


def gf2_matmul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """Small independent GF(2) matrix multiply used by reference controls."""

    if not left or not right or not left[0] or not right[0]:
        raise ValueError("matrices must be nonempty")
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix is ragged")
    if len(right) != inner:
        raise ValueError("inner dimensions differ")
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise ValueError("right matrix is ragged")
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(inner)) & 1
            for j in range(width)
        ]
        for i in range(len(left))
    ]


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix is ragged")
    return [list(column) for column in zip(*matrix)]


def cartesian_bilinear(
    left_directions: list[list[int]],
    weight: list[list[int]],
    forward_states: list[list[int]],
) -> list[list[int]]:
    """Return ``R.T @ W @ U`` over GF(2).

    Directions and states are stored as columns.  This intentionally exposes
    every Cartesian scalar while retaining the number of state columns.
    """

    return gf2_matmul(
        transpose(left_directions), gf2_matmul(weight, forward_states)
    )


def causal_utility_accounting(
    left_direction_count: int,
    forward_state_count: int,
) -> dict[str, int | bool | str]:
    """Separate exact scalar count from causally distinct forward states."""

    if left_direction_count <= 0 or forward_state_count <= 0:
        raise ValueError("counts must be positive")
    scalars = left_direction_count * forward_state_count
    return {
        "left_direction_count": left_direction_count,
        "forward_state_count": forward_state_count,
        "cartesian_exact_scalar_count": scalars,
        "maximum_distinct_forward_states": forward_state_count,
        "maximum_causal_tokens_from_these_states": forward_state_count,
        "scalar_to_state_ratio": str(
            Decimal(scalars) / Decimal(forward_state_count)
        ),
        "left_direction_creates_successor_state": False,
        "cartesian_scalars_may_be_counted_as_tokens": False,
    }


def full_sweep_accounting(
    state_count: int = REGISTERED_BATCH_STATES,
) -> dict[str, int | str | bool]:
    """Charge one optimistic near-Shannon BF16 sweep.

    All decompression, GEMM, metadata, state, KV, and scheduler costs are
    granted free.  The only denominator allowed is causally distinct forward
    states, never the number of left/right scalar cross-products.
    """

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    with localcontext() as context:
        context.prec = 50
        parameters = Decimal(REGISTERED_PARAMETERS)
        sweep_bytes = parameters * REGISTERED_EFFECTIVE_BITS_PER_WEIGHT / 8
        sweep_seconds = sweep_bytes / EFFECTIVE_LINK_BYTES_PER_SECOND
        seconds_per_state = sweep_seconds / Decimal(state_count)
        false_cartesian_seconds = sweep_seconds / Decimal(state_count**2)
        required_tokens_decimal = sweep_seconds / TARGET_SECONDS_PER_TOKEN
        required_tokens = int(
            required_tokens_decimal.to_integral_value(rounding=ROUND_CEILING)
        )
        dfloat_seconds = REGISTERED_DFLOAT_BYTES / EFFECTIVE_LINK_BYTES_PER_SECOND
        return {
            "parameters": REGISTERED_PARAMETERS,
            "effective_bits_per_weight": str(
                REGISTERED_EFFECTIVE_BITS_PER_WEIGHT
            ),
            "optimistic_sweep_bytes": str(sweep_bytes),
            "optimistic_sweep_seconds": str(sweep_seconds),
            "causal_forward_states": state_count,
            "exact_scalars_in_square_cartesian_table": state_count**2,
            "valid_milliseconds_per_token": str(
                seconds_per_state * Decimal(1000)
            ),
            "valid_target_multiple": str(
                seconds_per_state / TARGET_SECONDS_PER_TOKEN
            ),
            "invalid_scalar_denominator_milliseconds": str(
                false_cartesian_seconds * Decimal(1000)
            ),
            "invalid_scalar_denominator_would_pass": (
                false_cartesian_seconds <= TARGET_SECONDS_PER_TOKEN
            ),
            "minimum_causally_usable_tokens_per_full_sweep": required_tokens,
            "registered_states_meet_full_sweep_floor": (
                state_count >= required_tokens
            ),
            "dfloat_milliseconds_per_token": str(
                dfloat_seconds / Decimal(state_count) * Decimal(1000)
            ),
            "all_non_link_costs_granted_free": True,
        }


def derive_audit() -> dict[str, object]:
    causal = causal_utility_accounting(
        REGISTERED_BATCH_STATES, REGISTERED_BATCH_STATES
    )
    sweep = full_sweep_accounting()
    return {
        "classification": "E0_CARTESIAN_BILINEAR_CAUSAL_UTILITY_GATE",
        "exact_identity": "C=R^T W U",
        "causal_utility": causal,
        "full_sweep": sweep,
        "new_lossless_literature_boundary": {
            "bf16_entropy_bits_per_weight_range": "10-12",
            "llama_405b_bf16_reported_compression_factor": "about 1.5x",
            "tile_random_access": True,
            "consumed_tiles_are_fully_decoded": True,
            "partial_transformer_decision_certificate_supplied": False,
        },
        "claim_boundary": {
            "cartesian_scalar_identity_exact": True,
            "cartesian_scalar_reuse_useful": True,
            "causal_token_count_equals_scalar_count": False,
            "full_sweep_core_passes": False,
            "partial_evidence_source_rejected": False,
            "nonlinear_rank_one_source_rejected": False,
            "native_model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "PROMOTE_CARTESIAN_BILINEAR_BATCHING_AS_EXACT_AUXILIARY",
            "REJECT_K_SQUARED_SCALARS_AS_K_SQUARED_CAUSAL_TOKENS",
            "REJECT_FULL_LOSSLESS_SWEEP_AT_32_CAUSAL_STATES",
            "REQUIRE_PARTIAL_EXACT_INFORMATION_SOURCE",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
