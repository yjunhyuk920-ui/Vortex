"""E0 accounting for a causal Bilinear Cross Residual query restriction.

This module studies one deliberately favorable execution interface.  A
``Causal Bilinear Span Ledger`` stores a factorized basis of aligned
rank-one query tuples ``q_i = r_i u_i^T`` for every registered matrix.  A
query scans the stored factors, reconstructs a certified linear combination
of cached scalar answers, and falls back on every non-member.

The calculator is a necessary-condition screen, not a native Transformer
executor.  In particular, it grants the current residual-pair extractor and
the local result/proposal source for free.  The only currently known general
extractor is a dense reverse traversal, so passing the population Gate would
authorize only a paid-extractor/numerical-semantics Gate.
"""

from __future__ import annotations

from fractions import Fraction

from vortex_runtime.lower_bound_audit import TensorSpec, llama_405b_tensor_plan
from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
)


P50_TARGET_FRACTION = Fraction(8, 675)
HOT_STATE_BYTES = 8 * GIB
REGISTERED_Q4_BYTES = REGISTERED_NON_EMBEDDING_COEFFICIENTS // 2

# The favorable ledger stores one two-byte scalar for every left/right factor
# coordinate.  Real exact residual factors can require more; two bytes is a
# grant that makes the rejection threshold harder to reach.
FACTOR_SCALAR_BYTES = 2
GRAM_SCALAR_BYTES = 4
CACHED_ANSWER_BYTES = 4
PIVOT_DESCRIPTOR_BYTES = 8
FINGERPRINT_SCALAR_BYTES = 2

VERIFIER_CHALLENGES = 6
VERIFIER_FIELD_PRIME = 65_521
REGISTERED_RANK_PRIMES = (65_521, 65_519, 65_497)

# Frozen six-check trace verifier values from
# docs/research/E0_DECISION_AND_PROOF_TRACE_TRIAGE.md.
VERIFIER_OPERATIONS = 234_659_328
VERIFIER_TRAFFIC_BYTES = 538_678_272
VERIFIER_STATE_BYTES = 458_686_464

CHECKPOINT_SERVICE_TOKENS = 20_000_000
BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE = 2

# Cheapest pinned E1 screen: six build prompts x four decode positions and
# eighteen held-out prompts x two decode positions.  Only the last small-model
# down projection is observed; its miss is weighted as one registered 405B
# down-projection instance, while all other matrix queries are granted hits.
BUILD_PROMPTS = 6
BUILD_POSITIONS_PER_PROMPT = 4
EVALUATION_PROMPTS = 18
EVALUATION_POSITIONS_PER_PROMPT = 2
EVALUATION_QUERY_COUNT = EVALUATION_PROMPTS * EVALUATION_POSITIONS_PER_PROMPT
SMALL_MODEL_SIDE_RANK = 16
SMALL_MODEL_LAST_DOWN_ROWS = 1_024
SMALL_MODEL_LAST_DOWN_COLUMNS = 3_584


def registered_non_embedding_specs() -> tuple[TensorSpec, ...]:
    return tuple(
        spec for spec in llama_405b_tensor_plan() if spec.name != "embedding"
    )


def aligned_pair_coordinate_count() -> int:
    """Coordinates in one aligned ``(r_i,u_i)`` tuple over all matrices."""

    return sum(
        spec.count * (spec.rows + spec.columns)
        for spec in registered_non_embedding_specs()
    )


def one_down_projection_coefficients() -> int:
    down = next(
        spec for spec in registered_non_embedding_specs() if spec.name == "down_proj"
    )
    return down.rows * down.columns


def factor_scan_bytes(span_dimension: int) -> int:
    if span_dimension < 0:
        raise ValueError("span_dimension must be non-negative")
    return span_dimension * FACTOR_SCALAR_BYTES * aligned_pair_coordinate_count()


def ledger_metadata_bytes(span_dimension: int) -> dict[str, int]:
    """Compact per-query metadata read and resident state for the ledger."""

    if span_dimension < 0:
        raise ValueError("span_dimension must be non-negative")
    matrices = REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES
    return {
        "gram_inverse": GRAM_SCALAR_BYTES * matrices * span_dimension**2,
        "cached_answers": CACHED_ANSWER_BYTES * matrices * span_dimension,
        "pivot_descriptors": PIVOT_DESCRIPTOR_BYTES * matrices * span_dimension,
        "basis_fingerprints": (
            FINGERPRINT_SCALAR_BYTES
            * VERIFIER_CHALLENGES
            * matrices
            * span_dimension
        ),
    }


def ledger_query_operations(span_dimension: int) -> int:
    """Favorable coefficient-equivalent operations before pair extraction."""

    if span_dimension < 0:
        raise ValueError("span_dimension must be non-negative")
    coordinates = aligned_pair_coordinate_count()
    matrices = REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES
    return (
        span_dimension * coordinates
        + matrices * span_dimension**2
        + matrices * span_dimension
        + VERIFIER_CHALLENGES * matrices * span_dimension
        + VERIFIER_OPERATIONS
    )


def ledger_query_traffic_bytes(span_dimension: int) -> int:
    metadata = ledger_metadata_bytes(span_dimension)
    return (
        factor_scan_bytes(span_dimension)
        + sum(metadata.values())
        + VERIFIER_TRAFFIC_BYTES
    )


def ledger_state_bytes(span_dimension: int) -> int:
    metadata = ledger_metadata_bytes(span_dimension)
    return (
        factor_scan_bytes(span_dimension)
        + sum(metadata.values())
        + VERIFIER_STATE_BYTES
    )


def build_amortized_fraction(span_dimension: int) -> Fraction:
    if span_dimension < 0:
        raise ValueError("span_dimension must be non-negative")
    return Fraction(
        BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE * span_dimension,
        CHECKPOINT_SERVICE_TOKENS,
    )


def traffic_fraction_before_fallback(span_dimension: int) -> Fraction:
    return Fraction(
        ledger_query_traffic_bytes(span_dimension), REGISTERED_Q4_BYTES
    ) + build_amortized_fraction(span_dimension)


def operation_fraction_before_fallback(span_dimension: int) -> Fraction:
    return Fraction(
        ledger_query_operations(span_dimension),
        REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    ) + build_amortized_fraction(span_dimension)


def largest_scannable_span_dimension() -> int:
    """Largest dimension meeting traffic, operations, and component state."""

    dimension = 0
    while True:
        candidate = dimension + 1
        if (
            traffic_fraction_before_fallback(candidate) > P50_TARGET_FRACTION
            or operation_fraction_before_fallback(candidate) > P50_TARGET_FRACTION
            or ledger_state_bytes(candidate) > HOT_STATE_BYTES
        ):
            return dimension
        dimension = candidate


def span_miss_lower_bound(
    certified_rank: int,
    population_size: int,
    span_dimension: int,
) -> int:
    """Minimum misses for any post-hoc best ``span_dimension`` subspace.

    If all but ``m`` population vectors lie in a ``B``-dimensional subspace,
    their total rank is at most ``B+m``.  Therefore a certified population
    rank ``R`` forces at least ``max(0, R-B)`` misses.  The argument grants a
    future-aware best subspace and is consequently safe for rejection.
    """

    if certified_rank < 0 or population_size < 0 or span_dimension < 0:
        raise ValueError("rank, population, and span must be non-negative")
    if certified_rank > population_size:
        raise ValueError("certified_rank cannot exceed population_size")
    return max(0, certified_rank - span_dimension)


def maximum_weighted_fallback_misses(
    *,
    population_size: int,
    matrix_coefficients: int,
    common_fraction: Fraction,
    target_fraction: Fraction = P50_TARGET_FRACTION,
) -> int:
    """Whole-matrix fallbacks allowed after the common ledger cost."""

    if population_size <= 0 or matrix_coefficients <= 0:
        raise ValueError("population and matrix coefficients must be positive")
    if common_fraction < 0 or target_fraction <= 0:
        raise ValueError("fractions must be non-negative and target positive")
    headroom = target_fraction - common_fraction
    if headroom < 0:
        return -1
    matrix_share = Fraction(
        matrix_coefficients, REGISTERED_NON_EMBEDDING_COEFFICIENTS
    )
    return int((headroom * population_size) // matrix_share)


def minimum_rejecting_rank(
    *,
    population_size: int,
    span_dimension: int,
    allowed_misses: int,
) -> int | None:
    """First rank whose oracle miss bound exceeds the miss allowance."""

    if allowed_misses < 0:
        return 0
    threshold = span_dimension + allowed_misses + 1
    return threshold if threshold <= population_size else None


def fingerprint_union_bound(
    *,
    prime: int = VERIFIER_FIELD_PRIME,
    challenges: int = VERIFIER_CHALLENGES,
    queries: int = (
        REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES
        * CHECKPOINT_SERVICE_TOKENS
    ),
) -> Fraction:
    """Union bound for independent rank-one bilinear identity checks.

    For nonzero matrix ``E`` over ``F_p``, uniformly random ``a,b`` satisfy
    ``a^T E b = 0`` with probability at most ``2/p - 1/p^2``.  Exact-rational
    membership remains a separate prerequisite; this verifier only protects
    the declared finite-field identity and implementation path.
    """

    if prime <= 2 or challenges <= 0 or queries < 0:
        raise ValueError("invalid fingerprint parameters")
    one_check = Fraction(2 * prime - 1, prime * prime)
    return queries * one_check**challenges


def factor_scan_address_counts(
    span_dimension: int,
    *,
    line_bytes: int = 64,
    page_bytes: int = 4_096,
) -> dict[str, int]:
    """Best-case contiguous factor payload plus vector boundary rounding."""

    if span_dimension < 0 or line_bytes <= 0 or page_bytes <= 0:
        raise ValueError("invalid span or address unit")
    segments = 0
    lines = 0
    pages = 0
    for spec in registered_non_embedding_specs():
        for length in (spec.rows, spec.columns):
            payload = FACTOR_SCALAR_BYTES * length
            segments += spec.count * span_dimension
            lines += (
                spec.count
                * span_dimension
                * ((payload + line_bytes - 1) // line_bytes)
            )
            pages += (
                spec.count
                * span_dimension
                * ((payload + page_bytes - 1) // page_bytes)
            )
    return {
        "vector_segments": segments,
        "rounded_64_byte_lines": lines,
        "rounded_4k_pages": pages,
    }


def derive_frontier() -> dict[str, object]:
    span_dimension = largest_scannable_span_dimension()
    common_traffic = traffic_fraction_before_fallback(span_dimension)
    common_operations = operation_fraction_before_fallback(span_dimension)
    down_coefficients = one_down_projection_coefficients()
    allowed_misses = maximum_weighted_fallback_misses(
        population_size=EVALUATION_QUERY_COUNT,
        matrix_coefficients=down_coefficients,
        common_fraction=common_traffic,
    )
    rejecting_rank = minimum_rejecting_rank(
        population_size=EVALUATION_QUERY_COUNT,
        span_dimension=span_dimension,
        allowed_misses=allowed_misses,
    )
    independent_rank = EVALUATION_QUERY_COUNT
    independent_misses = span_miss_lower_bound(
        independent_rank, EVALUATION_QUERY_COUNT, span_dimension
    )
    independent_fallback = Fraction(
        independent_misses * down_coefficients,
        EVALUATION_QUERY_COUNT * REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    )

    return {
        "classification": "E0_CAUSAL_BILINEAR_SPAN_LEDGER_NECESSARY_GATE",
        "target": {
            "non_embedding_coefficients": REGISTERED_NON_EMBEDDING_COEFFICIENTS,
            "non_embedding_q4_bytes": REGISTERED_Q4_BYTES,
            "matrix_instances": REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES,
            "p50_fraction": str(P50_TARGET_FRACTION),
            "hot_state_bytes": HOT_STATE_BYTES,
        },
        "ledger_shape": {
            "aligned_pair_coordinates": aligned_pair_coordinate_count(),
            "factor_scalar_bytes_favorable": FACTOR_SCALAR_BYTES,
            "maximum_scannable_span_dimension": span_dimension,
            "next_dimension_fails": span_dimension + 1,
            "basis_factor_bytes": factor_scan_bytes(span_dimension),
            "metadata_bytes": ledger_metadata_bytes(span_dimension),
            "verifier_traffic_bytes": VERIFIER_TRAFFIC_BYTES,
            "verifier_state_bytes": VERIFIER_STATE_BYTES,
            "component_state_bytes": ledger_state_bytes(span_dimension),
            "component_state_gib": ledger_state_bytes(span_dimension) / GIB,
            "address_counts": factor_scan_address_counts(span_dimension),
        },
        "resource_equation": {
            "build_dense_equivalents": (
                BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE * span_dimension
            ),
            "checkpoint_service_tokens": CHECKPOINT_SERVICE_TOKENS,
            "build_amortized_fraction": str(
                build_amortized_fraction(span_dimension)
            ),
            "common_traffic_fraction": str(common_traffic),
            "common_traffic_fraction_decimal": float(common_traffic),
            "common_operation_fraction": str(common_operations),
            "common_operation_fraction_decimal": float(common_operations),
            "traffic_headroom_for_weighted_fallback": str(
                P50_TARGET_FRACTION - common_traffic
            ),
            "pair_extraction": "FREE FAVORABLE ORACLE; KNOWN GENERAL VJP IS DENSE",
            "local_trace_or_proposal_source": "FREE FAVORABLE ORACLE",
            "native_rounding_and_exact_rational_decode": "UNVERIFIED",
            "kv_workspace_fallback_overlap": "OMITTED",
        },
        "rank_gate": {
            "side_rank": SMALL_MODEL_SIDE_RANK,
            "build_prompts": BUILD_PROMPTS,
            "build_positions_per_prompt": BUILD_POSITIONS_PER_PROMPT,
            "build_query_rows": BUILD_PROMPTS * BUILD_POSITIONS_PER_PROMPT,
            "evaluation_prompts": EVALUATION_PROMPTS,
            "evaluation_positions_per_prompt": EVALUATION_POSITIONS_PER_PROMPT,
            "evaluation_query_rows": EVALUATION_QUERY_COUNT,
            "small_last_down_shape": [
                SMALL_MODEL_LAST_DOWN_ROWS,
                SMALL_MODEL_LAST_DOWN_COLUMNS,
            ],
            "registered_down_coefficients_per_instance": down_coefficients,
            "registered_down_fraction": str(
                Fraction(
                    down_coefficients,
                    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
                )
            ),
            "maximum_allowed_down_fallback_rows": allowed_misses,
            "minimum_rejecting_certified_rank": rejecting_rank,
            "registered_rank_primes": list(REGISTERED_RANK_PRIMES),
            "rank_to_oracle_miss_rule": "misses >= max(0, rank - 23)",
        },
        "strongest_counterexample": {
            "all_evaluation_tensors_independent_rank": independent_rank,
            "oracle_miss_rows": independent_misses,
            "weighted_down_fallback_fraction": str(independent_fallback),
            "fully_charged_traffic_fraction": str(
                common_traffic + independent_fallback
            ),
            "target_miss_factor": float(
                (common_traffic + independent_fallback)
                / P50_TARGET_FRACTION
            ),
        },
        "fingerprint_contract": {
            "field_prime": VERIFIER_FIELD_PRIME,
            "challenges": VERIFIER_CHALLENGES,
            "service_query_union": (
                REGISTERED_NON_EMBEDDING_MATRIX_INSTANCES
                * CHECKPOINT_SERVICE_TOKENS
            ),
            "union_false_accept_bound": str(fingerprint_union_bound()),
            "union_false_accept_bound_decimal": float(
                fingerprint_union_bound()
            ),
            "exact_rational_witness_required_before_hit": True,
            "modular_nonincrease_alone_is_not_a_hit": True,
        },
        "decision": "PREREGISTER_LAST_DOWN_CAUSAL_BILINEAR_RANK_GATE_ONLY",
        "claim_boundary": {
            "query_restriction_proved": "NO; MEASURED POPULATION PENDING",
            "core_candidate": "NONE; PAIR EXTRACTOR AND LOCAL RESULT SOURCE ARE FREE",
            "activation_span_replay": "NOT REOPENED",
            "general_nonlinear_or_compressed_query_code": "NOT RULED OUT",
            "native_floating_operation_replacement": "NOT TESTED",
            "model_or_hardware_execution": "NOT RUN",
        },
    }
