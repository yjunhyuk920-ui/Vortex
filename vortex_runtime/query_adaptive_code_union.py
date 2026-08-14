"""E0 bound for query-adaptive unions of exact bilinear codebooks.

The screened mechanism lets a causal router choose one linear codebook (a
``leaf``) for the current Bilinear Cross Residual query.  Only that leaf is
scanned, so a nonlinear union can contain more exact query directions than one
globally scanned ledger.  Routing does not, however, manufacture the cached
answers stored by those leaves.  This module charges every stored direction's
build and persistent representation and proves the resulting conservation
bound.

The result is scoped to trace-built linear leaves.  It is not a lower bound for
an implicit nonlinear checkpoint data structure that can produce exact
bilinear answers without materializing cached query/answer directions.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from vortex_runtime.causal_bilinear_query_restriction import (
    BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE,
    CHECKPOINT_SERVICE_TOKENS,
    HOT_STATE_BYTES,
    P50_TARGET_FRACTION,
    REGISTERED_Q4_BYTES,
    VERIFIER_STATE_BYTES,
    factor_scan_bytes,
    ledger_metadata_bytes,
    ledger_query_operations,
    ledger_query_traffic_bytes,
    ledger_state_bytes,
    largest_scannable_span_dimension,
)
from vortex_runtime.query_adaptive_cold_equation import (
    GIB,
    REGISTERED_NON_EMBEDDING_COEFFICIENTS,
    REGISTERED_TOTAL_Q4_BYTES,
)


REJECT_DECISION = "REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE"
MAXIMUM_LEAF_DIMENSION = 23
REGISTERED_LEAF_DIMENSIONS = (1, 2, 4, 8, 16, 23)
EXP084_REGISTERED_PRIMES = (65_521, 65_519, 65_497)


def _nonnegative(value: int, *, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _floor_nonnegative(value: Fraction) -> int:
    return 0 if value <= 0 else value.numerator // value.denominator


def query_fraction_without_build(
    leaf_dimension: int, *, resource: str
) -> Fraction:
    """Return one selected leaf's favorable every-query resource fraction."""

    _nonnegative(leaf_dimension, name="leaf_dimension")
    if resource == "traffic":
        return Fraction(
            ledger_query_traffic_bytes(leaf_dimension), REGISTERED_Q4_BYTES
        )
    if resource == "operations":
        return Fraction(
            ledger_query_operations(leaf_dimension),
            REGISTERED_NON_EMBEDDING_COEFFICIENTS,
        )
    raise ValueError("resource must be 'traffic' or 'operations'")


def build_amortized_fraction(total_cached_directions: int) -> Fraction:
    """Charge one forward and one reverse dense equivalent per direction."""

    _nonnegative(total_cached_directions, name="total_cached_directions")
    return Fraction(
        BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE * total_cached_directions,
        CHECKPOINT_SERVICE_TOKENS,
    )


def maximum_total_cached_directions(leaf_dimension: int) -> int:
    """Largest union-wide direction count fitting before any fallback.

    Only one leaf is read per query, but the build of every leaf is amortized
    over the same registered service population.  Selector, leaf lookup, cold
    address traffic, and misses are granted free, making this an upper bound.
    """

    if leaf_dimension <= 0:
        raise ValueError("leaf_dimension must be positive")
    if ledger_state_bytes(leaf_dimension) > HOT_STATE_BYTES:
        return 0
    limits: list[int] = []
    for resource in ("traffic", "operations"):
        headroom = P50_TARGET_FRACTION - query_fraction_without_build(
            leaf_dimension, resource=resource
        )
        limits.append(
            _floor_nonnegative(
                headroom
                * CHECKPOINT_SERVICE_TOKENS
                / BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE
            )
        )
    return min(limits)


def packed_leaf_dimensions(
    total_cached_directions: int, leaf_dimension: int
) -> tuple[int, ...]:
    """Pack directions into the fewest leaves of at most ``leaf_dimension``."""

    _nonnegative(total_cached_directions, name="total_cached_directions")
    if leaf_dimension <= 0:
        raise ValueError("leaf_dimension must be positive")
    full, remainder = divmod(total_cached_directions, leaf_dimension)
    return (leaf_dimension,) * full + ((remainder,) if remainder else ())


def explicit_union_state_bytes(leaf_dimensions: Sequence[int]) -> int:
    """Persistent bytes for materialized factor leaves and one shared verifier."""

    if any(dimension <= 0 for dimension in leaf_dimensions):
        raise ValueError("every leaf dimension must be positive")
    total_directions = sum(leaf_dimensions)
    return (
        factor_scan_bytes(total_directions)
        + sum(
            sum(ledger_metadata_bytes(dimension).values())
            for dimension in leaf_dimensions
        )
        + VERIFIER_STATE_BYTES
    )


def minimum_required_full_fallback_coverage(
    leaf_dimension: int, total_cached_directions: int
) -> Fraction:
    """Coverage required when any miss performs one full dense fallback."""

    _nonnegative(total_cached_directions, name="total_cached_directions")
    build = build_amortized_fraction(total_cached_directions)
    requirements = (
        1
        + query_fraction_without_build(leaf_dimension, resource=resource)
        + build
        - P50_TARGET_FRACTION
        for resource in ("traffic", "operations")
    )
    return max(Fraction(0), *requirements)


def independent_query_hit_upper_bound(
    total_leaf_dimensions: int, query_count: int
) -> int:
    """Maximum hits from a globally independent exact query population.

    A ``d``-dimensional leaf contains at most ``d`` members of a globally
    linearly independent set.  Summing over all leaves gives this bound even
    to a future-aware perfect router.
    """

    _nonnegative(total_leaf_dimensions, name="total_leaf_dimensions")
    _nonnegative(query_count, name="query_count")
    return min(total_leaf_dimensions, query_count)


def independent_stream_fraction(
    leaf_dimension: int,
    total_cached_directions: int,
    *,
    resource: str,
    query_count: int = CHECKPOINT_SERVICE_TOKENS,
) -> Fraction:
    """Best charged fraction on a globally independent finite query stream."""

    if query_count <= 0:
        raise ValueError("query_count must be positive")
    hits = independent_query_hit_upper_bound(
        total_cached_directions, query_count
    )
    fallback = Fraction(query_count - hits, query_count)
    return (
        query_fraction_without_build(leaf_dimension, resource=resource)
        + build_amortized_fraction(total_cached_directions)
        + fallback
    )


def online_independent_stream_lower_bound(
    leaf_dimension: int, *, resource: str
) -> Fraction:
    """One-pass online grant when each miss itself becomes a cached answer.

    On a never-repeating independent stream every query still executes its
    dense fallback once.  Reusing that exact fallback as the build avoids the
    registered extra forward/reverse charge, yet leaves one dense equivalent
    plus the routed-leaf query overhead.
    """

    return 1 + query_fraction_without_build(leaf_dimension, resource=resource)


def calibration_union_contains_only_calibration_span(
    calibration_span: Iterable[int], leaves: Sequence[Iterable[int]]
) -> bool:
    """Finite reference check for the union-containment identity.

    Bit words represent elements of a small GF(2) vector space.  This helper is
    intentionally only a test/reference surface; the theorem over any field is
    immediate because every leaf generated from calibration rows is a
    subspace of their global span.
    """

    global_words = set(calibration_span)
    return all(set(leaf) <= global_words for leaf in leaves)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exp084_full_build_span_diagnostic(output_dir: Path) -> dict[str, Any]:
    """Replay all 24 build rows against the five frozen evaluation rows.

    This performs no Transformer forward and does not alter EXP-084A.  A rank
    increase modulo any registered prime certifies rational non-membership in
    the complete 24-row calibration span, hence in every union of leaf spaces
    generated only from those rows.
    """

    import numpy as np

    from vortex_runtime.causal_bilinear_rank_gate import (
        FactorizedModularSpan,
        factor_pair_sha256,
    )

    root = output_dir.resolve()
    build_path = root / "raw" / "build_rows.jsonl"
    evaluation_path = root / "raw" / "evaluation_rows.jsonl"

    def load_rows(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def load_pair(row: dict[str, Any]) -> tuple[Any, Any]:
        with np.load(root / row["array_file"], allow_pickle=False) as arrays:
            pair = (
                np.ascontiguousarray(arrays["r"], dtype=np.float32),
                np.ascontiguousarray(arrays["u"], dtype=np.float32),
            )
        if factor_pair_sha256(*pair) != row["pair_sha256"]:
            raise ValueError(f"pair hash mismatch: {row['row_id']}")
        return pair

    build_rows = load_rows(build_path)
    evaluation_rows = load_rows(evaluation_path)
    build_pairs = [load_pair(row) for row in build_rows]
    evaluation_pairs = [load_pair(row) for row in evaluation_rows]
    if len(build_pairs) != 24 or len(evaluation_pairs) != 5:
        raise ValueError("unexpected frozen EXP-084A row counts")

    ranks: dict[str, int] = {}
    row_results: list[dict[str, Any]] = [
        {
            "row_id": row["row_id"],
            "membership_ranks": {},
            "rank_increases": {},
        }
        for row in evaluation_rows
    ]
    for prime in EXP084_REGISTERED_PRIMES:
        base = FactorizedModularSpan(1_024, 3_584, prime)
        for pair in build_pairs:
            incremented, _ = base.add(*pair)
            if not incremented:
                raise ValueError("frozen build row did not increase full span")
        ranks[str(prime)] = base.rank
        for result, pair in zip(row_results, evaluation_pairs, strict=True):
            probe = FactorizedModularSpan(1_024, 3_584, prime)
            for build_pair in build_pairs:
                probe.add(*build_pair)
            incremented, _ = probe.add(*pair)
            result["membership_ranks"][str(prime)] = probe.rank
            result["rank_increases"][str(prime)] = incremented

    for result in row_results:
        result["certified_outside_full_build_span"] = any(
            result["rank_increases"].values()
        )
    misses = sum(
        bool(result["certified_outside_full_build_span"])
        for result in row_results
    )
    return {
        "source": "FROZEN_EXP_084A_ZERO_FORWARD_REPLAY",
        "model_forward_calls": 0,
        "build_rows": len(build_rows),
        "evaluation_rows": len(evaluation_rows),
        "build_modular_ranks": ranks,
        "full_build_union_exact_hits": len(row_results) - misses,
        "full_build_union_certified_misses": misses,
        "rows": row_results,
        "build_rows_sha256": _sha256(build_path),
        "evaluation_rows_sha256": _sha256(evaluation_path),
    }


def derive_frontier(
    exp084_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the registered E0 union bound and optional frozen diagnostic."""

    if largest_scannable_span_dimension() != MAXIMUM_LEAF_DIMENSION:
        raise AssertionError("registered leaf dimension drifted")

    table: list[dict[str, Any]] = []
    for dimension in REGISTERED_LEAF_DIMENSIONS:
        total = maximum_total_cached_directions(dimension)
        leaves = packed_leaf_dimensions(total, dimension)
        build = build_amortized_fraction(total)
        required = minimum_required_full_fallback_coverage(dimension, total)
        independent_coverage = Fraction(total, CHECKPOINT_SERVICE_TOKENS)
        independent_traffic = independent_stream_fraction(
            dimension, total, resource="traffic"
        )
        persistent = explicit_union_state_bytes(leaves)
        table.append(
            {
                "leaf_dimension": dimension,
                "query_traffic_fraction_without_build": str(
                    query_fraction_without_build(dimension, resource="traffic")
                ),
                "query_operation_fraction_without_build": str(
                    query_fraction_without_build(dimension, resource="operations")
                ),
                "selected_leaf_state_bytes": ledger_state_bytes(dimension),
                "selected_leaf_state_gib": ledger_state_bytes(dimension) / GIB,
                "maximum_total_cached_directions": total,
                "packed_leaf_count": len(leaves),
                "packed_full_leaf_count": total // dimension,
                "packed_remainder_dimension": total % dimension,
                "build_amortized_fraction": str(build),
                "minimum_full_fallback_coverage": str(required),
                "minimum_full_fallback_coverage_decimal": float(required),
                "maximum_independent_service_coverage": str(
                    independent_coverage
                ),
                "maximum_independent_service_coverage_decimal": float(
                    independent_coverage
                ),
                "independent_stream_traffic_fraction": str(independent_traffic),
                "independent_stream_traffic_fraction_decimal": float(
                    independent_traffic
                ),
                "independent_stream_target_multiple": float(
                    independent_traffic / P50_TARGET_FRACTION
                ),
                "explicit_union_state_bytes": persistent,
                "explicit_union_state_tib": persistent / (1024**4),
                "checkpoint_plus_union_state_gib": (
                    REGISTERED_TOTAL_Q4_BYTES + persistent
                )
                / GIB,
            }
        )

    best = max(table, key=lambda row: int(row["maximum_total_cached_directions"]))
    result: dict[str, Any] = {
        "classification": "E0_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_BOUND",
        "decision": REJECT_DECISION,
        "model": "PERFECT_ROUTER_UNION_OF_MATERIALIZED_EXACT_LINEAR_LEAVES",
        "normalization": {
            "service_tokens": CHECKPOINT_SERVICE_TOKENS,
            "p50_target_fraction": str(P50_TARGET_FRACTION),
            "p50_target_fraction_decimal": float(P50_TARGET_FRACTION),
            "maximum_leaf_dimension": MAXIMUM_LEAF_DIMENSION,
            "build_dense_equivalents_per_cached_direction": (
                BUILD_DENSE_EQUIVALENTS_PER_BASIS_TUPLE
            ),
            "registered_total_q4_bytes": REGISTERED_TOTAL_Q4_BYTES,
        },
        "union_information_theorem": {
            "hit_set": "subset of union of selected linear leaf spaces",
            "independent_population_hits_at_most_sum_leaf_dimensions": True,
            "router_creates_new_answer_direction": False,
            "leaves_generated_from_one_calibration_span_remain_in_that_span": True,
            "one_pass_online_independent_stream_traffic_lower_bound": str(
                online_independent_stream_lower_bound(1, resource="traffic")
            ),
            "one_pass_online_independent_stream_target_multiple": float(
                online_independent_stream_lower_bound(1, resource="traffic")
                / P50_TARGET_FRACTION
            ),
        },
        "registered_leaf_table": table,
        "strongest_direction_count_ceiling": {
            "leaf_dimension": best["leaf_dimension"],
            "maximum_total_cached_directions": best[
                "maximum_total_cached_directions"
            ],
            "maximum_independent_service_coverage": best[
                "maximum_independent_service_coverage"
            ],
            "minimum_full_fallback_coverage": best[
                "minimum_full_fallback_coverage"
            ],
            "explicit_union_state_tib": best["explicit_union_state_tib"],
            "independent_stream_target_multiple": best[
                "independent_stream_target_multiple"
            ],
        },
        "favorable_cost_grants": {
            "causal_router": "FREE PERFECT ROUTER",
            "leaf_index_lookup": "FREE",
            "pair_extraction_and_local_result_source": "FREE",
            "native_numerical_repair": "FREE",
            "cold_address_and_request_latency": "FREE",
            "kv_workspace_fallback_overlap": "OMITTED",
            "only_selected_leaf_hot": True,
            "all_leaf_build_and_persistent_factor_state": "CHARGED",
        },
        "claim_boundary": {
            "single_span_parameter_sweep": "NOT PERFORMED",
            "nonlinear_routing_over_trace_built_linear_leaves": "REJECTED AS NEW CORE",
            "restricted_repeated_query_cache": "AUXILIARY ONLY",
            "implicit_nonlinear_checkpoint_data_structure": "NOT RULED OUT",
            "finite_causal_query_union_complexity_beyond_frozen_rows": "NOT MEASURED",
            "paid_general_pair_extractor": "NOT CONSTRUCTED",
            "model_operation_replacement": "NOT RUN",
            "physical_hardware_or_latency": "NOT RUN",
            "core_candidate": "NONE",
        },
    }
    if exp084_diagnostic is not None:
        result["exp084_full_build_span_diagnostic"] = exp084_diagnostic
    return result
