"""Finite audit of extension-field rank-saturating query layouts.

Packing an ``m x k`` binary rank-one mask by columns identifies it with a
rank-weight-one vector in ``GF(2**m)**k``.  Rank-saturating systems therefore
look superficially like a low-probe functional dictionary.  The key accounting
distinction is that coefficient *rank* is not coefficient support: even a
rank-one coefficient vector may have every basis coordinate nonzero.

This module charges the two direct materializations and reproduces tiny
nonlinear-coset witnesses.  It does not reject every nonlinear decoder.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations


REGISTERED_PARAMETERS = 405_849_243_648
GLOBAL_ADVICE_BITS = 8 * 8 * (1 << 30)
REGISTERED_BLOCK_TOKENS = 32
REGISTERED_BLOCK_ALLOWANCE_BYTES = 13_780_500_000
FAVORABLE_SHARED_BYTES_PER_SECOND = 32_000_000_000

LAYERS = 126
HIDDEN = 16_384
KV = 1_024
INTERMEDIATE = 53_248
VOCAB = 128_256

DECISION = (
    "REJECT_EXTENSION_FIELD_RANK_SATURATION_AS_LOW_PROBE_CONSTRUCTOR_"
    "KEEP_QUERY_DEPENDENT_NONLINEAR_COSET_DECODER_OPEN"
)


def registered_matrix_shapes() -> tuple[tuple[int, int, int], ...]:
    return (
        (HIDDEN, HIDDEN, 2 * LAYERS),
        (KV, HIDDEN, 2 * LAYERS),
        (INTERMEDIATE, HIDDEN, 2 * LAYERS),
        (HIDDEN, INTERMEDIATE, LAYERS),
        (VOCAB, HIDDEN, 1),
        (HIDDEN, VOCAB, 1),
    )


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def full_ambient_rank_one_materialization(
    rows: int, columns: int, multiplicity: int = 1
) -> dict[str, object]:
    """Charge the published rank-one-saturating full-ambient construction.

    Orient the matrix so the extension degree is the smaller side ``m`` and
    the extension-vector length is the larger side ``k``.  The exact minimum
    binary dimension for a rank-one-saturating q-system is ``m*(k-1)+1``.
    Materializing one ``GF(2**m)`` checkpoint summary for each basis vector
    therefore costs ``m`` bits per basis vector.
    """

    if rows <= 0 or columns <= 0 or multiplicity <= 0:
        raise ValueError("shape and multiplicity must be positive")
    extension_degree = min(rows, columns)
    vector_length = max(rows, columns)
    basis_dimension = extension_degree * (vector_length - 1) + 1
    raw_bits = rows * columns * multiplicity
    summary_bits = extension_degree * basis_dimension * multiplicity
    return {
        "rows": rows,
        "columns": columns,
        "multiplicity": multiplicity,
        "extension_degree": extension_degree,
        "extension_vector_length": vector_length,
        "q_system_binary_basis_dimension_rho_1": basis_dimension,
        "raw_one_bit_source_bits": raw_bits,
        "direct_field_summary_bits": summary_bits,
        "direct_summary_to_raw_ratio": str(Fraction(summary_bits, raw_bits)),
        "direct_summary_to_raw_ratio_decimal": summary_bits / raw_bits,
        "direct_summary_tib": summary_bits / (8 * (1 << 40)),
    }


def aggregate_full_ambient_materialization() -> dict[str, object]:
    cases = [
        full_ambient_rank_one_materialization(rows, columns, multiplicity)
        for rows, columns, multiplicity in registered_matrix_shapes()
    ]
    raw_bits = sum(int(case["raw_one_bit_source_bits"]) for case in cases)
    summary_bits = sum(int(case["direct_field_summary_bits"]) for case in cases)
    if raw_bits != REGISTERED_PARAMETERS:
        raise AssertionError("registered shape population drifted")
    return {
        "matrices": cases,
        "raw_one_bit_source_bits": raw_bits,
        "direct_field_summary_bits": summary_bits,
        "direct_summary_to_raw_ratio": str(Fraction(summary_bits, raw_bits)),
        "direct_summary_to_raw_ratio_decimal": summary_bits / raw_bits,
        "direct_summary_tib": summary_bits / (8 * (1 << 40)),
    }


def query_specialized_identity_batch(
    query_count: int = REGISTERED_BLOCK_TOKENS,
) -> dict[str, object]:
    """Charge the non-overkill q-system ``U=GF(2)^k``.

    A rank-one query is ``alpha*u`` with ``u`` binary.  The identity q-system
    stores one extension-field summary per coordinate, exactly the raw one-bit
    source size.  For independent uniform ``u`` vectors, each summary is used
    by the batch with probability ``1-2**(-K)``.  We grant every hot advice bit
    as an arbitrary source bit and perfect bit packing.
    """

    if query_count <= 0:
        raise ValueError("query count must be positive")
    union_probability = Fraction((1 << query_count) - 1, 1 << query_count)
    cold_bits = max(0, REGISTERED_PARAMETERS - GLOBAL_ADVICE_BITS)
    active_cold_bits = _ceil_fraction(union_probability * cold_bits)
    active_cold_bytes = (active_cold_bits + 7) // 8
    seconds_per_block = (
        active_cold_bytes / FAVORABLE_SHARED_BYTES_PER_SECOND
    )
    return {
        "query_count": query_count,
        "direct_persistent_bits": REGISTERED_PARAMETERS,
        "direct_persistent_equals_raw_one_bit_source": True,
        "uniform_batch_union_probability": str(union_probability),
        "uniform_batch_union_probability_decimal": float(union_probability),
        "complete_hot_grant_bits": GLOBAL_ADVICE_BITS,
        "expected_active_cold_bits_ceiling": active_cold_bits,
        "expected_active_cold_bytes_ceiling": active_cold_bytes,
        "registered_block_allowance_bytes": REGISTERED_BLOCK_ALLOWANCE_BYTES,
        "allowance_ratio": (
            active_cold_bytes / REGISTERED_BLOCK_ALLOWANCE_BYTES
        ),
        "zero_compute_ms_per_token_at_favorable_bandwidth": (
            1_000 * seconds_per_block / query_count
        ),
        "worst_case_single_query_can_activate_every_coordinate": True,
    }


def _gf2_rank(columns: tuple[int, ...], dimension: int) -> int:
    pivots: dict[int, int] = {}
    for column in columns:
        value = column
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def _rank_one_queries(rows: int, columns: int) -> tuple[int, ...]:
    result = {0}
    for left in range(1, 1 << rows):
        for right in range(1, 1 << columns):
            mask = 0
            for row in range(rows):
                if (left >> row) & 1:
                    mask ^= right << (row * columns)
            result.add(mask)
    return tuple(sorted(result))


def small_dictionary_witness(
    rows: int, columns: int, atoms: tuple[int, ...], query_count: int = 32
) -> dict[str, object]:
    """Evaluate one explicit dictionary with canonical minimum-weight leaders."""

    dimension = rows * columns
    if _gf2_rank(atoms, dimension) != dimension:
        raise ValueError("dictionary does not span the ambient space")
    infinity = len(atoms) + 1
    distance = [infinity] * (1 << dimension)
    leader = [0] * (1 << dimension)
    distance[0] = 0
    value = 0
    previous_gray = 0
    for counter in range(1, 1 << len(atoms)):
        gray = counter ^ (counter >> 1)
        changed = gray ^ previous_gray
        value ^= atoms[changed.bit_length() - 1]
        weight = gray.bit_count()
        if weight < distance[value]:
            distance[value] = weight
            leader[value] = gray
        previous_gray = gray

    queries = _rank_one_queries(rows, columns)
    nonzero = queries[1:]
    query_distances = [distance[query] for query in queries]
    activity = [0] * len(atoms)
    for query in nonzero:
        selected = leader[query]
        for index in range(len(atoms)):
            activity[index] += (selected >> index) & 1
    expected_union = sum(
        1.0 - (1.0 - count / len(nonzero)) ** query_count
        for count in activity
    )
    return {
        "rows": rows,
        "columns": columns,
        "ambient_dimension": dimension,
        "stored_atoms": len(atoms),
        "redundancy_fraction": (len(atoms) - dimension) / dimension,
        "atoms_hex": [hex(atom) for atom in atoms],
        "nonzero_rank_one_queries": len(nonzero),
        "maximum_minimum_atoms": max(query_distances),
        "mean_minimum_atoms_including_zero": (
            sum(query_distances) / len(query_distances)
        ),
        "distance_histogram": {
            str(item): query_distances.count(item)
            for item in sorted(set(query_distances))
        },
        "canonical_expected_union_atoms": expected_union,
        "canonical_expected_union_fraction": expected_union / len(atoms),
        "query_count": query_count,
        "is_general_lower_bound": False,
    }


def exhaustive_two_by_two_best(stored_atoms: int) -> dict[str, object]:
    """Find the exact lexicographic optimum for the 2x2 toy only."""

    if not 4 <= stored_atoms <= 15:
        raise ValueError("invalid 2x2 dictionary size")
    best: dict[str, object] | None = None
    best_score: tuple[int, float] | None = None
    tested = 0
    for atoms in combinations(range(1, 16), stored_atoms):
        if _gf2_rank(atoms, 4) != 4:
            continue
        tested += 1
        result = small_dictionary_witness(2, 2, atoms)
        score = (
            int(result["maximum_minimum_atoms"]),
            float(result["mean_minimum_atoms_including_zero"]),
        )
        if best_score is None or score < best_score:
            best, best_score = result, score
    if best is None:
        raise AssertionError("no spanning dictionary found")
    best["full_rank_dictionaries_tested"] = tested
    best["search"] = "EXHAUSTIVE"
    return best


def derive_audit() -> dict[str, object]:
    full_ambient = aggregate_full_ambient_materialization()
    identity = query_specialized_identity_batch()
    two_by_two = exhaustive_two_by_two_best(5)
    two_by_three = small_dictionary_witness(
        2, 3, (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3F)
    )
    three_by_three = small_dictionary_witness(
        3,
        3,
        (0x001, 0x002, 0x004, 0x008, 0x010, 0x020,
         0x036, 0x040, 0x080, 0x100, 0x1EF),
    )
    return {
        "classification": "E0_EXTENSION_FIELD_RANK_SATURATING_FRONTIER",
        "contract": {
            "registered_parameters": REGISTERED_PARAMETERS,
            "global_advice_bits": GLOBAL_ADVICE_BITS,
            "block_tokens": REGISTERED_BLOCK_TOKENS,
            "block_allowance_bytes": REGISTERED_BLOCK_ALLOWANCE_BYTES,
            "favorable_shared_bytes_per_second": (
                FAVORABLE_SHARED_BYTES_PER_SECOND
            ),
            "binary_screen_only": True,
        },
        "exact_repacking": {
            "rank_one_mask": (
                "r*u^T becomes (u_j*alpha)_j in GF(2^m)^k"
            ),
            "answer": (
                "<r*u^T,W>_GF2 = Trace(alpha * sum_j u_j beta_j)"
            ),
            "coefficient_rank_is_probe_support": False,
        },
        "published_full_ambient_rho_one": full_ambient,
        "query_specialized_identity_system": identity,
        "tiny_nonlinear_coset_witnesses": {
            "two_by_two_exact_optimum_at_five_atoms": two_by_two,
            "two_by_three_explicit_witness_not_optimality_claim": (
                two_by_three
            ),
            "three_by_three_prototype_witness_not_optimality_claim": (
                three_by_three
            ),
            "prototype_commit": "2f68300",
            "interpretation": (
                "SINGLE_QUERY_LEADERS_IMPROVE_BUT_32_QUERY_CANONICAL_"
                "UNION_IS_NEAR_THE_COMPLETE_TINY_DICTIONARY"
            ),
        },
        "claim_boundary": {
            "extension_field_trace_repacking_exact": True,
            "published_rho_is_number_of_probed_cells": False,
            "direct_full_ambient_materialization_fits": False,
            "query_specialized_direct_layout_meets_latency": False,
            "tiny_witness_is_asymptotic_lower_bound": False,
            "nonlinear_minimum_weight_decoder_rejected": False,
            "native_numerical_lift": False,
            "target_candidate": False,
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
        "decisions": [
            "REJECT_RANK_PARAMETER_AS_PROBE_COUNT",
            "REJECT_FULL_AMBIENT_RHO_ONE_DIRECT_MATERIALIZATION",
            "REJECT_QUERY_SPECIALIZED_IDENTITY_DIRECT_LAYOUT",
            "KEEP_QUERY_DEPENDENT_NONLINEAR_COSET_DECODER_OPEN",
            "REQUIRE_SMALL_32_QUERY_UNION_NOT_ONLY_SMALL_SINGLE_QUERY_LEADERS",
            "NO_SURVIVING_CANDIDATE",
        ],
        "decision": DECISION,
    }
