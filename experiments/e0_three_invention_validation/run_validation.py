#!/usr/bin/env python3
"""Deterministic E0 validation of CSPE, STC, and NRM.

This audit intentionally reuses already committed authoritative evidence and
adds only finite counting / adversarial controls. It does not execute a new
checkpoint and does not claim a universal lower bound beyond the declared
mechanism interfaces.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

TARGET_FRACTION = 8.0 / 675.0
HIDDEN_WIDTH = 16_384
QUERY_SUPPORT_BUDGET = math.floor(HIDDEN_WIDTH * TARGET_FRACTION)

# Frozen evidence values copied from the authoritative committed JSON sources.
EVIDENCE: dict[str, Any] = {
    "exp_086a": {
        "path": "results/exp_086a/fe457a464e465ecd561e352b7af637a90e17ced9/result.json",
        "blob_sha": "9d902bd0331afd0f7d3bd94d06b397fd422882fe",
        "coordinate_k128_input_residual_nonzero_p50": 0.8006342957130359,
        "coordinate_k128_down_residual_nonzero_p50": 0.8935469980314961,
        "coordinate_k128_operation_fraction_p50": 0.20239988251214963,
        "coordinate_k128_weight_fraction_p50": 0.006348049854281639,
        "reconstruction_mismatches": 0,
    },
    "exp_069": {
        "path": "results/exp_069/summary.json",
        "blob_sha": "1462229e99377d63ad7f32e7b76e80edd51abb5d",
        "p50_mandatory_operation_fraction": 1.0,
        "p90_mandatory_operation_fraction": 1.0,
        "verified_exact_replay_hits": 0,
    },
    "exp_049": {
        "path": "results/exp_049/summary.json",
        "blob_sha": "fb004723629f99893748c7e6b9545a234b26d49a",
        "triangular_one_position_per_round_barrier": True,
        "triangular_transcript_indistinguishability": True,
        "oracle_p50_matching_prefix_after_four_passes": 4.5,
        "oracle_max_matching_prefix_after_four_passes": 6.0,
    },
    "exp_080a": {
        "path": "results/exp_080a/summary.json",
        "blob_sha": "83e278b76d4348754b728ac3e0fb908441855d37",
        "best_constructive_arithmetic_fraction": 0.3611157967135951,
        "best_constructive_block_length": 16_384,
        "constructive_joint_pass_count": 0,
    },
    "native_shortcut": {
        "path": "results/e0_native_exact_shortcut_frontier/summary.json",
        "blob_sha": "f8bbf7181ef1e71c3f7999094ad97e3082b6b2f1",
        "unlimited_same_coordinate_history_hit_fraction": 0.029854910714285716,
        "best_combined_local_elimination_upper": 0.08677455357142855,
        "unique_product_fraction_p50": 0.9782366071428571,
    },
}


def cspe_coverage_upper_bits(n: int, k: int, s: int, r: int) -> float:
    """Upper-bound log2 count covered by <=s dictionary and <=r residual supports.

    For binary activations, sum_{i<=s} C(k,i) <= (e*k/s)^s and similarly
    for coordinate residuals. For a fixed support, the real span has dimension
    at most s+r and therefore intersects the binary hypercube in at most
    2^(s+r) points. This is deliberately favorable because it ignores collisions.
    """
    if s < 0 or r < 0 or s + r > n:
        raise ValueError("invalid supports")
    dict_bits = 0.0 if s == 0 else s * math.log2(math.e * k / s)
    residual_bits = 0.0 if r == 0 else r * math.log2(math.e * n / r)
    # For each fixed support, the resulting real subspace has dimension at most
    # s+r and intersects the binary hypercube in at most 2^(s+r) points.
    coefficient_bits = s + r
    return dict_bits + residual_bits + coefficient_bits


def cspe_minimum_log2_atoms(n: int, support_budget: int) -> dict[str, float | int]:
    """Necessary atom-count lower bound for covering every binary activation."""
    best: tuple[float, int, int] | None = None
    for s in range(1, support_budget + 1):
        r = support_budget - s
        residual_bits = 0.0 if r == 0 else r * math.log2(math.e * n / r)
        # s*log2(e*k/s) + residual_bits + (s+r) >= n
        coefficient_bits = s + r
        log2_k = (
            (n - residual_bits - coefficient_bits) / s
            - math.log2(math.e / s)
        )
        candidate = (log2_k, s, r)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {
        "minimum_log2_atom_count_lower_bound": best[0],
        "best_dictionary_support": best[1],
        "best_coordinate_residual_support": best[2],
    }


def stc_pointer_chasing_control(k: int) -> dict[str, int | bool]:
    """Finite adversarial control for a causal transition chain.

    A black-box transition H is exposed only at queried states. An adaptive
    chain s_{t+1}=H(s_t) has an indistinguishable completion at every unqueried
    successor. Therefore fewer than k sequential path queries cannot determine
    H^k(s0) for every H. The control explicitly builds the two completions.
    """
    if k < 1:
        raise ValueError("k must be positive")
    # Common transcript: 0 -> 1 -> ... -> k-1. The kth edge is not queried.
    common = {i: i + 1 for i in range(k - 1)}
    completion_a = dict(common)
    completion_b = dict(common)
    completion_a[k - 1] = k
    completion_b[k - 1] = k + 1
    same_before_final_query = all(
        completion_a[i] == completion_b[i] for i in range(k - 1)
    )
    different_kth_state = completion_a[k - 1] != completion_b[k - 1]
    return {
        "k": k,
        "common_sequential_queries": k - 1,
        "same_transcript_before_final_query": same_before_final_query,
        "different_kth_state": different_kth_state,
        "universal_queries_required": k,
    }


def largest_registered_subtree_cover(
    build: list[list[int]], query: list[int], sizes: tuple[int, ...]
) -> tuple[set[int], set[int]]:
    """NRM toy oracle: exact same-position patterns, largest first."""
    n = len(query)
    if any(len(row) != n for row in build):
        raise ValueError("shape mismatch")

    scalar_hits = {
        j for j, value in enumerate(query) if any(row[j] == value for row in build)
    }
    covered: set[int] = set()
    for width in sorted(sizes, reverse=True):
        if width <= 0:
            raise ValueError("nonpositive subtree")
        for start in range(0, n - width + 1, width):
            indexes = range(start, start + width)
            if any(j in covered for j in indexes):
                continue
            pattern = query[start : start + width]
            if any(row[start : start + width] == pattern for row in build):
                covered.update(indexes)
    return covered, scalar_hits


def nrm_subset_control() -> dict[str, Any]:
    build = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 9, 3, 0, 5, 0, 7, 0],
    ]
    query = [1, 2, 3, 0, 5, 6, 7, 1]
    covered, scalar_hits = largest_registered_subtree_cover(
        build, query, sizes=(8, 4, 2, 1)
    )
    return {
        "macro_covered_indexes": sorted(covered),
        "scalar_history_hit_indexes": sorted(scalar_hits),
        "macro_cover_subset_of_scalar_hits": covered <= scalar_hits,
    }


def build_summary() -> dict[str, Any]:
    target = TARGET_FRACTION
    required_elimination = 1.0 - target

    # Give the entire 8 GiB to one square binary Z=WP sidecar.
    # A 16,384-bit output column permits at most 2^25 atoms.
    favorable_one_bit_atom_cap = 2**25
    max_coverage = (-1.0, -1, -1)
    for s in range(QUERY_SUPPORT_BUDGET + 1):
        r = QUERY_SUPPORT_BUDGET - s
        bits = cspe_coverage_upper_bits(
            HIDDEN_WIDTH, favorable_one_bit_atom_cap, s, r
        )
        if bits > max_coverage[0]:
            max_coverage = (bits, s, r)

    cspe_count = cspe_minimum_log2_atoms(
        HIDDEN_WIDTH, QUERY_SUPPORT_BUDGET
    )
    cspe_empirical = EVIDENCE["exp_086a"]
    cspe_operation_miss = (
        cspe_empirical["coordinate_k128_operation_fraction_p50"] / target
    )

    stc = EVIDENCE["exp_080a"]
    stc_constructive_miss = (
        stc["best_constructive_arithmetic_fraction"] / target
    )
    pointer_controls = [
        stc_pointer_chasing_control(k) for k in (2, 4, 8, 16, 85)
    ]

    nrm = EVIDENCE["native_shortcut"]
    nrm_same_position_remaining = (
        1.0 - nrm["unlimited_same_coordinate_history_hit_fraction"]
    )
    nrm_same_position_miss = nrm_same_position_remaining / target
    nrm_local_remaining = 1.0 - nrm["best_combined_local_elimination_upper"]
    nrm_local_miss = nrm_local_remaining / target
    nrm_control = nrm_subset_control()

    summary: dict[str, Any] = {
        "schema": "vortex-e0-three-invention-validation-v1",
        "target": {
            "p50_fraction": target,
            "required_elimination_fraction": required_elimination,
            "registered_hidden_width": HIDDEN_WIDTH,
            "maximum_query_terms_before_all_overhead": QUERY_SUPPORT_BUDGET,
        },
        "cspe": {
            "identity_valid": True,
            "universal_binary_counting_gate": {
                **cspe_count,
                "minimum_atom_count_scientific_notation": (
                    f"2^{cspe_count['minimum_log2_atom_count_lower_bound']:.6f}"
                ),
                "favorable_8gib_one_bit_atom_cap": favorable_one_bit_atom_cap,
                "maximum_covered_information_bits_under_cap": max_coverage[0],
                "ambient_information_bits": HIDDEN_WIDTH,
                "coverage_information_fraction_upper": (
                    max_coverage[0] / HIDDEN_WIDTH
                ),
                "best_support_split_under_cap": {
                    "dictionary": max_coverage[1],
                    "residual": max_coverage[2],
                },
                "gate_pass": False,
            },
            "restricted_real_checkpoint_evidence": {
                "input_residual_nonzero_p50": cspe_empirical[
                    "coordinate_k128_input_residual_nonzero_p50"
                ],
                "down_residual_nonzero_p50": cspe_empirical[
                    "coordinate_k128_down_residual_nonzero_p50"
                ],
                "operation_fraction_p50": cspe_empirical[
                    "coordinate_k128_operation_fraction_p50"
                ],
                "operation_miss_factor": cspe_operation_miss,
                "weight_fraction_p50": cspe_empirical[
                    "coordinate_k128_weight_fraction_p50"
                ],
                "reconstruction_mismatches": cspe_empirical[
                    "reconstruction_mismatches"
                ],
                "temporal_span_mandatory_operation_p50": EVIDENCE["exp_069"][
                    "p50_mandatory_operation_fraction"
                ],
                "temporal_exact_replay_hits": EVIDENCE["exp_069"][
                    "verified_exact_replay_hits"
                ],
            },
            "decision": (
                "REJECT_CURRENT_CSPE_AS_UNIVERSAL_CORE_"
                "KEEP_ONLY_CONCRETE_DISTRIBUTION_SPECIFIC_NONLINEAR_CODE_GATE_OPEN"
            ),
            "scope": (
                "Rejects the stated fixed-dictionary sparse-code core and its "
                "registered linear/state-axis realizations. It does not prove "
                "that every checkpoint-specific nonlinear implicit code is impossible."
            ),
        },
        "stc": {
            "relation_formulation_valid": True,
            "pointer_chasing_controls": pointer_controls,
            "causal_real_checkpoint_evidence": {
                "triangular_one_position_per_round_barrier": EVIDENCE["exp_049"][
                    "triangular_one_position_per_round_barrier"
                ],
                "triangular_transcript_indistinguishability": EVIDENCE["exp_049"][
                    "triangular_transcript_indistinguishability"
                ],
                "oracle_p50_prefix_after_four_passes": EVIDENCE["exp_049"][
                    "oracle_p50_matching_prefix_after_four_passes"
                ],
                "oracle_max_prefix_after_four_passes": EVIDENCE["exp_049"][
                    "oracle_max_matching_prefix_after_four_passes"
                ],
            },
            "perfect_future_constructive_arithmetic": {
                "best_fraction": stc[
                    "best_constructive_arithmetic_fraction"
                ],
                "miss_factor": stc_constructive_miss,
                "block_length": stc["best_constructive_block_length"],
                "joint_pass_count": stc["constructive_joint_pass_count"],
            },
            "decision": (
                "REJECT_CURRENT_STC_AS_REORDERING_NOT_SUBDENSE_EXECUTOR_"
                "RETAIN_FACTOR_GRAPH_REFERENCE_ONLY"
            ),
            "scope": (
                "Rejects the current variable-elimination description as a core: "
                "it supplies neither a compact causal path source nor a constructive "
                "subdense contraction. It does not reject a future explicit algorithm "
                "with a new information source."
            ),
        },
        "nrm": {
            "exact_macro_semantics_valid_on_hit": True,
            "same_position_macro_coverage_lemma_control": nrm_control,
            "real_checkpoint_free_oracle_upper": {
                "unlimited_scalar_history_elimination_upper": nrm[
                    "unlimited_same_coordinate_history_hit_fraction"
                ],
                "same_position_macro_remaining_fraction_lower": (
                    nrm_same_position_remaining
                ),
                "same_position_macro_miss_factor": nrm_same_position_miss,
                "best_combined_local_elimination_upper": nrm[
                    "best_combined_local_elimination_upper"
                ],
                "best_combined_local_remaining_fraction_lower": nrm_local_remaining,
                "best_combined_local_miss_factor": nrm_local_miss,
                "unique_product_fraction_p50": nrm[
                    "unique_product_fraction_p50"
                ],
            },
            "favorable_table_bound": {
                "output_width": HIDDEN_WIDTH,
                "bytes_per_bf16_partial_vector": 2 * HIDDEN_WIDTH,
                "maximum_macros_if_all_8gib_given_to_one_matrix": (
                    (8 * 2**30) // (2 * HIDDEN_WIDTH)
                ),
                "average_size1_values_per_coordinate": (
                    ((8 * 2**30) // (2 * HIDDEN_WIDTH)) / HIDDEN_WIDTH
                ),
                "bf16_value_count": 2**16,
            },
            "decision": "REJECT_NRM_EXACT_PATTERN_MACROS_AS_CORE",
            "scope": (
                "Rejects same-position exact native subtree-pattern macros. "
                "Cross-position algebraic transforms are a different mechanism."
            ),
        },
        "overall": {
            "surviving_core_candidate_count": 0,
            "decision": "NO_THREE_INVENTION_CORE_SURVIVES_E0_EVIDENCE_AUDIT",
            "next_requirement": (
                "A new candidate must provide a concrete causal information source, "
                "a finite-word executable operation, complete successor-state semantics, "
                "and a fully charged path below the target before another model run."
            ),
        },
        "evidence": EVIDENCE,
        "claim_boundary": {
            "new_checkpoint_forward": False,
            "new_hardware_measurement": False,
            "405b_execution": False,
            "universal_impossibility": False,
            "evidence_level": "E0 finite proof plus reuse of E1/E2 committed evidence",
        },
    }
    core = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    summary["deterministic_core_sha256"] = hashlib.sha256(core).hexdigest()
    return summary


def main() -> int:
    summary = build_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
