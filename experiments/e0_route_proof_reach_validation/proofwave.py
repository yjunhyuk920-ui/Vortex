#!/usr/bin/env python3
"""Deterministic E0/E1 validation for ROUTECELL, PROOFWAVE, and REACHQUOTIENT.

The suite is deliberately cheapest-kill-first.  It does not run a checkpoint or
claim a general impossibility theorem.  It exactly tests one registered
ROUTECELL grammar, exact finite lazy-decision controls for PROOFWAVE, and exact
Moore-machine quotient controls for REACHQUOTIENT.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Sequence

TARGET_NUMERATOR = 8
TARGET_DENOMINATOR = 675
TARGET_FRACTION = TARGET_NUMERATOR / TARGET_DENOMINATOR
ALL64 = (1 << 64) - 1


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _quantile_nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


# ---------------------------------------------------------------------------
# PROOFWAVE Gate
# ---------------------------------------------------------------------------


def _stopping_depths_for_threshold(bit_count: int, threshold: int) -> list[int]:
    depths: list[int] = []
    for assignment in range(1 << bit_count):
        ones = 0
        zeros = 0
        depth = 0
        for coordinate in range(bit_count):
            depth += 1
            if (assignment >> coordinate) & 1:
                ones += 1
            else:
                zeros += 1
            if ones >= threshold or zeros > bit_count - threshold:
                break
        depths.append(depth)
    return depths


def _depth_metrics(depths: Iterable[int]) -> dict[str, Any]:
    values = list(depths)
    return {
        "mean": sum(values) / len(values),
        "p50": _quantile_nearest_rank(values, 0.50),
        "p95": _quantile_nearest_rank(values, 0.95),
        "worst": max(values),
        "histogram": {str(key): value for key, value in sorted(Counter(values).items())},
    }


def validate_proofwave() -> dict[str, Any]:
    bit_count = 12
    first_coordinate = [1] * (1 << bit_count)
    or_depths = []
    parity_depths = [bit_count] * (1 << bit_count)
    for assignment in range(1 << bit_count):
        depth = 0
        for coordinate in range(bit_count):
            depth += 1
            if (assignment >> coordinate) & 1:
                break
        or_depths.append(depth)
    majority_depths = _stopping_depths_for_threshold(bit_count, threshold=6)

    controls = {
        "one_coordinate": _depth_metrics(first_coordinate),
        "easy_or_certificate": _depth_metrics(or_depths),
        "balanced_dense_margin": _depth_metrics(majority_depths),
        "parity_diagnostic_only": _depth_metrics(parity_depths),
    }
    for metrics in controls.values():
        metrics["p50_fraction"] = metrics["p50"] / bit_count
        metrics["p95_fraction"] = metrics["p95"] / bit_count

    real_checkpoint_evidence = {
        "source_path": "docs/research/EXP_096A_LATEST_RESULT.md",
        "source_ref": "f5a432a5152ee6d9e1399583a4829a39aacdf5e0",
        "source_blob_sha": "fe9247a2173e95a4ad64912913a34b38eb3403bf",
        "checkpoint": "HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "target_seeing_certificates_build": [384, 384],
        "target_seeing_certificates_holdout": [384, 384],
        "holdout_coefficient_support_p50": 5.0,
        "holdout_coefficient_support_p95": 13.0,
        "projected_residual_scoring_fraction_percent": 0.004085191,
        "projected_hot_state_gib": 7.554739,
        "perfect_block_compressed_traffic_percent": 0.619070788,
        "fine_dense_arithmetic_percent": 100.0,
        "causal_coefficient_generator": "NOT_CONSTRUCTED",
    }

    generator_plus_checker_identity = {
        "equation": "total = local_generator + proof_emission + independent_checker + state_update + fallback",
        "nonnegative_terms": True,
        "checker_cannot_remove_generator_work": True,
    }

    return {
        "candidate": "PROOFWAVE",
        "registered_scope": {
            "finite_control_bits": bit_count,
            "control_semantics": "zero-error adaptive source reads; free logic between reads",
            "balanced_control": "sign of an equal-magnitude 12-contribution logit difference",
            "positive_control": "one-coordinate and OR certificates",
        },
        "controls": controls,
        "real_checkpoint_evidence": real_checkpoint_evidence,
        "accounting": generator_plus_checker_identity,
        "results": {
            "balanced_margin_p50_fraction": controls["balanced_dense_margin"]["p50_fraction"],
            "balanced_margin_p95_fraction": controls["balanced_dense_margin"]["p95_fraction"],
            "balanced_margin_miss_factor_p50": controls["balanced_dense_margin"]["p50_fraction"] / TARGET_FRACTION,
            "balanced_margin_miss_factor_p95": controls["balanced_dense_margin"]["p95_fraction"] / TARGET_FRACTION,
            "local_causal_proof_generator_specified": False,
            "symbolic_successor_transition_program_specified": False,
        },
        "decision": "REJECT_PROOFWAVE_AS_STANDALONE_NEW_INFORMATION_SOURCE_RETAIN_CHECKER_AUXILIARY",
        "claim_boundary": [
            "does not reject every SAT/SMT/bit-vector proof search on a named checkpoint",
            "parity is a diagnostic control, not the primary scientific result",
            "a future candidate must specify and charge the causal local proof generator itself",
        ],
    }


