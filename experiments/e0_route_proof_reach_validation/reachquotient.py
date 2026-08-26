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
# REACHQUOTIENT Gate
# ---------------------------------------------------------------------------


def minimize_moore_machine(
    outputs: Sequence[int], transitions: Sequence[Sequence[int]], alphabet_size: int
) -> dict[str, Any]:
    state_count = len(outputs)
    initial: dict[int, list[int]] = {}
    for state, output in enumerate(outputs):
        initial.setdefault(output, []).append(state)
    blocks = list(initial.values())
    class_of = [0] * state_count
    for class_index, block in enumerate(blocks):
        for state in block:
            class_of[state] = class_index

    rounds = 0
    while True:
        grouped: dict[tuple[int, ...], list[int]] = {}
        for state in range(state_count):
            signature = (outputs[state],) + tuple(
                class_of[transitions[state][symbol]] for symbol in range(alphabet_size)
            )
            grouped.setdefault(signature, []).append(state)
        new_blocks = list(grouped.values())
        new_class_of = [0] * state_count
        for class_index, block in enumerate(new_blocks):
            for state in block:
                new_class_of[state] = class_index
        rounds += 1
        if new_class_of == class_of:
            return {
                "classes": len(new_blocks),
                "rounds": rounds,
                "class_of": new_class_of,
            }
        class_of = new_class_of


def _shift_register_machine(width: int) -> tuple[list[int], list[list[int]]]:
    state_count = 1 << width
    state_mask = state_count - 1
    outputs = [(state >> (width - 1)) & 1 for state in range(state_count)]
    transitions = [
        [((state << 1) & state_mask) | symbol for symbol in (0, 1)]
        for state in range(state_count)
    ]
    return outputs, transitions


def _duplicated_shift_machine(width: int, copies: int) -> tuple[list[int], list[list[int]]]:
    base_outputs, base_transitions = _shift_register_machine(width)
    outputs: list[int] = []
    transitions: list[list[int]] = []
    for base_state in range(1 << width):
        for copy in range(copies):
            outputs.append(base_outputs[base_state])
            transitions.append(
                [base_transitions[base_state][symbol] * copies + copy for symbol in (0, 1)]
            )
    return outputs, transitions


def _random_machine(state_count: int, seed: int) -> tuple[list[int], list[list[int]]]:
    generator = random.Random(seed)
    outputs = [generator.randrange(2) for _ in range(state_count)]
    transitions = [
        [generator.randrange(state_count), generator.randrange(state_count)]
        for _ in range(state_count)
    ]
    return outputs, transitions


def validate_reachquotient() -> dict[str, Any]:
    positive_outputs, positive_transitions = _duplicated_shift_machine(width=3, copies=4)
    positive = minimize_moore_machine(positive_outputs, positive_transitions, 2)

    shift_rows: list[dict[str, Any]] = []
    for width in range(2, 9):
        outputs, transitions = _shift_register_machine(width)
        result = minimize_moore_machine(outputs, transitions, 2)
        shift_rows.append(
            {
                "width": width,
                "states": 1 << width,
                "classes": result["classes"],
                "class_fraction": result["classes"] / (1 << width),
                "refinement_rounds": result["rounds"],
            }
        )

    random_rows: list[dict[str, Any]] = []
    for state_count in (16, 32, 64, 128):
        class_counts: list[int] = []
        refinement_rounds: list[int] = []
        for sample in range(100):
            outputs, transitions = _random_machine(
                state_count, seed=100_000 + state_count * 1_000 + sample
            )
            result = minimize_moore_machine(outputs, transitions, 2)
            class_counts.append(result["classes"])
            refinement_rounds.append(result["rounds"])
        random_rows.append(
            {
                "states": state_count,
                "samples": 100,
                "minimum_classes": min(class_counts),
                "p50_classes": statistics.median(class_counts),
                "p95_classes": _quantile_nearest_rank(class_counts, 0.95),
                "p50_class_fraction": statistics.median(
                    [count / state_count for count in class_counts]
                ),
                "full_quotient_fraction": sum(count == state_count for count in class_counts) / 100,
                "maximum_refinement_rounds": max(refinement_rounds),
            }
        )

    # Contract-weakening control: restricting future actions can merge states
    # that are distinguishable under the required all-legal-suffix contract.
    weak_outputs = [0, 0, 0, 1]
    weak_transitions = [[2, 2], [2, 3], [2, 2], [3, 3]]
    full_suffix = minimize_moore_machine(weak_outputs, weak_transitions, 2)
    fixed_zero_suffix = minimize_moore_machine(
        weak_outputs, [[row[0]] for row in weak_transitions], 1
    )

    real_checkpoint_evidence = {
        "source_path": "docs/research/EXP_089A_LATEST_RESULT.md",
        "source_ref": "f5a432a5152ee6d9e1399583a4829a39aacdf5e0",
        "source_blob_sha": "aed26836718caf953949599361c0df66cd7a1709",
        "checkpoint": "HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "compiled_state": "exact prompt/generated token prefix + exact RNG bytes",
        "transitions": 64,
        "token_mismatches": 0,
        "logits_byte_mismatches": 0,
        "cache_byte_or_digest_mismatches": 0,
        "rng_state_mismatches": 0,
        "reference_target_calls": 68,
        "replay_target_calls": 612,
        "replayed_token_positions": 3791,
        "identity_replay_is_fast_executor": False,
    }

    return {
        "candidate": "REACHQUOTIENT",
        "registered_scope": {
            "equivalence": "exact Moore/bisimulation equivalence under every legal future input symbol",
            "algorithm": "exact partition refinement",
            "random_population": "100 deterministic machines at each of 16/32/64/128 states",
        },
        "positive_control": {
            "states": len(positive_outputs),
            "classes": positive["classes"],
            "class_fraction": positive["classes"] / len(positive_outputs),
        },
        "shift_register_adversary": shift_rows,
        "random_machines": random_rows,
        "contract_weakening_control": {
            "all_legal_suffix_classes": full_suffix["classes"],
            "fixed_zero_only_classes": fixed_zero_suffix["classes"],
            "weaker_contract_merges_more": fixed_zero_suffix["classes"] < full_suffix["classes"],
        },
        "real_checkpoint_evidence": real_checkpoint_evidence,
        "results": {
            "shift_register_full_class_cases": sum(
                row["states"] == row["classes"] for row in shift_rows
            ),
            "shift_register_cases": len(shift_rows),
            "random_p50_class_fraction_all_sizes": [
                row["p50_class_fraction"] for row in random_rows
            ],
            "checkpoint_specific_H_lambda_delta_constructed": False,
            "all_suffix_congruence_proof_constructed": False,
        },
        "decision": "VALID_BISIMULATION_INTERFACE_NO_COMPRESSED_REACHABLE_QUOTIENT_CONSTRUCTOR",
        "claim_boundary": [
            "does not prove every pretrained Transformer has a trivial quotient",
            "does prove that exact quotienting can be either powerful or completely vacuous",
            "the existing real-checkpoint witness is identity prefix replay, not a compressed executor",
        ],
    }


