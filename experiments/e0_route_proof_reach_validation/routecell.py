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
# ROUTECELL Gate
# ---------------------------------------------------------------------------


def rank_one_masks_2x3() -> list[int]:
    masks: list[int] = []
    for left in range(1, 1 << 2):
        for right in range(1, 1 << 3):
            mask = 0
            for row in range(2):
                for column in range(3):
                    if ((left >> row) & 1) and ((right >> column) & 1):
                        mask |= 1 << (row * 3 + column)
            masks.append(mask)
    if len(masks) != 21 or len(set(masks)) != 21:
        raise AssertionError("2x3 rank-one mask registration failed")
    return masks


def _encoded_coordinate_cells() -> list[int]:
    cells: list[int] = []
    for coordinate in range(6):
        ones = 0
        for encoded_word in range(64):
            if (encoded_word >> coordinate) & 1:
                ones |= 1 << encoded_word
        cells.append(ones)
    return cells


def _linear_target_truth(mask: int) -> int:
    truth = 0
    for encoded_word in range(64):
        if _parity(mask & encoded_word):
            truth |= 1 << encoded_word
    return truth


def _constant_on(target: int, state_set: int) -> bool:
    selected = target & state_set
    return selected == 0 or selected == state_set


def _depth_one(target: int, state_set: int, cells: Sequence[int]) -> bool:
    if _constant_on(target, state_set):
        return True
    for cell in cells:
        branch_one = state_set & cell
        branch_zero = state_set & (~cell) & ALL64
        if _constant_on(target, branch_zero) and _constant_on(target, branch_one):
            return True
    return False


def _depth_two(target: int, cells: Sequence[int]) -> bool:
    """Return whether an arbitrary deterministic adaptive two-read decoder exists."""
    if _depth_one(target, ALL64, cells):
        return True
    for first_cell in cells:
        branch_one = ALL64 & first_cell
        branch_zero = ALL64 & (~first_cell) & ALL64
        if _depth_one(target, branch_zero, cells) and _depth_one(target, branch_one, cells):
            return True
    return False


def _minimum_query_depth(target: int, cells: Sequence[int]) -> int:
    memo: dict[tuple[int, int], int] = {}

    def solve(state_set: int, available: int) -> int:
        key = (state_set, available)
        if key in memo:
            return memo[key]
        if _constant_on(target, state_set):
            memo[key] = 0
            return 0
        best = 100
        for index, cell in enumerate(cells):
            if not ((available >> index) & 1):
                continue
            branch_one = state_set & cell
            branch_zero = state_set & (~cell) & ALL64
            if branch_zero == 0 or branch_one == 0:
                continue
            candidate = 1 + max(
                solve(branch_zero, available ^ (1 << index)),
                solve(branch_one, available ^ (1 << index)),
            )
            best = min(best, candidate)
        if best == 100:
            raise AssertionError("non-constant target had no separating cell")
        memo[key] = best
        return best

    return solve(ALL64, (1 << len(cells)) - 1)


def _affine_round_table(linear_code: int, offset: int = 0) -> list[int]:
    rows = (
        (linear_code >> 0) & 0b111,
        (linear_code >> 3) & 0b111,
        (linear_code >> 6) & 0b111,
    )
    table: list[int] = []
    for right in range(8):
        value = 0
        for output_bit, row_mask in enumerate(rows):
            bit = _parity(row_mask & right) ^ ((offset >> output_bit) & 1)
            value |= bit << output_bit
        table.append(value)
    return table


def _feistel_encode(source: int, round_table: Sequence[int], rounds: int) -> int:
    left = source & 0b111
    right = (source >> 3) & 0b111
    for _ in range(rounds):
        left, right = right, left ^ round_table[right]
    return left | (right << 3)


def _transformed_query_masks(linear_code: int, rounds: int, offset: int = 0) -> list[int]:
    table = _affine_round_table(linear_code, offset)
    inverse = [0] * 64
    for source in range(64):
        encoded = _feistel_encode(source, table, rounds)
        inverse[encoded] = source
    if len(set(inverse)) != 64:
        raise AssertionError("Feistel encoder must be a permutation")

    transformed: list[int] = []
    for source_mask in rank_one_masks_2x3():
        constant = _parity(source_mask & inverse[0])
        encoded_mask = 0
        for coordinate in range(6):
            coefficient = _parity(source_mask & inverse[1 << coordinate]) ^ constant
            encoded_mask |= coefficient << coordinate
        # Exact linearity/permutation control over the complete finite domain.
        for encoded in range(64):
            expected = _parity(source_mask & inverse[encoded])
            observed = _parity(encoded_mask & encoded) ^ constant
            if expected != observed:
                raise AssertionError("affine Feistel query transform mismatch")
        transformed.append(encoded_mask)
    return transformed


def _registered_three_input_tags(base_cells: Sequence[int]) -> list[tuple[int, tuple[int, int, int], int]]:
    """All unique one-bit partitions generated by any Boolean function of <=3 cells.

    Every 3-input truth table is enumerated.  A tag and its complement expose the
    same partition to a query decoder, so complements are canonicalized.
    """
    unique: dict[int, tuple[tuple[int, int, int], int]] = {}
    for indices in combinations(range(6), 3):
        for truth_table in range(256):
            tag = 0
            for encoded_word in range(64):
                local_index = (
                    ((encoded_word >> indices[0]) & 1)
                    | (((encoded_word >> indices[1]) & 1) << 1)
                    | (((encoded_word >> indices[2]) & 1) << 2)
                )
                if (truth_table >> local_index) & 1:
                    tag |= 1 << encoded_word
            canonical = min(tag, ALL64 ^ tag)
            unique.setdefault(canonical, (indices, truth_table))
    return [(tag, description[0], description[1]) for tag, description in unique.items()]


def validate_routecell() -> dict[str, Any]:
    base_cells = _encoded_coordinate_cells()
    linear_truth = {mask: _linear_target_truth(mask) for mask in range(1, 64)}
    tags = _registered_three_input_tags(base_cells)

    tag_good_masks: list[tuple[int, tuple[int, int, int], int, int]] = []
    for tag, indices, truth_table in tags:
        cells = [*base_cells, tag]
        good_mask = 0
        for target_mask, target_truth in linear_truth.items():
            if _depth_two(target_truth, cells):
                good_mask |= 1 << target_mask
        tag_good_masks.append((good_mask, indices, truth_table, tag))

    best_coverage = -1
    best_record: tuple[int, int, tuple[int, int, int], int, list[int], int] | None = None
    full_passes = 0
    per_encoder_best: Counter[int] = Counter()
    offset_invariance_failures = 0

    for linear_code in range(512):
        for rounds in (1, 2, 3):
            transformed = _transformed_query_masks(linear_code, rounds, offset=0)
            query_set = 0
            for mask in transformed:
                query_set |= 1 << mask
            if query_set.bit_count() != 21:
                raise AssertionError("rank-one query masks collapsed under permutation")

            # Affine offsets alter only output constants, not query partitions/depth.
            for offset in range(1, 8):
                if _transformed_query_masks(linear_code, rounds, offset) != transformed:
                    offset_invariance_failures += 1

            local_best = -1
            for good_mask, indices, truth_table, tag in tag_good_masks:
                coverage = (good_mask & query_set).bit_count()
                local_best = max(local_best, coverage)
                if coverage == 21:
                    full_passes += 1
                candidate_key = (linear_code, rounds, indices, truth_table)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_record = (linear_code, rounds, indices, truth_table, transformed, tag)
                elif coverage == best_coverage and best_record is not None:
                    prior_key = (best_record[0], best_record[1], best_record[2], best_record[3])
                    if candidate_key < prior_key:
                        best_record = (linear_code, rounds, indices, truth_table, transformed, tag)
            per_encoder_best[local_best] += 1

    if best_record is None:
        raise AssertionError("ROUTECELL search emitted no candidate")

    raw_coverage = sum(
        _depth_two(_linear_target_truth(mask), base_cells)
        for mask in rank_one_masks_2x3()
    )
    _, _, best_indices, best_truth_table, best_masks, best_tag = best_record
    best_cells = [*base_cells, best_tag]
    best_depths = [
        _minimum_query_depth(linear_truth[mask], best_cells)
        for mask in best_masks
    ]
    depth_histogram = {str(depth): best_depths.count(depth) for depth in sorted(set(best_depths))}

    return {
        "candidate": "ROUTECELL",
        "registered_scope": {
            "source": "all 2x3 binary matrices",
            "query_family": "all 21 nonzero GF(2) rank-one parities",
            "encoder": "3+3 Feistel; repeated identical affine 3-bit round; 1/2/3 rounds",
            "linear_round_maps": 512,
            "affine_offsets_covered": 8,
            "route_encoders": 512 * 3,
            "redundancy": "one extra route tag bit",
            "tag_grammar": "every Boolean function of any <=3 encoded bits",
            "unique_tag_partitions": len(tags),
            "decoder": "arbitrary exact deterministic adaptive decision tree",
            "probe_budget": 2,
        },
        "integrity": {
            "offset_invariance_failures": offset_invariance_failures,
            "complete_encoder_tag_pairs": 512 * 3 * len(tags),
            "positive_one_coordinate_depth": _minimum_query_depth(linear_truth[1], base_cells),
            "positive_two_coordinate_parity_depth": _minimum_query_depth(linear_truth[3], base_cells),
        },
        "results": {
            "raw_coordinate_coverage": raw_coverage,
            "raw_coordinate_fraction": raw_coverage / 21,
            "full_two_probe_encoders": full_passes,
            "best_two_probe_coverage": best_coverage,
            "best_two_probe_fraction": best_coverage / 21,
            "best_encoder_linear_code": best_record[0],
            "best_encoder_rounds": best_record[1],
            "best_tag_input_coordinates": list(best_indices),
            "best_tag_truth_table": best_truth_table,
            "best_exact_depth_histogram": depth_histogram,
            "best_worst_query_depth": max(best_depths),
            "per_encoder_best_coverage_histogram": {
                str(key): per_encoder_best[key] for key in sorted(per_encoder_best)
            },
        },
        "decision": "REJECT_REGISTERED_AFFINE_FEISTEL_SINGLE_TAG_ROUTECELL_GRAMMAR",
        "claim_boundary": [
            "does not reject non-affine/multiword/global adaptive nonlinear cold encodings",
            "does not construct or reject the 25x108/50-word/two-probe frontier object",
            "does not test native Q4/BF16/FP32 partial reductions or physical I/O",
        ],
    }


