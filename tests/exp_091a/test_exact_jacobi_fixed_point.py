from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.exact_jacobi_fixed_point import (
    JacobiInvariantError,
    build_seed,
    changed_positions,
    exhaustive_toy_prefix_control,
    future_mutation_control,
    jacobi_input,
    logical_weight_fraction,
    required_tokens_per_sweep,
    run_triangular_toy,
    select_seed_mode,
    self_consistent_prefix,
    stable_argmax_rows,
)


LIMIT = 0.011851851851851851
RATIO = 1.261972


def test_boundary_repeat_seed_is_deterministic() -> None:
    first, first_manifest = build_seed(
        "boundary_repeat", prompt_ids=[1, 2, 3], boundary_token=9, block_length=8
    )
    second, second_manifest = build_seed(
        "boundary_repeat", prompt_ids=[1, 2, 3], boundary_token=9, block_length=8
    )
    assert first == (9,) * 8
    assert first == second
    assert first_manifest == second_manifest


def test_suffix_cycle_uses_only_committed_prompt_tail() -> None:
    seed, _ = build_seed(
        "prompt_suffix_cycle_16",
        prompt_ids=list(range(1, 21)),
        boundary_token=999,
        block_length=20,
    )
    source = tuple(range(5, 21))
    assert seed[:16] == source
    assert seed[16:] == source[:4]
    assert 999 not in seed


def test_longest_ngram_seed_replays_committed_pattern() -> None:
    prompt = [10, 20, 30, 10, 20, 30, 10, 20]
    seed, _ = build_seed(
        "prompt_longest_ngram_8",
        prompt_ids=prompt,
        boundary_token=30,
        block_length=9,
    )
    assert seed[:6] == (10, 20, 30, 10, 20, 30)
    assert len(seed) == 9


def test_invalid_seed_requests_fail_closed() -> None:
    with pytest.raises(JacobiInvariantError):
        build_seed("unknown", prompt_ids=[1], boundary_token=2, block_length=4)
    with pytest.raises(JacobiInvariantError):
        build_seed("boundary_repeat", prompt_ids=[], boundary_token=2, block_length=4)
    with pytest.raises(JacobiInvariantError):
        build_seed("boundary_repeat", prompt_ids=[1], boundary_token=-1, block_length=4)


def test_jacobi_input_alignment() -> None:
    assert jacobi_input(7, [11, 12, 13, 14]) == (7, 11, 12, 13)


def test_stable_argmax_rows_uses_smallest_token_id_on_ties() -> None:
    logits = torch.tensor(
        [[[1.0, 3.0, 3.0], [5.0, 5.0, 2.0]]], dtype=torch.float32
    )
    tokens = stable_argmax_rows(logits)
    assert tokens.tolist() == [[1, 0]]


def test_self_consistent_prefix_and_changes() -> None:
    guess = [1, 2, 9, 4]
    proposal = [1, 2, 3, 4]
    assert self_consistent_prefix(guess, proposal) == 2
    assert changed_positions(guess, proposal) == 1


def test_exact_traffic_thresholds_are_frozen() -> None:
    assert required_tokens_per_sweep(LIMIT) == 85
    assert required_tokens_per_sweep(LIMIT, compression_ratio=RATIO) == 67
    assert logical_weight_fraction(1, 85) <= LIMIT
    assert logical_weight_fraction(1, 84) > LIMIT
    assert logical_weight_fraction(1, 67, compression_ratio=RATIO) <= LIMIT
    assert logical_weight_fraction(2, 128) > LIMIT
    assert logical_weight_fraction(2, 128, compression_ratio=RATIO) > LIMIT
    assert math.isinf(logical_weight_fraction(1, 0))


def test_triangular_map_propagates_one_token_per_sweep_and_reaches_fixed_point() -> None:
    target = [1, 2, 3, 4, 5]
    seed = [101, 102, 103, 104, 105]
    rows = run_triangular_toy(seed, target, maximum_iterations=6)
    assert rows[0]["self_consistent_prefix"] == 0
    assert rows[1]["self_consistent_prefix"] == 1
    assert rows[2]["self_consistent_prefix"] == 2
    assert rows[5]["fixed_point"]
    assert rows[5]["guess"] == target


def test_exhaustive_prefix_theorem_control_passes() -> None:
    control = exhaustive_toy_prefix_control(8)
    assert control["checked"] == 256
    assert control["mismatches"] == 0
    assert control["passed"]


def test_future_mutation_cannot_change_earlier_toy_proposals() -> None:
    assert future_mutation_control()["passed"]


def test_seed_selection_prioritizes_first_sweep_population() -> None:
    rows = [
        {
            "mode": "a",
            "first_sweep_accepted": 10,
            "best_accepted": 100,
            "fixed_point_sweep": 8,
        },
        {
            "mode": "a",
            "first_sweep_accepted": 9,
            "best_accepted": 110,
            "fixed_point_sweep": 8,
        },
        {
            "mode": "b",
            "first_sweep_accepted": 11,
            "best_accepted": 20,
            "fixed_point_sweep": None,
        },
        {
            "mode": "b",
            "first_sweep_accepted": 11,
            "best_accepted": 20,
            "fixed_point_sweep": None,
        },
    ]
    selected = select_seed_mode(rows, ["a", "b"])
    assert selected.mode == "b"
    assert selected.minimum_first_sweep_accepted == 11


def test_selection_tie_preserves_frozen_mode_order() -> None:
    rows = []
    for mode in ("first", "second"):
        rows.append(
            {
                "mode": mode,
                "first_sweep_accepted": 3,
                "best_accepted": 4,
                "fixed_point_sweep": None,
            }
        )
    assert select_seed_mode(rows, ["first", "second"]).mode == "first"
