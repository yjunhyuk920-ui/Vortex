from __future__ import annotations

import numpy as np

from vortex_runtime.progressive_head_tournament import (
    activation_ordered_tournament_runtime,
    ceil_sqrt_int,
    exact_activation_ordered_tournament,
    full_q4_head_bytes,
    make_late_decision_control,
    make_structured_positive_control,
)


def test_ceil_sqrt_exact() -> None:
    values = np.array([0, 1, 2, 3, 4, 15, 16, 17, 10**12 + 1], dtype=np.int64)
    got = ceil_sqrt_int(values)
    assert np.array_equal(got, np.array([0, 1, 2, 2, 2, 4, 4, 5, 1000001]))


def test_full_q4_head_bytes_group128() -> None:
    assert full_q4_head_bytes(128256, 16384) == 1116340224.0


def test_structured_positive_control_stops_early() -> None:
    rng = np.random.default_rng(1)
    weights, activation, expected = make_structured_positive_control(rng, 512, 256, 32)
    row = exact_activation_ordered_tournament(weights, activation, 32)
    assert row.exact
    assert row.winner == expected
    assert row.stopped_early
    assert row.weight_element_fraction < 0.5


def test_random_dense_matches_full_argmax() -> None:
    rng = np.random.default_rng(2)
    weights = rng.integers(-8, 8, size=(512, 512), dtype=np.int8)
    activation = rng.integers(-127, 128, size=512, dtype=np.int16).astype(np.int8)
    for block_size in (32, 64, 128):
        row = exact_activation_ordered_tournament(weights, activation, block_size)
        assert row.exact


def test_late_decision_control_forces_large_scan() -> None:
    rng = np.random.default_rng(3)
    weights, activation = make_late_decision_control(rng, 1024, 512, 64)
    row = exact_activation_ordered_tournament(weights, activation, 64)
    assert row.exact
    assert row.weight_element_fraction > 0.85


def test_tie_break_is_lowest_id() -> None:
    weights = np.zeros((8, 64), dtype=np.int8)
    activation = np.ones(64, dtype=np.int8)
    row = exact_activation_ordered_tournament(weights, activation, 16)
    assert row.exact
    assert row.winner == 0


def test_causal_dependency_gate_is_serial() -> None:
    from experiments.exp_105a.run_experiment import causal_dependency_gate

    row = causal_dependency_gate(tokens=5, layers=3)
    assert row["minimum_serial_weight_stages"] == 20
    assert row["longest_dependency_path_nodes"] == 25


def test_runtime_winner_is_independent_of_reference_helper() -> None:
    rng = np.random.default_rng(4)
    weights = rng.integers(-8, 8, size=(256, 256), dtype=np.int8)
    activation = rng.integers(-127, 128, size=256, dtype=np.int16).astype(np.int8)
    winner, ledger = activation_ordered_tournament_runtime(weights, activation, 64)
    reference = weights.astype(np.int64) @ activation.astype(np.int64)
    assert winner == int(np.flatnonzero(reference == reference.max())[0])
    assert ledger["total_head_query_bytes"] > 0
