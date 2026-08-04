from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.output_head_demand import (
    analyze_output_head_demand_lower_bound,
    exact_minimum_tile_count_bruteforce,
    subset_certifies_winner,
)


def test_sparse_large_margin_certifies_with_one_coordinate() -> None:
    weight = np.asarray(
        [[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    hidden = np.ones(4, dtype=np.float32)
    result = analyze_output_head_demand_lower_bound(weight, hidden)
    assert result.winner_index == 0
    assert result.necessary_tile_count == 1
    assert result.necessary_column_count == 1
    assert exact_minimum_tile_count_bruteforce(weight, hidden) == 1


def test_late_flip_chain_requires_every_decisive_positive_tile() -> None:
    weight = np.asarray(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.9]],
        dtype=np.float32,
    )
    hidden = np.ones(4, dtype=np.float32)
    result = analyze_output_head_demand_lower_bound(weight, hidden)
    assert result.exact_real_margin == pytest.approx(0.1, abs=1e-6)
    assert result.necessary_tile_count == 3
    assert exact_minimum_tile_count_bruteforce(weight, hidden) == 3


def test_unread_flip_capable_coordinate_prevents_commitment() -> None:
    weight = np.asarray(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.9]],
        dtype=np.float32,
    )
    hidden = np.ones(4, dtype=np.float32)
    assert not subset_certifies_winner(weight, hidden, 0, (0, 1))
    assert subset_certifies_winner(weight, hidden, 0, (0, 1, 2))


def test_lower_bound_never_exceeds_exact_small_oracle() -> None:
    rng = np.random.default_rng(680068)
    for _ in range(12):
        weight = rng.normal(size=(7, 6)).astype(np.float32)
        hidden = rng.normal(size=6).astype(np.float32)
        logits = weight.astype(np.float64) @ hidden.astype(np.float64)
        if np.partition(logits, -2)[-1] == np.partition(logits, -2)[-2]:
            continue
        result = analyze_output_head_demand_lower_bound(weight, hidden)
        exact = exact_minimum_tile_count_bruteforce(weight, hidden)
        assert result.necessary_tile_count <= exact


def test_expected_winner_mismatch_fails_closed_in_metadata() -> None:
    weight = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hidden = np.ones(2, dtype=np.float32)
    result = analyze_output_head_demand_lower_bound(
        weight, hidden, expected_winner=1
    )
    assert result.winner_index == 0
    assert result.domain_winner_mismatch_count == 1


def test_nonfinite_values_fail_closed() -> None:
    weight = np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)
    hidden = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        analyze_output_head_demand_lower_bound(weight, hidden)
