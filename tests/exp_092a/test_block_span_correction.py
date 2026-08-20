from __future__ import annotations

import numpy as np
import pytest
import torch

from vortex_runtime.block_span_correction import (
    BlockSpanCorrectionError,
    analyze_block_span,
    correction_operation_multiplier,
    deterministic_columns,
    joint_dyadic_integer_rows,
    modular_rank,
    optimistic_sidecar_bytes,
)


def test_deterministic_columns_are_stable_unique_and_bounded() -> None:
    a = deterministic_columns(576, 192, seed="layer0:attn")
    b = deterministic_columns(576, 192, seed="layer0:attn")
    c = deterministic_columns(576, 192, seed="layer1:attn")
    assert a == b
    assert a != c
    assert len(a) == len(set(a)) == 192
    assert min(a) >= 0 and max(a) < 576


def test_identical_blocks_have_zero_extra_rank() -> None:
    torch.manual_seed(1)
    guess = torch.randn(12, 24, dtype=torch.float32).to(torch.bfloat16)
    report = analyze_block_span(
        guess,
        guess.clone(),
        columns=tuple(range(24)),
        primes=(1_000_003, 1_000_033),
    )
    assert report.certified_extra_rank_lower_bound == 0
    assert report.certified_extra_fraction_lower_bound == 0.0
    assert report.zero_extra_directions_observed


def test_injected_new_directions_are_certified() -> None:
    guess = torch.zeros(8, 24, dtype=torch.bfloat16)
    true = torch.zeros(8, 24, dtype=torch.bfloat16)
    for row in range(8):
        guess[row, row] = 1
        true[row, row] = 1
    for row in range(4):
        true[row, 8 + row] = 1
    report = analyze_block_span(
        guess,
        true,
        columns=tuple(range(24)),
        primes=(1_000_003, 1_000_033, 1_000_037),
    )
    assert report.certified_extra_rank_lower_bound == 4
    assert report.certified_extra_fraction_lower_bound == pytest.approx(0.5)
    assert not report.zero_extra_directions_observed


def test_true_rows_inside_guess_span_have_zero_increment() -> None:
    guess = torch.zeros(6, 12, dtype=torch.bfloat16)
    for row in range(6):
        guess[row, row] = 1
    true = torch.stack(
        [
            guess[0] + guess[1],
            guess[2] - guess[3],
            guess[4],
            guess[5],
            guess[0] - guess[5],
            guess[1] + guess[3],
        ]
    ).to(torch.bfloat16)
    report = analyze_block_span(
        guess,
        true,
        columns=tuple(range(12)),
        primes=(1_000_003, 1_000_033),
    )
    assert report.certified_extra_rank_lower_bound == 0


def test_dyadic_encoding_preserves_rank_over_a_prime() -> None:
    guess = torch.tensor(
        [[1.0, 0.5, -2.0], [0.0, 1.0, 4.0]], dtype=torch.bfloat16
    )
    true = torch.tensor(
        [[0.0, 0.0, 1.0], [2.0, 1.0, -4.0]], dtype=torch.bfloat16
    )
    g, t = joint_dyadic_integer_rows(guess, true, columns=(0, 1, 2))
    assert modular_rank(g, 1_000_003) == 2
    assert modular_rank(np.concatenate([g, t], axis=0), 1_000_003) == 3


def test_nonfinite_and_bad_selection_fail_closed() -> None:
    source = torch.ones(2, 4, dtype=torch.bfloat16)
    with pytest.raises(BlockSpanCorrectionError):
        analyze_block_span(source, source, columns=(0, 0))
    bad = source.clone()
    bad[0, 0] = float("inf")
    with pytest.raises(BlockSpanCorrectionError):
        analyze_block_span(bad, source, columns=(0, 1, 2, 3))


def test_target_projection_equations() -> None:
    assert correction_operation_multiplier(128, 0) == 1.0
    assert correction_operation_multiplier(128, 8) == pytest.approx(1.0625)
    assert optimistic_sidecar_bytes(0) == 0
    assert optimistic_sidecar_bytes(8) == 206_438_400
