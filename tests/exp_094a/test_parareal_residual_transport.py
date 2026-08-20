from __future__ import annotations

import math

import pytest
import torch

from vortex_runtime.parareal_residual_transport import (
    PararealTransportInvariantError,
    accepted_prefix,
    exact_residual,
    logical_source_fraction,
    model_free_controls,
    rank_true_token,
    residual_buffer_bytes,
    summarize_transport,
    transport_argmax,
    transport_scores,
)


def test_transport_recovers_fine_scores_at_same_coarse_state() -> None:
    fine = torch.tensor([9.0, 4.0, -3.0, 1.0], dtype=torch.float32)
    coarse = torch.tensor([1.5, 2.0, 0.0, -1.0], dtype=torch.float32)
    transported = transport_scores(fine, coarse, coarse)
    assert transported.dtype == torch.float64
    assert torch.equal(transported, fine.to(torch.float64))
    assert torch.equal(exact_residual(fine, coarse), fine.double() - coarse.double())


def test_transport_applies_coarse_state_delta() -> None:
    fine = torch.tensor([9.0, 4.0, -3.0, 1.0], dtype=torch.float32)
    old = torch.tensor([1.5, 2.0, 0.0, -1.0], dtype=torch.float32)
    new = torch.tensor([1.5, 12.0, 0.0, -1.0], dtype=torch.float32)
    scores = transport_scores(fine, old, new)
    assert torch.equal(scores, torch.tensor([9.0, 14.0, -3.0, 1.0], dtype=torch.float64))
    assert transport_argmax(fine, old, new) == 1
    assert rank_true_token(scores, 0) == 2


def test_transport_rejects_bad_shapes_and_nonfinite_values() -> None:
    with pytest.raises(PararealTransportInvariantError):
        transport_scores(torch.ones(2), torch.ones(3), torch.ones(2))
    with pytest.raises(PararealTransportInvariantError):
        transport_scores(
            torch.tensor([1.0, float("inf")]), torch.ones(2), torch.ones(2)
        )


def test_summary_prefix_and_ranks() -> None:
    candidate = [4, 5, 6, 9]
    target = [4, 5, 7, 9]
    assert accepted_prefix(candidate, target) == 2
    metrics = summarize_transport(candidate, target, [1, 1, 2, 1])
    assert metrics.accepted_prefix == 2
    assert metrics.exact_positions == 3
    assert metrics.exact_fraction == 0.75
    assert metrics.rank_maximum == 2
    assert metrics.top4_fraction == 1.0


def test_resource_equations() -> None:
    assert residual_buffer_bytes(
        block_length=128, vocabulary_size=49_152, value_bytes=8
    ) == 50_331_648
    assert logical_source_fraction(
        128, fine_sweeps=1, compression_ratio=1.261972
    ) == pytest.approx(0.006190707876244481)
    assert math.isinf(logical_source_fraction(0))


def test_model_free_controls_pass() -> None:
    controls = model_free_controls()
    assert controls["passed"]
