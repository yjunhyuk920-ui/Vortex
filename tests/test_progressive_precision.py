from __future__ import annotations

import torch
from torch import nn

from vortex_runtime.feasibility import default_specs
from vortex_runtime.progressive_precision import (
    fake_quantize_full_rank_modules,
    full_rank_hot_budget,
    symmetric_per_row_fake_quantize,
)


def test_per_row_quantization_preserves_shape_and_all_directions() -> None:
    weight = torch.tensor(
        [
            [1.0, -0.5, 0.25, -0.125],
            [0.4, 0.3, -0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    restored, stats = symmetric_per_row_fake_quantize(
        weight,
        bits=4,
        source_bits=16,
        row_chunk=1,
    )
    assert restored.shape == weight.shape
    assert stats.elements == weight.numel()
    assert stats.hot_bytes == weight.numel() * 4 / 8
    assert stats.residual_bitplane_bytes == weight.numel() * 12 / 8
    assert stats.relative_l2_error > 0
    assert torch.linalg.matrix_rank(restored) == torch.linalg.matrix_rank(weight)


class TiedToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8, bias=False)
        self.projection.weight = self.embedding.weight
        self.hidden = nn.Linear(4, 4, bias=False)


def test_model_quantization_handles_tied_weight_once() -> None:
    model = TiedToy()
    aggregate, tensors = fake_quantize_full_rank_modules(
        model,
        bits=4,
        source_bits=16,
        row_chunk=2,
    )
    assert aggregate.tensors == 2
    assert len(tensors) == 2
    assert aggregate.elements == model.embedding.weight.numel() + model.hidden.weight.numel()
    assert aggregate.hot_gib < aggregate.original_gib


def test_two_bit_hot_path_has_finite_block_threshold_when_compute_allows() -> None:
    target, baseline = default_specs()
    point = full_rank_hot_budget(
        target=target,
        baseline=baseline,
        hot_bits=2,
        block_positions=2048,
        host_to_device_gib_s=32.0,
        hot_effective_tops=320.0,
    )
    assert point.hot_weight_gib > 90
    assert point.minimum_full_commit_block_ideal is not None
    assert point.minimum_full_commit_block_serialized is not None
    assert point.hot_compute_seconds_per_token < point.baseline_seconds_per_token * 1.2


def test_insufficient_low_precision_compute_has_no_finite_threshold() -> None:
    target, baseline = default_specs()
    point = full_rank_hot_budget(
        target=target,
        baseline=baseline,
        hot_bits=8,
        block_positions=4096,
        host_to_device_gib_s=128.0,
        hot_effective_tops=80.0,
    )
    assert point.minimum_full_commit_block_ideal is None
    assert point.minimum_full_commit_block_serialized is None
    assert not point.ideal_pass
    assert not point.serialized_pass
