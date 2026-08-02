from __future__ import annotations

import torch

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.quantized_partition_metadata import (
    quantize_nonnegative_rows_upward,
    quantized_partition_metadata_budget,
)


def test_upward_metadata_quantization_never_understates_norms() -> None:
    generator = torch.Generator().manual_seed(32031)
    values = torch.rand(19, 23, generator=generator) * 7.0
    restored, scales = quantize_nonnegative_rows_upward(values, bits=8)
    assert restored.shape == values.shape
    assert scales.shape == (values.shape[0],)
    assert torch.all(restored >= values)
    assert torch.all(restored - values <= scales[:, None] + 1e-6)


def test_zero_rows_remain_sound() -> None:
    values = torch.zeros(4, 9)
    restored, _ = quantize_nonnegative_rows_upward(values, bits=8)
    assert torch.all(restored >= values)
    assert torch.all(restored >= 0)


def test_405b_metadata_budget_charges_row_scales() -> None:
    target = ModelSpec(
        parameters=405_849_243_648,
        layers=126,
        hidden_size=16_384,
        intermediate_size=53_248,
        attention_heads=128,
        kv_heads=8,
        vocab_size=128_256,
        context_tokens=8_192,
        weight_bits=16,
        kv_bits=4,
    )
    budget = quantized_partition_metadata_budget(
        target=target,
        block_size=256,
        norm_bits=8,
        scale_bits=16,
        metadata_limit_gib=2.5,
    )
    assert budget.blocks_per_hidden_vector == 64
    assert abs(budget.metadata_gib - 1.2371978759765625) < 1e-12
    assert budget.metadata_pass
