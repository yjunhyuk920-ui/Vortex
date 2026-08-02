from __future__ import annotations

import torch
from torch import nn

from vortex_runtime.layer_precision_oracle import (
    build_layer_precision_effect,
    precision_module_groups,
    rank_precision_effects,
    unique_weight_elements,
)


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([ToyLayer() for _ in range(5)])
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_precision_groups_contiguous_layers_and_io() -> None:
    model = ToyDecoder()
    groups = precision_module_groups(model, layers_per_group=2)
    assert "io" in groups
    assert "layers_000_001" in groups
    assert "layers_002_003" in groups
    assert "layers_004_005" in groups
    assert "embed_tokens" in groups["io"]
    assert "lm_head" in groups["io"]


def test_unique_elements_counts_tied_weight_once() -> None:
    model = ToyDecoder()
    total = unique_weight_elements(model)
    expected = model.embed_tokens.weight.numel()
    expected += sum(
        layer.up_proj.weight.numel() + layer.down_proj.weight.numel()
        for layer in model.layers
    )
    assert total == expected


def test_effect_ranking_prefers_net_correction_and_gap_gain() -> None:
    weak = build_layer_precision_effect(
        group="weak",
        module_names=("a",),
        unique_elements=100,
        total_unique_elements=1000,
        residual_bits=4,
        corrected_base_errors=1,
        introduced_errors=1,
        exact_top1_tokens=8,
        total_tokens=10,
        exact_gap_reduction=0.2,
    )
    strong = build_layer_precision_effect(
        group="strong",
        module_names=("b",),
        unique_elements=200,
        total_unique_elements=1000,
        residual_bits=4,
        corrected_base_errors=2,
        introduced_errors=0,
        exact_top1_tokens=10,
        total_tokens=10,
        exact_gap_reduction=1.0,
    )
    assert rank_precision_effects([weak, strong])[0].group == "strong"
    assert strong.layer_fraction == 0.2
    assert strong.residual_bytes == 100.0
