from __future__ import annotations

import torch
from torch import nn

from vortex_runtime.cross_layer_program_sharing import (
    build_checkpoint_static_page_plan,
    gray_kept_masks,
    select_minimal_shared_program,
    tensor_bytes,
)


class ResidualScalarLayer(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


def test_page_plan_is_checkpoint_static_cross_layer_and_complete() -> None:
    torch.manual_seed(1)
    layers = [(3, nn.Sequential(nn.Linear(7, 9, bias=False))), (4, nn.Sequential(nn.Linear(9, 7, bias=False)))]
    plan = build_checkpoint_static_page_plan(layers, group_count=3, tile_rows=4, tile_cols=4)
    expected = sum(module.weight.numel() * module.weight.element_size() for _, layer in layers for module in layer.modules() if isinstance(module, nn.Linear))
    assert plan.total_linear_bytes == expected
    assert sum(plan.group_bytes) == expected
    assert all(value > 0 for value in plan.group_bytes)
    assert set(plan.layer_indices) == {3, 4}
    assert all(set(layers_seen) == {3, 4} for layers_seen in plan.group_layers)


def test_gray_traversal_is_exhaustive_and_toggles_one_family() -> None:
    masks = gray_kept_masks(5)
    assert masks[0] == 0b11111
    assert len(masks) == 32
    assert len(set(masks)) == 32
    for left, right in zip(masks, masks[1:]):
        assert (left ^ right).bit_count() == 1


def test_mask_executor_restores_original_checkpoint_bytes() -> None:
    torch.manual_seed(2)
    layer0 = nn.Sequential(nn.Linear(4, 4, bias=False))
    layer1 = nn.Sequential(nn.Linear(4, 4, bias=False))
    plan = build_checkpoint_static_page_plan([(0, layer0), (1, layer1)], group_count=2, tile_rows=2, tile_cols=2)
    before = [tensor_bytes(module.weight) for layer in (layer0, layer1) for module in layer.modules() if isinstance(module, nn.Linear)]
    with plan.executor() as executor:
        executor.apply(0)
        assert all(torch.count_nonzero(module.weight) == 0 for layer in (layer0, layer1) for module in layer.modules() if isinstance(module, nn.Linear))
        executor.apply(plan.full_mask)
    after = [tensor_bytes(module.weight) for layer in (layer0, layer1) for module in layer.modules() if isinstance(module, nn.Linear)]
    assert after == before


def test_nonseparable_two_page_cancellation_positive_control() -> None:
    # Full transition: (1 - 0.5) * (1 + 1) * x = x.
    # Omitting either page alone changes the transition, but omitting both also
    # yields identity. The oracle language must therefore evaluate joint masks,
    # not require layer-wise or page-wise exactness.
    layer0 = ResidualScalarLayer(1.0)
    layer1 = ResidualScalarLayer(-0.5)
    plan = build_checkpoint_static_page_plan([(0, layer0), (1, layer1)], group_count=2, tile_rows=1, tile_cols=1)
    x = torch.tensor([[3.0]], dtype=torch.float32)
    reference = layer1(layer0(x)).detach().clone()
    exact: list[int] = []
    with plan.executor() as executor:
        for mask in gray_kept_masks(2):
            executor.apply(mask)
            candidate = layer1(layer0(x))
            if tensor_bytes(candidate) == tensor_bytes(reference):
                exact.append(mask)
    assert plan.full_mask in exact
    assert 0 in exact
    assert 0b01 not in exact
    assert 0b10 not in exact


def test_minimal_program_is_selected_from_build_intersection() -> None:
    exact = {
        "build_a": {0b111, 0b011},
        "build_b": {0b111, 0b011, 0b001},
        "holdout": {0b111, 0b011},
    }
    selected = select_minimal_shared_program(exact, ["build_a", "build_b"], group_count=3)
    assert selected.selected_mask == 0b011
    assert selected.selected_retained_groups == 2
    assert selected.selected_mask in exact["holdout"]


def test_bitwise_comparator_detects_one_byte_perturbation() -> None:
    from vortex_runtime.cross_layer_program_sharing import first_tensor_mismatch

    reference = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    candidate = reference.clone()
    candidate.view(torch.uint8).reshape(-1)[0] ^= 0x01
    mismatch = first_tensor_mismatch((reference,), (candidate,))
    assert mismatch is not None
    assert mismatch["kind"] == "bytes"
    assert mismatch["different_bytes"] >= 1
