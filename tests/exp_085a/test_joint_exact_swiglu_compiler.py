from __future__ import annotations

import copy

import torch
from torch import nn

from vortex_runtime.joint_exact_swiglu_compiler import (
    WholeSwiGLUIR,
    consecutive_true_prefix,
    deterministic_design,
    exact_bfloat16_report,
    quadratic_feature_count,
    quadratic_features,
)


class TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_quadratic_feature_order_and_count() -> None:
    x = torch.tensor([[2.0, 3.0]])
    phi = quadratic_features(x)
    assert phi.shape == (1, quadratic_feature_count(2))
    assert torch.equal(phi, torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]))


def test_design_is_deterministic_and_full_shape() -> None:
    a = deterministic_design(132, 10, seed=7)
    b = deterministic_design(132, 10, seed=7)
    assert torch.equal(a, b)
    assert a.shape == (132, 10)
    assert quadratic_features(a).shape == (132, 66)


def test_whole_swiglu_ir_covers_all_three_matrices_and_mutation() -> None:
    torch.manual_seed(11)
    module = TinySwiGLU()
    before = WholeSwiGLUIR.from_module(module)
    mutated = copy.deepcopy(module)
    with torch.no_grad():
        mutated.down_proj.weight[0, 0] = torch.nextafter(
            mutated.down_proj.weight[0, 0],
            torch.tensor(float("inf"), dtype=mutated.down_proj.weight.dtype),
        )
    after = WholeSwiGLUIR.from_module(mutated)
    assert before.gate_shape == (6, 4)
    assert before.down_shape == (4, 6)
    assert before.reference_macs_per_vector == 72
    assert before.module_sha256 != after.module_sha256
    assert before.down_sha256 != after.down_sha256


def test_bfloat16_exact_report_fails_closed_per_vector() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    candidate = reference.clone()
    candidate[1, 1] = torch.tensor(4.5, dtype=torch.bfloat16)
    report = exact_bfloat16_report(reference, candidate)
    assert report["vector_exact"] == [True, False]
    assert report["mismatched_rows"] == [0, 1]
    assert consecutive_true_prefix(report["vector_exact"]) == 1
