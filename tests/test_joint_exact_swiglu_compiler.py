from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from vortex_runtime.joint_exact_swiglu_compiler import (
    JointSwiGLUCompilerError,
    compile_joint_exact_swiglu,
)


class TinySwiGLU(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 4, output: int = 4) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, output, bias=False, dtype=torch.bfloat16)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def structured_mlp() -> TinySwiGLU:
    mlp = TinySwiGLU()
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()
        mlp.up_proj.weight.zero_()
        mlp.down_proj.weight.zero_()
        mlp.gate_proj.weight[:2, :2] = torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16)
        mlp.up_proj.weight[:2, :2] = torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16)
        mlp.down_proj.weight[:, :2] = torch.tensor(
            [[2, 0], [0, 2], [1, 1], [-1, 1]], dtype=torch.bfloat16
        )
    return mlp


def test_compiler_manifest_and_determinism() -> None:
    mlp = structured_mlp()
    a = compile_joint_exact_swiglu(mlp, page_channels=2)
    b = compile_joint_exact_swiglu(copy.deepcopy(mlp), page_channels=2)
    assert a.fingerprint == b.fingerprint
    assert a.page_count == 2
    assert a.manifest()["mechanism"] == "page_separable_fused_swiglu_exact_rounding_refinement"
    assert a.source_parameter_bytes > 0
    assert a.hot_metadata_bytes > 0


def test_structured_zero_tail_certifies_without_wrong_output() -> None:
    compiler = compile_joint_exact_swiglu(structured_mlp(), page_channels=2)
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.bfloat16)
    sound = compiler.query_sound_metadata(x)
    oracle = compiler.query_oracle_greedy(x)
    assert sound.bound_violations == 0
    assert sound.false_accepts == 0
    assert oracle.false_accepts == 0
    assert sound.candidate_matches_reference
    assert oracle.candidate_matches_reference
    assert sound.pages_read <= compiler.page_count
    assert oracle.pages_read <= compiler.page_count


def test_dense_random_fails_closed_or_certifies_exactly() -> None:
    torch.manual_seed(20260819)
    mlp = TinySwiGLU(hidden=8, intermediate=8, output=8)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(torch.randn_like(parameter.float()).to(torch.bfloat16))
    compiler = compile_joint_exact_swiglu(mlp, page_channels=2)
    x = torch.randn(1, 8).to(torch.bfloat16)
    for result in (compiler.query_sound_metadata(x), compiler.query_oracle_greedy(x)):
        assert result.false_accepts == 0
        assert result.candidate_matches_reference
        assert result.pages_read <= result.page_count
        assert result.bound_violations == 0
        if result.certified_without_fallback:
            assert not result.fallback_required
        else:
            assert result.fallback_required


def test_bound_metadata_is_sound_on_random_population() -> None:
    torch.manual_seed(7)
    mlp = TinySwiGLU(hidden=8, intermediate=8, output=8)
    compiler = compile_joint_exact_swiglu(mlp, page_channels=2)
    for _ in range(20):
        x = torch.randn(1, 8).to(torch.bfloat16)
        result = compiler.query_sound_metadata(x)
        assert result.bound_violations == 0
        assert result.false_accepts == 0
        assert result.candidate_matches_reference


def test_invalid_shapes_and_bias_fail_closed() -> None:
    mlp = TinySwiGLU(hidden=4, intermediate=4, output=4)
    mlp.gate_proj = nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
    with pytest.raises(JointSwiGLUCompilerError):
        compile_joint_exact_swiglu(mlp, page_channels=2)

    unbiased = TinySwiGLU(hidden=4, intermediate=4, output=4)
    with pytest.raises(JointSwiGLUCompilerError):
        compile_joint_exact_swiglu(unbiased, page_channels=3)
