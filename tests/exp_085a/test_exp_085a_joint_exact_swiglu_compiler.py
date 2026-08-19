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
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
        self.act_fn = nn.SiLU()


def test_compiler_manifest_and_mutation_fingerprint() -> None:
    torch.manual_seed(1)
    module = TinySwiGLU()
    first = compile_joint_exact_swiglu(module, page_channels=2)
    mutated = copy.deepcopy(module)
    with torch.no_grad():
        mutated.down_proj.weight[0, 0] = torch.nextafter(
            mutated.down_proj.weight[0, 0],
            torch.tensor(float("inf"), dtype=torch.bfloat16),
        )
    second = compile_joint_exact_swiglu(mutated, page_channels=2)
    assert first.page_count == 3
    assert first.manifest()["mechanism"] == "page_separable_fused_swiglu_exact_rounding_refinement"
    assert first.fingerprint != second.fingerprint


def test_queries_fail_closed_and_preserve_reference_candidate() -> None:
    torch.manual_seed(2)
    compiler = compile_joint_exact_swiglu(TinySwiGLU(), page_channels=2)
    x = torch.randn(1, 4, dtype=torch.bfloat16)
    for result in (compiler.query_oracle_greedy(x), compiler.query_sound_metadata(x)):
        assert result.candidate_matches_reference
        assert result.false_accepts == 0
        assert result.gate_up_row_split_mismatches == 0
        assert result.pages_read <= result.page_count
    with pytest.raises(JointSwiGLUCompilerError):
        compiler.query_sound_metadata(torch.full((1, 4), float("nan"), dtype=torch.bfloat16))
