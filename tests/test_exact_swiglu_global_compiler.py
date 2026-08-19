from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from vortex_runtime.exact_swiglu_global_compiler import (
    FINAL_P50_TARGET_FRACTION,
    ExactGlobalSwiGLUProgram,
    aggregate_plans,
    analyze_swiglu,
    compile_swiglu_function,
    probe_native_abi,
)
from vortex_runtime.fixed_public_dynamic_common import ExecutorInvariantError


class TinySwiGLU(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 12) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def seeded() -> TinySwiGLU:
    torch.manual_seed(20260819)
    return TinySwiGLU().eval()


def test_random_dense_program_has_no_false_equivalence() -> None:
    module = seeded()
    plan = analyze_swiglu(module)
    assert plan.unique_gate_rows == 12
    assert plan.unique_up_rows == 12
    assert plan.unique_gate_up_atoms == 12
    assert plan.duplicate_gate_rows == 0
    assert plan.duplicate_up_rows == 0
    assert plan.duplicate_gate_up_atoms == 0
    assert plan.optimistic_algebraic_operation_fraction == pytest.approx(1.0)
    assert plan.structural_gate_passed is False


def test_structured_positive_control_detects_global_nonlinear_cse() -> None:
    module = seeded()
    with torch.no_grad():
        module.gate_proj.weight[1].copy_(module.gate_proj.weight[0])
        module.up_proj.weight[1].copy_(module.up_proj.weight[0])
        module.gate_proj.weight[3].copy_(-module.gate_proj.weight[2])
        module.up_proj.weight[3].copy_(module.up_proj.weight[2])
        module.down_proj.weight[:, 5].zero_()
    plan = analyze_swiglu(module)
    assert plan.duplicate_gate_rows >= 1
    assert plan.duplicate_up_rows >= 1
    assert plan.duplicate_gate_up_atoms >= 1
    assert plan.antipodal_gate_pairs >= 1
    assert plan.signed_gate_up_pairs >= 1
    assert plan.zero_down_columns == 1
    assert plan.optimistic_algebraic_operation_fraction < 1.0


def test_reference_order_program_is_bitwise_exact() -> None:
    reference = seeded()
    candidate, plan = compile_swiglu_function(copy.deepcopy(reference), mode="reference_order")
    inputs = torch.randn(32, 8)
    probe = probe_native_abi(reference, candidate, inputs, mode="reference_order")
    assert probe.exact
    assert probe.mismatch_elements == 0
    assert candidate.plan.deterministic_sha256 == plan.deterministic_sha256
    assert candidate(inputs).shape == (32, 8)


def test_joint_gate_up_program_is_exact_on_frozen_cpu_float32_probe() -> None:
    reference = seeded()
    candidate = ExactGlobalSwiGLUProgram(copy.deepcopy(reference), mode="joint_gate_up")
    inputs = torch.cat(
        (
            torch.zeros(1, 8),
            torch.ones(1, 8),
            -torch.ones(1, 8),
            torch.randn(64, 8),
        ),
        dim=0,
    )
    probe = probe_native_abi(reference, candidate, inputs, mode="joint_gate_up")
    assert probe.exact
    assert candidate.stats_snapshot()["projection_calls"] > 0


def test_plan_hash_is_deterministic() -> None:
    module = seeded()
    first = analyze_swiglu(module)
    second = analyze_swiglu(copy.deepcopy(module))
    assert first.deterministic_sha256 == second.deterministic_sha256
    aggregate_first = aggregate_plans([first, second])
    aggregate_second = aggregate_plans([analyze_swiglu(module), analyze_swiglu(copy.deepcopy(module))])
    assert aggregate_first["deterministic_sha256"] == aggregate_second["deterministic_sha256"]
    assert aggregate_first["target_fraction"] == pytest.approx(FINAL_P50_TARGET_FRACTION)


def test_nonfinite_weights_fail_closed() -> None:
    module = seeded()
    with torch.no_grad():
        module.gate_proj.weight[0, 0] = float("nan")
    with pytest.raises(ExecutorInvariantError):
        analyze_swiglu(module)
