import json

import torch
from torch import nn

from vortex_runtime.falsification import (
    compute_repair_efficiency,
    replace_linear_modules,
    replacement_delta,
    set_replacement_modes,
    snapshot_replacements,
)


class TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def test_real_module_replacement_preserves_output_and_measures_hits() -> None:
    torch.manual_seed(3)
    model = TinyProjectionModel()
    reference_weight = model.o_proj.weight.detach().clone()
    reference_bias = model.o_proj.bias.detach().clone()
    replacements = replace_linear_modules(
        model,
        suffixes=("o_proj",),
        max_rank=4,
    )
    basis = torch.randn(8, 2)
    build = torch.randn(1, 16, 2) @ basis.T
    eval_rows = torch.randn(12, 2) @ basis.T

    torch.testing.assert_close(
        model(build),
        build @ reference_weight.T + reference_bias,
        atol=1e-5,
        rtol=1e-5,
    )

    before = snapshot_replacements(replacements)
    for row in eval_rows:
        expected = reference_weight @ row + reference_bias
        torch.testing.assert_close(
            model(row),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
    after = snapshot_replacements(replacements)
    delta = replacement_delta(after, before)["o_proj"]

    assert delta["fast_fraction"] == 1.0
    assert delta["decode_fast_fraction"] == 1.0
    assert delta["decode_cold_weight_reads"] == 0
    assert delta["rank_growth"] == 0


def test_project_mode_is_exact_inside_span_and_exact_mode_bypasses_capsule() -> None:
    torch.manual_seed(5)
    model = TinyProjectionModel()
    weight = model.o_proj.weight.detach().clone()
    bias = model.o_proj.bias.detach().clone()
    replacements = replace_linear_modules(
        model,
        suffixes=("o_proj",),
        max_rank=2,
    )
    basis = torch.randn(8, 2)
    build = torch.randn(10, 2) @ basis.T
    model(build)

    set_replacement_modes(replacements, "project")
    inside = torch.randn(4, 2) @ basis.T
    torch.testing.assert_close(
        model(inside),
        inside @ weight.T + bias,
        atol=1e-4,
        rtol=1e-4,
    )

    outside = torch.randn(4, 8)
    projected = model(outside)
    exact = outside @ weight.T + bias
    assert not torch.allclose(projected, exact, atol=1e-5, rtol=1e-5)

    set_replacement_modes(replacements, "exact")
    torch.testing.assert_close(model(outside), exact)


def test_phase_counters_separate_prefill_and_decode() -> None:
    torch.manual_seed(9)
    model = TinyProjectionModel()
    replacements = replace_linear_modules(
        model,
        suffixes=("o_proj",),
        max_rank=8,
    )

    model(torch.randn(1, 3, 8))
    model(torch.randn(1, 1, 8))
    snapshot = snapshot_replacements(replacements)["o_proj"]

    assert snapshot.prefill_vectors == 3
    assert snapshot.decode_vectors == 1
    assert snapshot.prefill_cold_weight_reads == 1
    assert snapshot.decode_cold_weight_reads == 1


def test_repair_efficiency_uses_full_model_equivalent_fraction() -> None:
    result = compute_repair_efficiency(
        generated_tokens=160,
        logical_cold_bytes=25,
        managed_weight_bytes=100,
        full_model_weight_bytes=1000,
    )
    assert result.managed_repair_fraction == 0.25
    assert result.full_model_repair_fraction == 0.025
    assert result.zero_cold_reads is False
    assert result.tokens_per_managed_repair_equivalent == 640
    assert result.tokens_per_full_repair_equivalent == 6400


def test_zero_cold_reads_are_strict_json_and_infinite_for_gate() -> None:
    result = compute_repair_efficiency(
        generated_tokens=12,
        logical_cold_bytes=0,
        managed_weight_bytes=100,
        full_model_weight_bytes=1000,
    )
    payload = result.to_dict()
    assert payload["zero_cold_reads"] is True
    assert payload["tokens_per_full_repair_equivalent"] is None
    assert result.full_model_efficiency_for_gate == float("inf")
    json.dumps(payload, allow_nan=False)
