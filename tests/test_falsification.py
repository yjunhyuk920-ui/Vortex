import json

import torch
from torch import nn

from vortex_runtime.falsification import (
    compute_repair_efficiency,
    replace_linear_modules,
    replacement_delta,
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
    build = torch.randn(16, 2) @ basis.T
    eval_rows = torch.randn(12, 2) @ basis.T

    for row in build:
        expected = reference_weight @ row + reference_bias
        torch.testing.assert_close(
            model(row),
            expected,
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
    assert delta["cold_weight_reads"] == 0
    assert delta["rank_growth"] == 0


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
