from __future__ import annotations

import pytest
import torch

from vortex_runtime.cross_layer_template_code import (
    CrossLayerTemplateError,
    audit_cross_layer_templates,
)


def _random_stack(layers: int = 8) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(88)
    gate = [torch.randn(12, 6, generator=generator).to(torch.bfloat16) for _ in range(layers)]
    up = [torch.randn(12, 6, generator=generator).to(torch.bfloat16) for _ in range(layers)]
    down_t = [torch.randn(12, 6, generator=generator).to(torch.bfloat16) for _ in range(layers)]
    return gate, up, down_t


def test_all_generators_reconstruct_random_stack_exactly() -> None:
    gate, up, down_t = _random_stack()
    result = audit_cross_layer_templates(
        gate_weights=gate,
        up_weights=up,
        down_weights_transposed=down_t,
        template_groups=4,
    )
    assert result["aggregate"]["all_reconstructions_exact"]
    assert result["aggregate"]["layer_count"] == 8
    assert result["aggregate"]["total_information_fraction"] > 0


def test_piecewise_template_positive_control_crosses_twenty_percent() -> None:
    generator = torch.Generator().manual_seed(89)
    templates = [torch.randn(12, 6, generator=generator).to(torch.bfloat16) for _ in range(4)]
    layers = [templates[index // 8].clone() for index in range(32)]
    result = audit_cross_layer_templates(
        gate_weights=layers,
        up_weights=[layer.clone() for layer in layers],
        down_weights_transposed=[layer.clone() for layer in layers],
        template_groups=4,
    )
    assert result["aggregate"]["all_reconstructions_exact"]
    assert result["aggregate"]["total_information_fraction"] < 0.20
    assert all(
        name in {
            "contiguous_4_template_generator",
            "previous_layer_xor",
            "previous_layer_modular_delta",
        }
        for name in result["aggregate"]["role_wise_best_candidate_names"].values()
    )


def test_deterministic_replay() -> None:
    gate, up, down_t = _random_stack()
    first = audit_cross_layer_templates(
        gate_weights=gate,
        up_weights=up,
        down_weights_transposed=down_t,
        template_groups=4,
    )
    second = audit_cross_layer_templates(
        gate_weights=gate,
        up_weights=up,
        down_weights_transposed=down_t,
        template_groups=4,
    )
    assert first == second


def test_invalid_shape_and_dtype_fail_closed() -> None:
    gate, up, down_t = _random_stack()
    up[0] = torch.ones(11, 6, dtype=torch.bfloat16)
    with pytest.raises(CrossLayerTemplateError):
        audit_cross_layer_templates(
            gate_weights=gate,
            up_weights=up,
            down_weights_transposed=down_t,
            template_groups=4,
        )
    gate, up, down_t = _random_stack()
    gate[0] = gate[0].float()
    with pytest.raises(CrossLayerTemplateError):
        audit_cross_layer_templates(
            gate_weights=gate,
            up_weights=up,
            down_weights_transposed=down_t,
            template_groups=4,
        )
