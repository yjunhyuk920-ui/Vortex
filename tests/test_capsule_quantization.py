import torch
from torch import nn

from vortex_runtime.capsule_quantization import (
    fake_quantize_columns,
    fake_quantize_response_capsules,
)
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


def test_zero_columns_remain_zero() -> None:
    tensor = torch.zeros(8, 3)
    quantized, stats = fake_quantize_columns(tensor, bits=4)
    torch.testing.assert_close(quantized, tensor)
    assert stats.maximum_absolute_error == 0
    assert stats.relative_l2_error == 0
    assert stats.logical_payload_bytes == 12
    assert stats.scale_bytes == 6


def test_more_bits_reduce_or_match_quantization_error() -> None:
    torch.manual_seed(211)
    tensor = torch.randn(32, 5)
    q4, stats4 = fake_quantize_columns(tensor, bits=4)
    q8, stats8 = fake_quantize_columns(tensor, bits=8)

    assert not torch.equal(q4, tensor)
    assert stats8.relative_l2_error <= stats4.relative_l2_error
    assert stats8.maximum_absolute_error <= stats4.maximum_absolute_error
    assert q4.shape == tensor.shape
    assert q8.shape == tensor.shape


def test_response_capsule_accounting_includes_scale_metadata() -> None:
    torch.manual_seed(223)
    module = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=False),
        max_rank=4,
    )
    module.atlas.input_basis = torch.randn(8, 4)
    module.atlas.output_image = torch.randn(6, 4)

    aggregate, per_module = fake_quantize_response_capsules(
        {"o_proj": module},
        bits=6,
    )

    assert aggregate.modules == 1
    assert aggregate.tensors == 2
    assert aggregate.elements == 56
    assert aggregate.logical_payload_bytes == 42
    assert aggregate.scale_bytes == 16
    assert aggregate.logical_total_bytes == 58
    assert set(per_module["o_proj"]) == {"input_basis", "output_image"}
