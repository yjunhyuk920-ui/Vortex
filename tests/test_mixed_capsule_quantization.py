import torch
from torch import nn

from vortex_runtime.capsule_quantization import (
    fake_quantize_mixed_response_capsules,
)
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


def test_session_columns_receive_higher_precision_and_accounting() -> None:
    torch.manual_seed(401)
    module = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=False),
        max_rank=6,
    )
    module.atlas.input_basis = torch.randn(8, 6)
    module.atlas.output_image = torch.randn(6, 6)

    aggregate, per_module = fake_quantize_mixed_response_capsules(
        {"o_proj": module},
        global_ranks={"o_proj": 2},
        global_bits=4,
        session_bits=8,
    )

    assert aggregate.global_columns == 2
    assert aggregate.session_columns == 4
    assert aggregate.global_elements == (8 + 6) * 2
    assert aggregate.session_elements == (8 + 6) * 4
    assert aggregate.logical_payload_bytes == 7 * 2 + 14 * 4
    assert aggregate.scale_bytes == 2 * 6 * 2
    assert aggregate.maximum_session_relative_l2_error <= (
        aggregate.maximum_global_relative_l2_error
    )
    assert per_module["o_proj"]["global_rank"] == 2
    assert per_module["o_proj"]["session_rank"] == 4


def test_split_map_must_match_modules() -> None:
    module = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=False),
        max_rank=2,
    )
    module.atlas.input_basis = torch.randn(8, 2)
    module.atlas.output_image = torch.randn(6, 2)

    try:
        fake_quantize_mixed_response_capsules(
            {"o_proj": module},
            global_ranks={},
            global_bits=4,
            session_bits=8,
        )
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("mismatched split map must be rejected")
