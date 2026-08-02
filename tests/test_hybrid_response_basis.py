import pytest
import torch
from torch import nn

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)
from vortex_runtime.hybrid_response_basis import (
    augment_response_basis_from_prompt_io,
)


def _wrapper() -> DecisionResidualTileAtlasLinearModule:
    return DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=True),
        max_rank=8,
    )


def test_full_hybrid_span_reconstructs_prompt_output() -> None:
    torch.manual_seed(307)
    module = _wrapper()
    global_basis, _ = torch.linalg.qr(torch.randn(8, 3))
    module.atlas.input_basis = global_basis
    module.atlas.output_image = module.exact.weight.detach() @ global_basis

    prompt_inputs = torch.randn(1, 5, 8)
    exact_outputs = module.exact(prompt_inputs)
    stats = augment_response_basis_from_prompt_io(
        module,
        input_tensor=prompt_inputs,
        output_tensor=exact_outputs,
        total_rank=8,
    )
    module.set_mode("project")

    torch.testing.assert_close(
        module(prompt_inputs),
        exact_outputs,
        atol=3e-5,
        rtol=3e-5,
    )
    assert stats.global_rank == 3
    assert stats.added_session_rank == 5
    assert stats.final_rank == 8
    assert stats.final_output_reconstruction_relative_error < 1e-5


def test_truncated_session_augmentation_reduces_prompt_output_error() -> None:
    torch.manual_seed(311)
    module = _wrapper()
    global_basis, _ = torch.linalg.qr(torch.randn(8, 3))
    module.atlas.input_basis = global_basis
    module.atlas.output_image = module.exact.weight.detach() @ global_basis

    prompt_inputs = torch.randn(1, 5, 8)
    exact_outputs = module.exact(prompt_inputs)
    stats = augment_response_basis_from_prompt_io(
        module,
        input_tensor=prompt_inputs,
        output_tensor=exact_outputs,
        total_rank=5,
    )

    assert stats.added_session_rank == 2
    assert stats.final_rank == 5
    assert (
        stats.final_output_reconstruction_relative_error
        < stats.global_output_reconstruction_relative_error
    )
    assert (
        stats.final_input_reconstruction_relative_error
        < stats.global_input_reconstruction_relative_error
    )


def test_total_rank_cannot_drop_existing_global_directions() -> None:
    module = _wrapper()
    global_basis, _ = torch.linalg.qr(torch.randn(8, 4))
    module.atlas.input_basis = global_basis
    module.atlas.output_image = module.exact.weight.detach() @ global_basis

    with pytest.raises(ValueError, match="smaller than the global rank"):
        augment_response_basis_from_prompt_io(
            module,
            input_tensor=torch.randn(1, 4, 8),
            output_tensor=torch.randn(1, 4, 6),
            total_rank=3,
        )
