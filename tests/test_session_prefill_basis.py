import torch
from torch import nn

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)
from vortex_runtime.session_prefill_basis import (
    compile_session_response_basis,
)


def test_full_prompt_row_space_reconstructs_exact_linear_output() -> None:
    torch.manual_seed(101)
    wrapper = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=True),
        max_rank=8,
    )
    prompt_inputs = torch.randn(1, 4, 8)
    exact_outputs = wrapper.exact(prompt_inputs)

    stats = compile_session_response_basis(
        wrapper,
        input_tensor=prompt_inputs,
        output_tensor=exact_outputs,
        max_rank=4,
    )
    wrapper.set_mode("project")

    torch.testing.assert_close(
        wrapper(prompt_inputs),
        exact_outputs,
        atol=2e-5,
        rtol=2e-5,
    )
    assert stats.compiled_rank == 4
    assert stats.output_reconstruction_relative_error < 1e-5
    assert stats.capsule_bytes > 0


def test_truncated_prompt_basis_reports_nonzero_reconstruction_error() -> None:
    torch.manual_seed(103)
    wrapper = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=False),
        max_rank=8,
    )
    prompt_inputs = torch.randn(1, 6, 8)
    exact_outputs = wrapper.exact(prompt_inputs)

    stats = compile_session_response_basis(
        wrapper,
        input_tensor=prompt_inputs,
        output_tensor=exact_outputs,
        max_rank=2,
    )

    assert stats.compiled_rank == 2
    assert stats.input_reconstruction_relative_error > 0
    assert stats.output_reconstruction_relative_error > 0


def test_compiler_rejects_contour_shape_mismatch() -> None:
    wrapper = DecisionResidualTileAtlasLinearModule(
        nn.Linear(8, 6, bias=False),
        max_rank=8,
    )
    try:
        compile_session_response_basis(
            wrapper,
            input_tensor=torch.randn(1, 4, 8),
            output_tensor=torch.randn(1, 3, 6),
            max_rank=4,
        )
    except ValueError as exc:
        assert "leading dimensions" in str(exc)
    else:
        raise AssertionError("shape mismatch must be rejected")
