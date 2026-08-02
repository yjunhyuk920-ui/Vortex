import torch
from torch import nn

from vortex_runtime.decision_tile_repair import (
    replace_with_decision_tile_modules,
    score_adjoint_residual_tiles,
)


class TinyDecisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def test_adjoint_tile_scores_sum_to_linearized_full_residual() -> None:
    torch.manual_seed(53)
    model = TinyDecisionModel()
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=("o_proj",),
        max_rank=2,
    )
    module = replacements["o_proj"]
    basis = torch.randn(8, 2)
    model(torch.randn(16, 2) @ basis.T)

    x = torch.randn(4, 8, requires_grad=True)
    module.set_mode("project")
    projected = model(x)
    adjoint = torch.randn_like(projected)
    objective = (projected * adjoint).sum()
    output_gradient = torch.autograd.grad(
        objective,
        projected,
        retain_graph=True,
    )[0]
    input_gradient = torch.autograd.grad(objective, x)[0]
    assert torch.linalg.vector_norm(input_gradient) > 0

    tiles = score_adjoint_residual_tiles(
        module,
        input_tensor=x,
        output_gradient=output_gradient,
        row_tile=4,
        col_tile=4,
    )
    signed_sum = sum(float(tile["signed_margin_contribution"]) for tile in tiles)

    exact = module.exact(x.detach())
    expected = float(((exact - projected.detach()) * adjoint).sum().item())
    assert abs(signed_sum - expected) <= 1e-4


def test_all_decision_residual_tiles_restore_exact_output() -> None:
    torch.manual_seed(59)
    model = TinyDecisionModel()
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=("o_proj",),
        max_rank=2,
    )
    module = replacements["o_proj"]
    model(torch.randn(10, 8))
    x = torch.randn(3, 8)
    exact = module.exact(x)

    module.configure_residual_tile_repair(
        row_tile=4,
        col_tile=4,
        tile_indices=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    module.set_mode("project_residual_repair")
    torch.testing.assert_close(model(x), exact, atol=2e-5, rtol=2e-5)
