import torch
from torch import nn

from vortex_runtime.residual_tile_repair import (
    replace_with_residual_tile_modules,
)


class TinyResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def test_all_residual_tiles_restore_exact_linear_output() -> None:
    torch.manual_seed(41)
    model = TinyResidualModel()
    weight = model.o_proj.weight.detach().clone()
    bias = model.o_proj.bias.detach().clone()
    replacements = replace_with_residual_tile_modules(
        model,
        suffixes=("o_proj",),
        max_rank=2,
    )
    module = replacements["o_proj"]

    basis = torch.randn(8, 2)
    model(torch.randn(16, 2) @ basis.T)
    outside = torch.randn(5, 8)
    exact = outside @ weight.T + bias

    module.set_mode("project")
    projected = model(outside)
    assert not torch.allclose(projected, exact, atol=1e-5, rtol=1e-5)

    module.reset_residual_tile_profile(row_tile=4, col_tile=4)
    module.set_mode("profile_residual")
    profiled = model(outside)
    torch.testing.assert_close(profiled, projected)
    tiles = module.profiled_residual_tiles()
    assert len(tiles) == 4
    assert sum(float(tile["score"]) for tile in tiles) > 0

    module.configure_residual_tile_repair(
        row_tile=4,
        col_tile=4,
        tile_indices=((0, 0),),
    )
    module.set_mode("project_residual_repair")
    partial = model(outside)
    assert not torch.allclose(partial, exact, atol=1e-5, rtol=1e-5)
    assert module.selected_residual_repair_bytes == 4 * 4 * weight.element_size()

    module.configure_residual_tile_repair(
        row_tile=4,
        col_tile=4,
        tile_indices=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    repaired = model(outside)
    torch.testing.assert_close(repaired, exact, atol=2e-5, rtol=2e-5)
    assert module.selected_residual_repair_bytes == weight.numel() * weight.element_size()


def test_residual_profile_ranks_tiles_and_validates_indices() -> None:
    torch.manual_seed(43)
    model = TinyResidualModel()
    replacements = replace_with_residual_tile_modules(
        model,
        suffixes=("o_proj",),
        max_rank=2,
    )
    module = replacements["o_proj"]
    model(torch.randn(12, 8))

    module.reset_residual_tile_profile(row_tile=2, col_tile=4)
    module.set_mode("profile_residual")
    model(torch.randn(3, 8))
    tiles = module.profiled_residual_tiles()
    assert len(tiles) == 8
    assert all(int(tile["weight_bytes"]) == 2 * 4 * 4 for tile in tiles)

    try:
        module.configure_residual_tile_repair(
            row_tile=2,
            col_tile=4,
            tile_indices=((10, 0),),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range residual tile was accepted")
