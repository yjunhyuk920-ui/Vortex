from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from vortex_runtime.residual_tile_repair import (
    ResidualTileAtlasLinearModule,
)


class DecisionResidualTileAtlasLinearModule(ResidualTileAtlasLinearModule):
    """Residual-tile module whose projected path keeps the autograd graph.

    Atlas capsules may be populated while generation runs under
    ``torch.inference_mode``. Such tensors cannot be saved by autograd. The
    adjoint path therefore materializes ordinary cloned tensors before using
    the capsule in differentiable operations.
    """

    def _differentiable_capsule(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.atlas.input_basis.to(
            dtype=x.dtype,
            device=x.device,
        ).clone()
        image = self.atlas.output_image.to(
            dtype=x.dtype,
            device=x.device,
        ).clone()
        return basis, image

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.exact.in_features)
        rank = self.atlas.rank
        if rank == 0:
            output = torch.zeros(
                (flat.shape[0], self.exact.out_features),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            basis, image = self._differentiable_capsule(x)
            output = (flat @ basis) @ image.T
        output = output.reshape(*original_shape, self.exact.out_features)
        if self.exact.bias is not None:
            output = output + self.exact.bias.to(
                dtype=output.dtype,
                device=output.device,
            )
        return output

    def _input_residual(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.exact.in_features)
        if self.atlas.rank == 0:
            return flat.reshape(original_shape)
        basis, _ = self._differentiable_capsule(x)
        coordinates = flat @ basis
        projected = coordinates @ basis.T
        return (flat - projected).reshape(original_shape)


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_with_decision_tile_modules(
    model: nn.Module,
    *,
    suffixes: Iterable[str],
    max_rank: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, DecisionResidualTileAtlasLinearModule]:
    selected = tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in selected
        ):
            matches.append((name, module))

    replacements: dict[str, DecisionResidualTileAtlasLinearModule] = {}
    for name, linear in matches:
        parent, attribute = _resolve_parent(model, name)
        wrapper = DecisionResidualTileAtlasLinearModule(
            linear,
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
        )
        setattr(parent, attribute, wrapper)
        replacements[name] = wrapper
    return replacements


def score_adjoint_residual_tiles(
    module: DecisionResidualTileAtlasLinearModule,
    *,
    input_tensor: torch.Tensor,
    output_gradient: torch.Tensor,
    row_tile: int,
    col_tile: int,
) -> list[dict[str, int | float]]:
    """Return signed first-order logit-margin contributions by weight tile.

    Let ``r = (I-UU.T)x`` and ``g = d margin / d y``. The first-order margin
    contribution of a residual weight tile is:

    ``<W_rc, g_r.T @ r_c>``.

    Summing signed contributions over all tiles equals the linearized full
    residual correction for the captured input and adjoint.
    """

    if row_tile <= 0 or col_tile <= 0:
        raise ValueError("tile dimensions must be positive")
    if input_tensor.shape[-1] != module.exact.in_features:
        raise ValueError("captured input dimension mismatch")
    if output_gradient.shape[-1] != module.exact.out_features:
        raise ValueError("captured output-gradient dimension mismatch")

    residual = module._input_residual(input_tensor).detach().to(
        "cpu", torch.float32
    )
    gradient = output_gradient.detach().to("cpu", torch.float32)
    residual_flat = residual.reshape(-1, module.exact.in_features)
    gradient_flat = gradient.reshape(-1, module.exact.out_features)
    if residual_flat.shape[0] != gradient_flat.shape[0]:
        raise ValueError("captured input and output-gradient batch mismatch")

    cross = gradient_flat.T @ residual_flat
    weight = module.exact.weight.detach().to("cpu", torch.float32)
    contribution_matrix = weight * cross

    row_tiles = (module.exact.out_features + row_tile - 1) // row_tile
    col_tiles = (module.exact.in_features + col_tile - 1) // col_tile
    padded_rows = row_tiles * row_tile
    padded_cols = col_tiles * col_tile
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32)
    padded[: contribution_matrix.shape[0], : contribution_matrix.shape[1]] = (
        contribution_matrix
    )
    tiled = padded.reshape(row_tiles, row_tile, col_tiles, col_tile)
    signed = tiled.sum(dim=(1, 3))

    element_size = module.exact.weight.element_size()
    result: list[dict[str, int | float]] = []
    for row_index in range(row_tiles):
        row_start = row_index * row_tile
        row_end = min(row_start + row_tile, module.exact.out_features)
        for col_index in range(col_tiles):
            col_start = col_index * col_tile
            col_end = min(col_start + col_tile, module.exact.in_features)
            tile_bytes = (
                (row_end - row_start)
                * (col_end - col_start)
                * element_size
            )
            value = float(signed[row_index, col_index].item())
            result.append(
                {
                    "row_tile": row_index,
                    "col_tile": col_index,
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "weight_bytes": tile_bytes,
                    "signed_margin_contribution": value,
                    "absolute_margin_contribution": abs(value),
                    "positive_contribution_per_byte": max(0.0, value)
                    / max(1, tile_bytes),
                    "absolute_contribution_per_byte": abs(value)
                    / max(1, tile_bytes),
                }
            )
    return result
