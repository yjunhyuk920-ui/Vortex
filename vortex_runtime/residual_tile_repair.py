from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.falsification import AtlasLinearModule

ResidualExecutionMode = Literal[
    "learn_exact",
    "project",
    "profile_residual",
    "project_residual_repair",
    "exact",
]


class ResidualTileAtlasLinearModule(AtlasLinearModule):
    """Atlas projection plus exact two-dimensional residual weight tiles.

    For an orthonormal atlas basis ``U``:

    ``W x = W U U.T x + W (I - U U.T) x``.

    The inherited capsule computes the first term. Selected row/column blocks
    of ``W`` compute partial contributions to the second term. Selecting every
    residual tile reproduces the original linear operation up to floating-point
    accumulation order.

    ``profile_residual`` is an optimistic oracle. It scores each weight tile by
    a Cauchy-style energy proxy:

    ``||W_rc||_F^2 * ||residual_c||_2^2``.

    The profiler reads exact weights and is not counted as candidate runtime
    traffic.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        max_rank: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        super().__init__(
            linear,
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
        )
        self.residual_mode: ResidualExecutionMode = "learn_exact"
        self.residual_row_tile = 128
        self.residual_col_tile = 128
        self.selected_residual_tiles: set[tuple[int, int]] = set()
        self.residual_tile_scores: torch.Tensor | None = None
        self.weight_tile_energy: torch.Tensor | None = None

    def set_mode(self, mode: ResidualExecutionMode) -> None:  # type: ignore[override]
        if mode not in {
            "learn_exact",
            "project",
            "profile_residual",
            "project_residual_repair",
            "exact",
        }:
            raise ValueError(f"unsupported residual execution mode: {mode}")
        self.residual_mode = mode
        if mode in {"learn_exact", "project", "exact"}:
            self.mode = mode

    @property
    def residual_tile_shape(self) -> tuple[int, int]:
        return self.residual_row_tile, self.residual_col_tile

    @property
    def selected_residual_repair_bytes(self) -> int:
        element_size = self.exact.weight.element_size()
        total = 0
        for row_index, col_index in self.selected_residual_tiles:
            row_start = row_index * self.residual_row_tile
            row_end = min(
                row_start + self.residual_row_tile,
                self.exact.out_features,
            )
            col_start = col_index * self.residual_col_tile
            col_end = min(
                col_start + self.residual_col_tile,
                self.exact.in_features,
            )
            if row_start < row_end and col_start < col_end:
                total += (
                    (row_end - row_start)
                    * (col_end - col_start)
                    * element_size
                )
        return total

    def _tile_counts(self) -> tuple[int, int]:
        row_tiles = (
            self.exact.out_features + self.residual_row_tile - 1
        ) // self.residual_row_tile
        col_tiles = (
            self.exact.in_features + self.residual_col_tile - 1
        ) // self.residual_col_tile
        return row_tiles, col_tiles

    def reset_residual_tile_profile(
        self,
        *,
        row_tile: int,
        col_tile: int,
    ) -> None:
        if row_tile <= 0 or col_tile <= 0:
            raise ValueError("residual tile dimensions must be positive")
        self.residual_row_tile = int(row_tile)
        self.residual_col_tile = int(col_tile)
        self.selected_residual_tiles.clear()

        weight = self.exact.weight.detach().to("cpu", torch.float32)
        row_tiles, col_tiles = self._tile_counts()
        padded_rows = row_tiles * self.residual_row_tile
        padded_cols = col_tiles * self.residual_col_tile
        padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32)
        padded[: weight.shape[0], : weight.shape[1]] = weight
        tiled = padded.reshape(
            row_tiles,
            self.residual_row_tile,
            col_tiles,
            self.residual_col_tile,
        )
        self.weight_tile_energy = tiled.square().sum(dim=(1, 3)).contiguous()
        self.residual_tile_scores = torch.zeros_like(self.weight_tile_energy)

    def configure_residual_tile_repair(
        self,
        *,
        row_tile: int,
        col_tile: int,
        tile_indices: Iterable[tuple[int, int]],
    ) -> None:
        if row_tile <= 0 or col_tile <= 0:
            raise ValueError("residual tile dimensions must be positive")
        self.residual_row_tile = int(row_tile)
        self.residual_col_tile = int(col_tile)
        row_tiles, col_tiles = self._tile_counts()
        selected = {(int(row), int(col)) for row, col in tile_indices}
        if any(
            row < 0 or row >= row_tiles or col < 0 or col >= col_tiles
            for row, col in selected
        ):
            raise ValueError("residual tile index out of range")
        self.selected_residual_tiles = selected

    def clear_residual_tile_repair(self) -> None:
        self.selected_residual_tiles.clear()

    def _input_residual(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.detach().reshape(-1, self.exact.in_features)
        if self.atlas.rank == 0:
            return flat.reshape(original_shape)
        basis = self.atlas.input_basis.to(dtype=x.dtype, device=x.device)
        coordinates = flat @ basis
        projected = coordinates @ basis.T
        return (flat - projected).reshape(original_shape)

    def _profile_residual_tiles(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_tile_energy is None or self.residual_tile_scores is None:
            raise RuntimeError("call reset_residual_tile_profile first")
        projected = self._project(x)
        residual = self._input_residual(x).to("cpu", torch.float32)
        flat = residual.reshape(-1, self.exact.in_features)
        _, col_tiles = self._tile_counts()
        padded_cols = col_tiles * self.residual_col_tile
        padded = torch.zeros((flat.shape[0], padded_cols), dtype=torch.float32)
        padded[:, : flat.shape[1]] = flat
        column_energy = padded.reshape(
            flat.shape[0],
            col_tiles,
            self.residual_col_tile,
        ).square().sum(dim=(0, 2))
        self.residual_tile_scores += (
            self.weight_tile_energy * column_energy.unsqueeze(0)
        )
        return projected

    def profiled_residual_tiles(self) -> list[dict[str, int | float]]:
        if self.residual_tile_scores is None:
            return []
        element_size = self.exact.weight.element_size()
        row_tiles, col_tiles = self._tile_counts()
        result: list[dict[str, int | float]] = []
        for row_index in range(row_tiles):
            row_start = row_index * self.residual_row_tile
            row_end = min(
                row_start + self.residual_row_tile,
                self.exact.out_features,
            )
            for col_index in range(col_tiles):
                col_start = col_index * self.residual_col_tile
                col_end = min(
                    col_start + self.residual_col_tile,
                    self.exact.in_features,
                )
                tile_bytes = (
                    (row_end - row_start)
                    * (col_end - col_start)
                    * element_size
                )
                score = float(
                    self.residual_tile_scores[row_index, col_index].item()
                )
                result.append(
                    {
                        "row_tile": row_index,
                        "col_tile": col_index,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "weight_bytes": tile_bytes,
                        "score": score,
                        "score_per_byte": score / max(1, tile_bytes),
                    }
                )
        return result

    def _project_with_residual_repairs(self, x: torch.Tensor) -> torch.Tensor:
        output = self._project(x).clone()
        if not self.selected_residual_tiles:
            return output
        residual = self._input_residual(x)
        weight = self.exact.weight
        for row_index, col_index in sorted(self.selected_residual_tiles):
            row_start = row_index * self.residual_row_tile
            row_end = min(
                row_start + self.residual_row_tile,
                self.exact.out_features,
            )
            col_start = col_index * self.residual_col_tile
            col_end = min(
                col_start + self.residual_col_tile,
                self.exact.in_features,
            )
            contribution = F.linear(
                residual[..., col_start:col_end],
                weight[row_start:row_end, col_start:col_end],
                None,
            )
            output[..., row_start:row_end] += contribution
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode == "profile_residual":
            return self._profile_residual_tiles(x)
        if self.residual_mode == "project_residual_repair":
            return self._project_with_residual_repairs(x)
        return super().forward(x)


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_with_residual_tile_modules(
    model: nn.Module,
    *,
    suffixes: Iterable[str],
    max_rank: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, ResidualTileAtlasLinearModule]:
    selected = tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in selected
        ):
            matches.append((name, module))

    replacements: dict[str, ResidualTileAtlasLinearModule] = {}
    for name, linear in matches:
        parent, attribute = _resolve_parent(model, name)
        wrapper = ResidualTileAtlasLinearModule(
            linear,
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
        )
        setattr(parent, attribute, wrapper)
        replacements[name] = wrapper
    return replacements


def set_residual_replacement_modes(
    replacements: dict[str, ResidualTileAtlasLinearModule],
    mode: ResidualExecutionMode,
) -> None:
    for module in replacements.values():
        module.set_mode(mode)
