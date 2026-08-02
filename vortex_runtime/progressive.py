from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CertificationResult:
    token_id: int
    exact_token_id: int
    certified: bool
    refined_tile_reads: int
    total_row_tiles: int
    residual_fraction_read: float
    coarse_token_id: int
    final_margin: float


class ProgressiveLinear:
    """Exact decision-directed linear operator.

    The weight is represented as a low-bit base plus an exact residual:
        W = dequant(Q, scale) + R

    Residual row/column tiles are read only when their precomputed norm bound
    can still change the argmax decision. If all needed tiles are refined, the
    selected token is mathematically identical to dense ``W @ x``.

    This prototype stores the residual in memory. A production implementation
    would place residual tiles in mmap/DirectStorage-backed shards.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        base_bits: int = 4,
        tile_cols: int = 128,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError("weight must be [out_features, in_features]")
        if not 2 <= base_bits <= 8:
            raise ValueError("base_bits must be in [2, 8]")
        if tile_cols <= 0:
            raise ValueError("tile_cols must be positive")

        self.out_features, self.in_features = weight.shape
        self.base_bits = base_bits
        self.tile_cols = tile_cols
        self.compute_dtype = compute_dtype

        w = weight.detach().to(torch.float32).contiguous()
        qmax = (1 << (base_bits - 1)) - 1
        self.num_tiles = math.ceil(self.in_features / self.tile_cols)
        quant = torch.empty_like(w, dtype=torch.int8)
        base = torch.empty_like(w)
        scale_columns = []
        for tile in range(self.num_tiles):
            start, end = self._tile_range(tile)
            block = w[:, start:end]
            block_max = block.abs().amax(dim=1, keepdim=True)
            block_scale = torch.where(
                block_max > 0, block_max / qmax, torch.ones_like(block_max)
            )
            block_quant = torch.round(block / block_scale).clamp(-qmax, qmax)
            quant[:, start:end] = block_quant.to(torch.int8)
            base[:, start:end] = block_quant * block_scale
            scale_columns.append(block_scale.squeeze(1))
        scale = torch.stack(scale_columns, dim=1)
        residual = w - base

        self.quant = quant
        self.scale = scale
        self.residual = residual
        self._exact_weight = w

        norms = []
        abs_sums = []
        max_abs = []
        for tile in range(self.num_tiles):
            start, end = self._tile_range(tile)
            residual_tile = residual[:, start:end]
            norms.append(torch.linalg.vector_norm(residual_tile, dim=1))
            abs_sums.append(residual_tile.abs().sum(dim=1))
            max_abs.append(residual_tile.abs().amax(dim=1))
        self.residual_tile_norms = torch.stack(norms, dim=1)
        self.residual_tile_l1 = torch.stack(abs_sums, dim=1)
        self.residual_tile_linf = torch.stack(max_abs, dim=1)

    def _tile_range(self, tile: int) -> tuple[int, int]:
        start = tile * self.tile_cols
        return start, min(start + self.tile_cols, self.in_features)

    @property
    def storage_bytes(self) -> dict[str, int]:
        return {
            "quant": self.quant.numel() * self.quant.element_size(),
            "scale": self.scale.numel() * self.scale.element_size(),
            "residual": self.residual.numel() * self.residual.element_size(),
            "bounds": self.residual_tile_norms.numel()
            * self.residual_tile_norms.element_size(),
        }

    def base_weight(self) -> torch.Tensor:
        base = torch.empty(
            (self.out_features, self.in_features), dtype=self.compute_dtype
        )
        quant = self.quant.to(self.compute_dtype)
        scales = self.scale.to(self.compute_dtype)
        for tile in range(self.num_tiles):
            start, end = self._tile_range(tile)
            base[:, start:end] = quant[:, start:end] * scales[:, tile : tile + 1]
        return base

    def coarse(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return F.linear(x.to(self.compute_dtype), self.base_weight())

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_x(x)
        return F.linear(x.to(torch.float32), self._exact_weight)

    def row_tile_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """Return certified |R_tile x_tile| bounds for every output row/tile."""
        self._validate_x(x)
        if x.ndim != 1:
            raise ValueError("certification currently supports a single vector")
        x32 = x.to(torch.float32)
        bounds = []
        for tile in range(self.num_tiles):
            start, end = self._tile_range(tile)
            x_tile = x32[start:end]
            l2 = self.residual_tile_norms[:, tile] * torch.linalg.vector_norm(x_tile)
            l1_linf = self.residual_tile_l1[:, tile] * x_tile.abs().amax()
            linf_l1 = self.residual_tile_linf[:, tile] * x_tile.abs().sum()
            bounds.append(torch.minimum(l2, torch.minimum(l1_linf, linf_l1)))
        return torch.stack(bounds, dim=1)

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

    def certify_argmax(
        self,
        x: torch.Tensor,
        *,
        initial_contenders: int = 16,
        max_refined_fraction: float = 1.0,
        refinement_batch: int = 128,
    ) -> CertificationResult:
        """Return the exact dense argmax using lazy residual refinement."""
        self._validate_x(x)
        if x.ndim != 1:
            raise ValueError("certify_argmax expects shape [in_features]")
        if not 0 < max_refined_fraction <= 1:
            raise ValueError("max_refined_fraction must be in (0, 1]")
        if refinement_batch <= 0:
            raise ValueError("refinement_batch must be positive")

        x32 = x.to(torch.float32)
        center = self.coarse(x32).clone()
        coarse_token = int(center.argmax().item())
        tile_bounds = self.row_tile_bounds(x32)
        remaining = tile_bounds.sum(dim=1)
        refined = torch.zeros(
            (self.out_features, self.num_tiles), dtype=torch.bool
        )

        contribution_columns = []
        for tile in range(self.num_tiles):
            start, end = self._tile_range(tile)
            contribution_columns.append(
                self.residual[:, start:end] @ x32[start:end]
            )
        contributions = torch.stack(contribution_columns, dim=1)

        max_reads = max(1, math.ceil(refined.numel() * max_refined_fraction))
        reads = 0
        contender_count = min(max(2, initial_contenders), self.out_features)

        while True:
            lower = center - remaining
            upper = center + remaining
            candidate = int(lower.argmax().item())
            challenger_upper = upper.clone()
            challenger_upper[candidate] = -torch.inf
            challenger = int(challenger_upper.argmax().item())

            if lower[candidate] > upper[challenger]:
                exact_token = int(self.exact(x32).argmax().item())
                margin = float((lower[candidate] - upper[challenger]).item())
                return CertificationResult(
                    token_id=candidate,
                    exact_token_id=exact_token,
                    certified=candidate == exact_token,
                    refined_tile_reads=reads,
                    total_row_tiles=refined.numel(),
                    residual_fraction_read=reads / refined.numel(),
                    coarse_token_id=coarse_token,
                    final_margin=margin,
                )

            if reads >= max_reads:
                exact_logits = self.exact(x32)
                exact_token = int(exact_logits.argmax().item())
                exact_sorted = torch.topk(
                    exact_logits, k=min(2, self.out_features)
                ).values
                exact_margin = (
                    float((exact_sorted[0] - exact_sorted[1]).item())
                    if exact_sorted.numel() > 1
                    else math.inf
                )
                return CertificationResult(
                    token_id=exact_token,
                    exact_token_id=exact_token,
                    certified=True,
                    refined_tile_reads=refined.numel(),
                    total_row_tiles=refined.numel(),
                    residual_fraction_read=1.0,
                    coarse_token_id=coarse_token,
                    final_margin=exact_margin,
                )

            top_rows = torch.topk(upper, k=contender_count).indices
            active_mask = torch.zeros(self.out_features, dtype=torch.bool)
            active_mask[top_rows] = True
            active_mask[candidate] = True
            active_mask[challenger] = True

            scores = tile_bounds.masked_fill(refined, -1.0)
            scores = scores.masked_fill(~active_mask[:, None], -1.0)
            available = int((scores >= 0).sum().item())
            if available == 0:
                scores = tile_bounds.masked_fill(refined, -1.0)
                available = int((scores >= 0).sum().item())
                if available == 0:
                    dense_token = int(center.argmax().item())
                    exact_token = int(self.exact(x32).argmax().item())
                    return CertificationResult(
                        token_id=dense_token,
                        exact_token_id=exact_token,
                        certified=dense_token == exact_token,
                        refined_tile_reads=reads,
                        total_row_tiles=refined.numel(),
                        residual_fraction_read=reads / refined.numel(),
                        coarse_token_id=coarse_token,
                        final_margin=0.0,
                    )

            batch = min(refinement_batch, available, max_reads - reads)
            flat = scores.flatten()
            chosen = torch.topk(flat, k=batch).indices
            rows = torch.div(chosen, self.num_tiles, rounding_mode="floor")
            tiles = chosen % self.num_tiles

            center.index_add_(0, rows, contributions[rows, tiles])
            remaining.index_add_(0, rows, -tile_bounds[rows, tiles])
            remaining.clamp_(min=0.0)
            refined[rows, tiles] = True
            reads += batch

    def approximate_matvec(
        self,
        x: torch.Tensor,
        *,
        absolute_error: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Progressively refine rows until each output bound is below a target."""
        self._validate_x(x)
        if x.ndim != 1:
            raise ValueError("approximate_matvec expects shape [in_features]")
        if absolute_error < 0:
            raise ValueError("absolute_error must be non-negative")

        x32 = x.to(torch.float32)
        center = self.coarse(x32).clone()
        bounds = self.row_tile_bounds(x32)
        remaining = bounds.sum(dim=1)
        refined = torch.zeros_like(bounds, dtype=torch.bool)
        reads = 0

        while torch.any(remaining > absolute_error):
            row = int(torch.argmax(remaining).item())
            masked = bounds[row].masked_fill(refined[row], -1)
            tile = int(torch.argmax(masked).item())
            start, end = self._tile_range(tile)
            center[row] += torch.dot(
                self.residual[row, start:end], x32[start:end]
            )
            remaining[row] -= bounds[row, tile]
            remaining[row] = torch.clamp(remaining[row], min=0.0)
            refined[row, tile] = True
            reads += 1

        return center, remaining, reads
