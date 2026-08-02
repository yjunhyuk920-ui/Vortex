from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .hf_loader import TensorLocator


@dataclass(frozen=True)
class DiskCertificationResult:
    token_id: int
    certified: bool
    coarse_token_id: int
    base_bytes_read: int
    residual_bytes_read: int
    residual_fraction_read: float
    refined_tiles: int
    total_row_tiles: int


def _tile_range(tile: int, tile_cols: int, in_features: int) -> tuple[int, int]:
    start = tile * tile_cols
    return start, min(start + tile_cols, in_features)


def transcode_hf_linear(
    locator: TensorLocator,
    tensor_name: str,
    output_dir: str | Path,
    *,
    base_bits: int = 5,
    tile_cols: int = 128,
    row_block: int = 512,
    residual_dtype: torch.dtype = torch.float32,
) -> Path:
    """Automatically convert one HF matrix without loading it in full."""
    if not 2 <= base_bits <= 8:
        raise ValueError("base_bits must be in [2, 8]")
    shape = locator.shape(tensor_name)
    if len(shape) != 2:
        raise ValueError("tensor must be a matrix")
    out_features, in_features = shape
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    expected = {
        "format": 1,
        "tensor_name": tensor_name,
        "shape": [out_features, in_features],
        "base_bits": base_bits,
        "tile_cols": tile_cols,
        "row_block": row_block,
        "residual_dtype": str(residual_dtype),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(k) == v for k, v in expected.items()):
            return output

    qmax = (1 << (base_bits - 1)) - 1
    num_tiles = math.ceil(in_features / tile_cols)
    blocks: list[dict[str, Any]] = []
    total_base_bytes = 0
    total_residual_bytes = 0
    total_metadata_bytes = 0

    for block_index, row_start in enumerate(range(0, out_features, row_block)):
        row_end = min(row_start + row_block, out_features)
        weight = locator.load_slice(
            tensor_name, (slice(row_start, row_end), slice(None))
        ).to(torch.float32).contiguous()
        quant = torch.empty_like(weight, dtype=torch.int8)
        base = torch.empty_like(weight)
        scales = []
        l2s = []
        l1s = []
        linfs = []
        for tile in range(num_tiles):
            col_start, col_end = _tile_range(tile, tile_cols, in_features)
            part = weight[:, col_start:col_end]
            maximum = part.abs().amax(dim=1, keepdim=True)
            scale = torch.where(
                maximum > 0, maximum / qmax, torch.ones_like(maximum)
            )
            q = torch.round(part / scale).clamp(-qmax, qmax)
            quant[:, col_start:col_end] = q.to(torch.int8)
            base[:, col_start:col_end] = q * scale
            residual_part = part - q * scale
            scales.append(scale.squeeze(1))
            l2s.append(torch.linalg.vector_norm(residual_part, dim=1))
            l1s.append(residual_part.abs().sum(dim=1))
            linfs.append(residual_part.abs().amax(dim=1))
        residual = (weight - base).to(residual_dtype)
        payload = {
            "quant": quant,
            "scale": torch.stack(scales, dim=1),
            "residual": residual,
            "residual_l2": torch.stack(l2s, dim=1),
            "residual_l1": torch.stack(l1s, dim=1),
            "residual_linf": torch.stack(linfs, dim=1),
        }
        filename = f"block-{block_index:06d}.safetensors"
        save_file(payload, output / filename)
        base_bytes = quant.numel() * quant.element_size() + payload["scale"].numel() * 4
        residual_bytes = residual.numel() * residual.element_size()
        metadata_bytes = sum(payload[k].numel() * 4 for k in ("residual_l2", "residual_l1", "residual_linf"))
        total_base_bytes += base_bytes
        total_residual_bytes += residual_bytes
        total_metadata_bytes += metadata_bytes
        blocks.append({
            "row_start": row_start,
            "row_end": row_end,
            "file": filename,
        })

    manifest = {
        **expected,
        "num_tiles": num_tiles,
        "blocks": blocks,
        "base_bytes": total_base_bytes,
        "residual_bytes": total_residual_bytes,
        "metadata_bytes": total_metadata_bytes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


class DiskProgressiveLinear:
    """Progressive argmax over a disk-backed VTX linear matrix."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.out_features, self.in_features = self.manifest["shape"]
        self.tile_cols = int(self.manifest["tile_cols"])
        self.num_tiles = int(self.manifest["num_tiles"])
        self.blocks = self.manifest["blocks"]
        self.row_block = int(self.manifest["row_block"])

    def _base_and_bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        centers = []
        bounds = []
        bytes_read = 0
        x = x.to(torch.float32)
        for block in self.blocks:
            path = self.directory / block["file"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                q = handle.get_tensor("quant")
                scale = handle.get_tensor("scale")
                l2 = handle.get_tensor("residual_l2")
                l1 = handle.get_tensor("residual_l1")
                linf = handle.get_tensor("residual_linf")
            base = torch.empty(q.shape, dtype=torch.float32)
            block_bounds = []
            for tile in range(self.num_tiles):
                start, end = _tile_range(tile, self.tile_cols, self.in_features)
                base[:, start:end] = q[:, start:end].to(torch.float32) * scale[:, tile:tile+1]
                x_tile = x[start:end]
                b2 = l2[:, tile] * torch.linalg.vector_norm(x_tile)
                b1i = l1[:, tile] * x_tile.abs().amax()
                bil = linf[:, tile] * x_tile.abs().sum()
                block_bounds.append(torch.minimum(b2, torch.minimum(b1i, bil)))
            centers.append(base @ x)
            bounds.append(torch.stack(block_bounds, dim=1))
            bytes_read += (
                q.numel() * q.element_size()
                + scale.numel() * scale.element_size()
                + l2.numel() * l2.element_size()
                + l1.numel() * l1.element_size()
                + linf.numel() * linf.element_size()
            )
        return torch.cat(centers), torch.cat(bounds), bytes_read

    def _read_contributions(
        self,
        rows: torch.Tensor,
        tiles: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        result = torch.empty(rows.numel(), dtype=torch.float32)
        bytes_read = 0
        x = x.to(torch.float32)
        by_block: dict[int, list[int]] = {}
        for index, row in enumerate(rows.tolist()):
            by_block.setdefault(row // self.row_block, []).append(index)
        for block_index, indices in by_block.items():
            block = self.blocks[block_index]
            path = self.directory / block["file"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                residual_slice = handle.get_slice("residual")
                for index in indices:
                    global_row = int(rows[index].item())
                    local_row = global_row - block["row_start"]
                    tile = int(tiles[index].item())
                    start, end = _tile_range(tile, self.tile_cols, self.in_features)
                    values = residual_slice[local_row, start:end].to(torch.float32)
                    result[index] = torch.dot(values, x[start:end])
                    bytes_read += values.numel() * values.element_size()
        return result, bytes_read

    def exact_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Stream all base and residual blocks for an exact dense matmul."""
        if x.shape[-1] != self.in_features:
            raise ValueError("invalid input shape")
        original_shape = x.shape[:-1]
        flat_x = x.to(torch.float32).reshape(-1, self.in_features)
        outputs = []
        for block in self.blocks:
            path = self.directory / block["file"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                q = handle.get_tensor("quant")
                scale = handle.get_tensor("scale")
                residual = handle.get_tensor("residual").to(torch.float32)
            base = torch.empty(q.shape, dtype=torch.float32)
            for tile in range(self.num_tiles):
                start, end = _tile_range(tile, self.tile_cols, self.in_features)
                base[:, start:end] = (
                    q[:, start:end].to(torch.float32)
                    * scale[:, tile : tile + 1]
                )
            weight = base + residual
            outputs.append(flat_x @ weight.T)
        result = torch.cat(outputs, dim=-1)
        return result.reshape(*original_shape, self.out_features)

    def certify_argmax(
        self,
        x: torch.Tensor,
        *,
        contenders: int = 32,
        refinement_batch: int = 64,
    ) -> DiskCertificationResult:
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError("invalid input shape")
        center, tile_bounds, base_bytes = self._base_and_bounds(x)
        coarse_token = int(center.argmax().item())
        remaining = tile_bounds.sum(dim=1)
        refined = torch.zeros_like(tile_bounds, dtype=torch.bool)
        residual_bytes = 0
        reads = 0

        while True:
            lower = center - remaining
            upper = center + remaining
            candidate = int(lower.argmax().item())
            other = upper.clone()
            other[candidate] = -torch.inf
            challenger = int(other.argmax().item())
            if lower[candidate] > upper[challenger]:
                return DiskCertificationResult(
                    token_id=candidate,
                    certified=True,
                    coarse_token_id=coarse_token,
                    base_bytes_read=base_bytes,
                    residual_bytes_read=residual_bytes,
                    residual_fraction_read=residual_bytes / max(1, self.manifest["residual_bytes"]),
                    refined_tiles=reads,
                    total_row_tiles=refined.numel(),
                )

            active = torch.zeros(self.out_features, dtype=torch.bool)
            top = torch.topk(upper, k=min(contenders, self.out_features)).indices
            active[top] = True
            active[candidate] = True
            active[challenger] = True
            scores = tile_bounds.masked_fill(refined | ~active[:, None], -1.0)
            available = int((scores >= 0).sum().item())
            if available == 0:
                scores = tile_bounds.masked_fill(refined, -1.0)
                available = int((scores >= 0).sum().item())
                if available == 0:
                    return DiskCertificationResult(
                        token_id=int(center.argmax().item()),
                        certified=True,
                        coarse_token_id=coarse_token,
                        base_bytes_read=base_bytes,
                        residual_bytes_read=residual_bytes,
                        residual_fraction_read=1.0,
                        refined_tiles=reads,
                        total_row_tiles=refined.numel(),
                    )
            batch = min(refinement_batch, available)
            chosen = torch.topk(scores.flatten(), k=batch).indices
            rows = torch.div(chosen, self.num_tiles, rounding_mode="floor")
            tiles = chosen % self.num_tiles
            values, bytes_now = self._read_contributions(rows, tiles, x)
            center.index_add_(0, rows, values)
            remaining.index_add_(0, rows, -tile_bounds[rows, tiles])
            remaining.clamp_(min=0.0)
            refined[rows, tiles] = True
            residual_bytes += bytes_now
            reads += batch
