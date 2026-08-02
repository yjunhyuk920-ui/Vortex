from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.atlas_linear import OnlineAtlasLinear

ExecutionMode = Literal[
    "learn_exact",
    "project",
    "profile",
    "project_repair",
    "exact",
]


@dataclass(frozen=True)
class ReplacementSnapshot:
    rank: int
    capsule_bytes: int
    vectors: int
    fast_vectors: int
    cold_vectors: int
    cold_weight_reads: int
    weight_bytes_read: int
    prefill_vectors: int
    prefill_fast_vectors: int
    prefill_cold_weight_reads: int
    prefill_weight_bytes_read: int
    decode_vectors: int
    decode_fast_vectors: int
    decode_cold_weight_reads: int
    decode_weight_bytes_read: int


@dataclass(frozen=True)
class RepairEfficiency:
    generated_tokens: int
    logical_cold_bytes: int
    managed_weight_bytes: int
    full_model_weight_bytes: int
    managed_repair_fraction: float
    full_model_repair_fraction: float
    zero_cold_reads: bool
    tokens_per_managed_repair_equivalent: float | None
    tokens_per_full_repair_equivalent: float | None

    @property
    def full_model_efficiency_for_gate(self) -> float:
        if self.zero_cold_reads:
            return inf
        assert self.tokens_per_full_repair_equivalent is not None
        return self.tokens_per_full_repair_equivalent

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


class AtlasLinearModule(nn.Module):
    """Drop-in linear wrapper for falsification and oracle repair profiling.

    Modes:

    - ``learn_exact``: exact fallback plus online basis expansion;
    - ``project``: use only the stored ``U/WU`` capsule;
    - ``profile``: return projected output while measuring exact row-tile error;
    - ``project_repair``: project, then replace selected output-row tiles with
      exact linear results;
    - ``exact``: invoke the original linear module.

    Profiling is intentionally oracle-only. It reads exact weights to identify
    the most valuable row tiles but does not count that profiling traffic as a
    candidate runtime result.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        max_rank: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.exact = linear
        self.atlas = OnlineAtlasLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_loader=lambda: self.exact.weight.detach(),
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
            basis_dtype=torch.float32,
        )
        self.mode: ExecutionMode = "learn_exact"
        self.repair_tile_rows = 128
        self.exact_row_tiles: set[int] = set()
        self.profile_tile_rows = 128
        self.tile_error_sums: dict[int, float] = {}
        self.prefill_vectors = 0
        self.prefill_fast_vectors = 0
        self.prefill_cold_weight_reads = 0
        self.prefill_weight_bytes_read = 0
        self.decode_vectors = 0
        self.decode_fast_vectors = 0
        self.decode_cold_weight_reads = 0
        self.decode_weight_bytes_read = 0

    @property
    def logical_weight_bytes(self) -> int:
        weight = self.exact.weight
        return weight.numel() * weight.element_size()

    @property
    def selected_repair_bytes(self) -> int:
        element_size = self.exact.weight.element_size()
        rows = 0
        for tile_index in self.exact_row_tiles:
            start = tile_index * self.repair_tile_rows
            end = min(start + self.repair_tile_rows, self.exact.out_features)
            if start < self.exact.out_features:
                rows += end - start
        return rows * self.exact.in_features * element_size

    def set_mode(self, mode: ExecutionMode) -> None:
        if mode not in {
            "learn_exact",
            "project",
            "profile",
            "project_repair",
            "exact",
        }:
            raise ValueError(f"unsupported execution mode: {mode}")
        self.mode = mode

    def configure_row_tile_repair(
        self,
        *,
        tile_rows: int,
        tile_indices: Iterable[int],
    ) -> None:
        if tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        max_tiles = (self.exact.out_features + tile_rows - 1) // tile_rows
        selected = {int(index) for index in tile_indices}
        if any(index < 0 or index >= max_tiles for index in selected):
            raise ValueError("row tile index out of range")
        self.repair_tile_rows = int(tile_rows)
        self.exact_row_tiles = selected

    def clear_row_tile_repair(self) -> None:
        self.exact_row_tiles.clear()

    def reset_tile_profile(self, *, tile_rows: int) -> None:
        if tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        self.profile_tile_rows = int(tile_rows)
        self.tile_error_sums = {}

    def profiled_row_tiles(self) -> list[dict[str, int | float]]:
        element_size = self.exact.weight.element_size()
        rows: list[dict[str, int | float]] = []
        for tile_index, error_sum in self.tile_error_sums.items():
            start = tile_index * self.profile_tile_rows
            end = min(start + self.profile_tile_rows, self.exact.out_features)
            tile_bytes = (
                (end - start) * self.exact.in_features * element_size
            )
            rows.append(
                {
                    "tile_index": tile_index,
                    "row_start": start,
                    "row_end": end,
                    "weight_bytes": tile_bytes,
                    "error_sum": error_sum,
                    "error_per_byte": error_sum / max(1, tile_bytes),
                }
            )
        return sorted(rows, key=lambda item: int(item["tile_index"]))

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.detach().reshape(-1, self.exact.in_features)
        rank = self.atlas.rank
        if rank == 0:
            output = torch.zeros(
                (flat.shape[0], self.exact.out_features),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            basis = self.atlas.input_basis.to(
                dtype=x.dtype,
                device=x.device,
            )
            image = self.atlas.output_image.to(
                dtype=x.dtype,
                device=x.device,
            )
            output = (flat @ basis) @ image.T
        output = output.reshape(*original_shape, self.exact.out_features)
        if self.exact.bias is not None:
            output = output + self.exact.bias.to(
                dtype=output.dtype,
                device=output.device,
            )
        return output

    def _profile_projected_error(self, x: torch.Tensor) -> torch.Tensor:
        projected = self._project(x)
        exact = self.exact(x)
        squared = (exact - projected).detach().to("cpu", torch.float32).square()
        flat = squared.reshape(-1, self.exact.out_features)
        tile_count = (
            self.exact.out_features + self.profile_tile_rows - 1
        ) // self.profile_tile_rows
        for tile_index in range(tile_count):
            start = tile_index * self.profile_tile_rows
            end = min(start + self.profile_tile_rows, self.exact.out_features)
            value = float(flat[:, start:end].sum().item())
            self.tile_error_sums[tile_index] = (
                self.tile_error_sums.get(tile_index, 0.0) + value
            )
        return projected

    def _project_with_row_repairs(self, x: torch.Tensor) -> torch.Tensor:
        output = self._project(x).clone()
        if not self.exact_row_tiles:
            return output
        weight = self.exact.weight
        bias = self.exact.bias
        for tile_index in sorted(self.exact_row_tiles):
            start = tile_index * self.repair_tile_rows
            end = min(start + self.repair_tile_rows, self.exact.out_features)
            exact_rows = F.linear(
                x,
                weight[start:end],
                None if bias is None else bias[start:end],
            )
            output[..., start:end] = exact_rows
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "exact":
            return self.exact(x)
        if self.mode == "project":
            return self._project(x)
        if self.mode == "profile":
            return self._profile_projected_error(x)
        if self.mode == "project_repair":
            return self._project_with_row_repairs(x)

        stats = self.atlas.stats
        before = (
            stats.vectors,
            stats.fast_vectors,
            stats.cold_weight_reads,
            stats.weight_bytes_read,
        )
        output = self.atlas(x)
        after = (
            stats.vectors,
            stats.fast_vectors,
            stats.cold_weight_reads,
            stats.weight_bytes_read,
        )
        vectors, fast, cold_reads, cold_bytes = (
            right - left for left, right in zip(before, after)
        )

        is_prefill = x.ndim >= 3 and x.shape[-2] > 1
        if is_prefill:
            self.prefill_vectors += vectors
            self.prefill_fast_vectors += fast
            self.prefill_cold_weight_reads += cold_reads
            self.prefill_weight_bytes_read += cold_bytes
        else:
            self.decode_vectors += vectors
            self.decode_fast_vectors += fast
            self.decode_cold_weight_reads += cold_reads
            self.decode_weight_bytes_read += cold_bytes

        if self.exact.bias is not None:
            output = output + self.exact.bias.to(
                dtype=output.dtype,
                device=output.device,
            )
        return output

    def snapshot(self) -> ReplacementSnapshot:
        stats = self.atlas.stats
        return ReplacementSnapshot(
            rank=self.atlas.rank,
            capsule_bytes=self.atlas.capsule_bytes,
            vectors=stats.vectors,
            fast_vectors=stats.fast_vectors,
            cold_vectors=stats.cold_vectors,
            cold_weight_reads=stats.cold_weight_reads,
            weight_bytes_read=stats.weight_bytes_read,
            prefill_vectors=self.prefill_vectors,
            prefill_fast_vectors=self.prefill_fast_vectors,
            prefill_cold_weight_reads=self.prefill_cold_weight_reads,
            prefill_weight_bytes_read=self.prefill_weight_bytes_read,
            decode_vectors=self.decode_vectors,
            decode_fast_vectors=self.decode_fast_vectors,
            decode_cold_weight_reads=self.decode_cold_weight_reads,
            decode_weight_bytes_read=self.decode_weight_bytes_read,
        )


def set_replacement_modes(
    replacements: dict[str, AtlasLinearModule],
    mode: ExecutionMode,
) -> None:
    for module in replacements.values():
        module.set_mode(mode)


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_modules(
    model: nn.Module,
    *,
    suffixes: Iterable[str],
    max_rank: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, AtlasLinearModule]:
    selected = tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in selected
        ):
            matches.append((name, module))

    replacements: dict[str, AtlasLinearModule] = {}
    for name, linear in matches:
        parent, attribute = _resolve_parent(model, name)
        wrapper = AtlasLinearModule(
            linear,
            max_rank=max_rank,
            atol=atol,
            rtol=rtol,
        )
        setattr(parent, attribute, wrapper)
        replacements[name] = wrapper
    return replacements


def snapshot_replacements(
    replacements: dict[str, AtlasLinearModule],
) -> dict[str, ReplacementSnapshot]:
    return {name: module.snapshot() for name, module in replacements.items()}


def replacement_delta(
    after: dict[str, ReplacementSnapshot],
    before: dict[str, ReplacementSnapshot],
) -> dict[str, dict[str, int | float]]:
    zero = ReplacementSnapshot(*([0] * 15))
    result: dict[str, dict[str, int | float]] = {}
    for name, current in after.items():
        previous = before.get(name, zero)
        vectors = current.vectors - previous.vectors
        fast = current.fast_vectors - previous.fast_vectors
        prefill_vectors = current.prefill_vectors - previous.prefill_vectors
        prefill_fast = (
            current.prefill_fast_vectors - previous.prefill_fast_vectors
        )
        decode_vectors = current.decode_vectors - previous.decode_vectors
        decode_fast = current.decode_fast_vectors - previous.decode_fast_vectors
        result[name] = {
            "rank_before": previous.rank,
            "rank_after": current.rank,
            "rank_growth": current.rank - previous.rank,
            "capsule_bytes": current.capsule_bytes,
            "vectors": vectors,
            "fast_vectors": fast,
            "cold_vectors": current.cold_vectors - previous.cold_vectors,
            "cold_weight_reads": (
                current.cold_weight_reads - previous.cold_weight_reads
            ),
            "logical_cold_bytes": (
                current.weight_bytes_read - previous.weight_bytes_read
            ),
            "fast_fraction": fast / max(1, vectors),
            "prefill_vectors": prefill_vectors,
            "prefill_fast_vectors": prefill_fast,
            "prefill_fast_fraction": prefill_fast / max(1, prefill_vectors),
            "prefill_cold_weight_reads": (
                current.prefill_cold_weight_reads
                - previous.prefill_cold_weight_reads
            ),
            "prefill_logical_cold_bytes": (
                current.prefill_weight_bytes_read
                - previous.prefill_weight_bytes_read
            ),
            "decode_vectors": decode_vectors,
            "decode_fast_vectors": decode_fast,
            "decode_fast_fraction": decode_fast / max(1, decode_vectors),
            "decode_cold_weight_reads": (
                current.decode_cold_weight_reads - previous.decode_cold_weight_reads
            ),
            "decode_logical_cold_bytes": (
                current.decode_weight_bytes_read
                - previous.decode_weight_bytes_read
            ),
        }
    return result


def compute_repair_efficiency(
    *,
    generated_tokens: int,
    logical_cold_bytes: int,
    managed_weight_bytes: int,
    full_model_weight_bytes: int,
) -> RepairEfficiency:
    if generated_tokens < 0:
        raise ValueError("generated_tokens must be non-negative")
    if min(logical_cold_bytes, managed_weight_bytes, full_model_weight_bytes) < 0:
        raise ValueError("byte counts must be non-negative")
    if managed_weight_bytes == 0 or full_model_weight_bytes == 0:
        raise ValueError("weight byte denominators must be positive")

    managed_fraction = logical_cold_bytes / managed_weight_bytes
    full_fraction = logical_cold_bytes / full_model_weight_bytes
    zero_cold_reads = logical_cold_bytes == 0
    return RepairEfficiency(
        generated_tokens=generated_tokens,
        logical_cold_bytes=logical_cold_bytes,
        managed_weight_bytes=managed_weight_bytes,
        full_model_weight_bytes=full_model_weight_bytes,
        managed_repair_fraction=managed_fraction,
        full_model_repair_fraction=full_fraction,
        zero_cold_reads=zero_cold_reads,
        tokens_per_managed_repair_equivalent=(
            None if zero_cold_reads else generated_tokens / managed_fraction
        ),
        tokens_per_full_repair_equivalent=(
            None if zero_cold_reads else generated_tokens / full_fraction
        ),
    )
