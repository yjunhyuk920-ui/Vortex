from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn


LLAMA405_IDEAL_BF16_BYTES = 810_000_000_000
TARGET_HOT_BYTES_PER_TOKEN = 100_000_000


class ProgramSharingInvariantError(RuntimeError):
    """Raised when the frozen program language or exactness contract is violated."""


@dataclass(frozen=True)
class TileDescriptor:
    parameter_name: str
    layer_index: int
    module_name: str
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    group: int
    elements: int
    raw_bytes: int


@dataclass(frozen=True)
class ProgramDescriptor:
    layer_indices: tuple[int, ...]
    group_count: int
    kept_mask: int
    kept_groups: tuple[int, ...]
    omitted_groups: tuple[int, ...]
    retained_linear_bytes: int
    total_linear_bytes: int
    retained_fraction: float
    program_sha256: str


@dataclass(frozen=True)
class SharedProgramSelection:
    build_state_ids: tuple[str, ...]
    shared_exact_masks: tuple[int, ...]
    selected_mask: int
    selected_retained_groups: int


@dataclass
class _TileBinding:
    descriptor: TileDescriptor
    weight: torch.Tensor


class CheckpointStaticPagePlan:
    """A deterministic partition of complete linear weights into page families.

    A family contains many non-overlapping 2-D checkpoint tiles and is allowed to
    span layers and projection types. The partition depends only on checkpoint
    layout, never on an activation or target output.
    """

    def __init__(
        self,
        *,
        layer_indices: Sequence[int],
        group_count: int,
        tile_rows: int,
        tile_cols: int,
        bindings: Sequence[_TileBinding],
    ) -> None:
        if group_count <= 0 or group_count > 63:
            raise ProgramSharingInvariantError("group_count must be in [1, 63]")
        if not bindings:
            raise ProgramSharingInvariantError("page plan contains no linear tiles")
        self.layer_indices = tuple(int(v) for v in layer_indices)
        self.group_count = int(group_count)
        self.tile_rows = int(tile_rows)
        self.tile_cols = int(tile_cols)
        self._bindings = tuple(bindings)
        self.full_mask = (1 << self.group_count) - 1

        group_bytes = [0 for _ in range(self.group_count)]
        group_elements = [0 for _ in range(self.group_count)]
        group_layers: list[set[int]] = [set() for _ in range(self.group_count)]
        group_modules: list[set[str]] = [set() for _ in range(self.group_count)]
        for binding in self._bindings:
            d = binding.descriptor
            group_bytes[d.group] += int(d.raw_bytes)
            group_elements[d.group] += int(d.elements)
            group_layers[d.group].add(int(d.layer_index))
            group_modules[d.group].add(d.parameter_name)
        if any(value == 0 for value in group_bytes):
            raise ProgramSharingInvariantError("every page family must own checkpoint bytes")
        self.group_bytes = tuple(group_bytes)
        self.group_elements = tuple(group_elements)
        self.group_layers = tuple(tuple(sorted(v)) for v in group_layers)
        self.group_modules = tuple(tuple(sorted(v)) for v in group_modules)
        self.total_linear_bytes = int(sum(group_bytes))
        self.total_linear_elements = int(sum(group_elements))

    @property
    def descriptors(self) -> tuple[TileDescriptor, ...]:
        return tuple(binding.descriptor for binding in self._bindings)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted({d.parameter_name for d in self.descriptors}))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "checkpoint-static-cross-layer-page-plan-v1",
            "layer_indices": list(self.layer_indices),
            "group_count": self.group_count,
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
            "full_mask": self.full_mask,
            "tile_count": len(self._bindings),
            "parameter_names": list(self.parameter_names),
            "group_bytes": list(self.group_bytes),
            "group_elements": list(self.group_elements),
            "group_layers": [list(v) for v in self.group_layers],
            "group_modules": [list(v) for v in self.group_modules],
            "total_linear_bytes": self.total_linear_bytes,
            "total_linear_elements": self.total_linear_elements,
            "tiles": [asdict(d) for d in self.descriptors],
        }

    def manifest_sha256(self) -> str:
        return sha256_json(self.manifest())

    def program(self, kept_mask: int) -> ProgramDescriptor:
        validate_mask(kept_mask, self.group_count)
        kept = mask_groups(kept_mask, self.group_count)
        omitted = tuple(v for v in range(self.group_count) if v not in kept)
        retained = int(sum(self.group_bytes[group] for group in kept))
        payload = {
            "schema": "cross-layer-page-mask-program-v1",
            "page_plan_sha256": self.manifest_sha256(),
            "layer_indices": list(self.layer_indices),
            "group_count": self.group_count,
            "kept_mask": int(kept_mask),
            "kept_groups": list(kept),
            "omitted_groups": list(omitted),
        }
        return ProgramDescriptor(
            layer_indices=self.layer_indices,
            group_count=self.group_count,
            kept_mask=int(kept_mask),
            kept_groups=kept,
            omitted_groups=omitted,
            retained_linear_bytes=retained,
            total_linear_bytes=self.total_linear_bytes,
            retained_fraction=retained / self.total_linear_bytes,
            program_sha256=sha256_json(payload),
        )

    def executor(self) -> "ProgramMaskExecutor":
        return ProgramMaskExecutor(self)


class ProgramMaskExecutor:
    """Applies page-family masks in-place and restores the checkpoint fail-closed.

    The evaluator still runs the official dense modules. It is an oracle semantic
    harness, not a claimed sparse kernel. A kept mask defines the exact graph in
    which omitted static weight tiles are zero. Gray-code traversal changes one
    family at a time and keeps all original bytes available for restoration.
    """

    def __init__(self, plan: CheckpointStaticPagePlan) -> None:
        self.plan = plan
        self._originals: dict[int, torch.Tensor] = {}
        self._weights: dict[int, torch.Tensor] = {}
        for binding in plan._bindings:
            key = id(binding.weight)
            if key not in self._originals:
                self._originals[key] = binding.weight.detach().clone()
                self._weights[key] = binding.weight
        self.current_mask = plan.full_mask
        self._closed = False

    def __enter__(self) -> "ProgramMaskExecutor":
        return self

    def apply(self, kept_mask: int) -> None:
        if self._closed:
            raise ProgramSharingInvariantError("program executor already closed")
        validate_mask(kept_mask, self.plan.group_count)
        changed = self.current_mask ^ int(kept_mask)
        if changed == 0:
            return
        with torch.no_grad():
            for group in mask_groups(changed, self.plan.group_count):
                enable = bool(kept_mask & (1 << group))
                for binding in self.plan._bindings:
                    d = binding.descriptor
                    if d.group != group:
                        continue
                    view = binding.weight[d.row_start : d.row_stop, d.col_start : d.col_stop]
                    if enable:
                        source = self._originals[id(binding.weight)][
                            d.row_start : d.row_stop, d.col_start : d.col_stop
                        ]
                        view.copy_(source)
                    else:
                        view.zero_()
        self.current_mask = int(kept_mask)

    def restore(self) -> None:
        if self._closed:
            return
        with torch.no_grad():
            for key, weight in self._weights.items():
                weight.copy_(self._originals[key])
        self.current_mask = self.plan.full_mask
        self._closed = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.restore()


def build_checkpoint_static_page_plan(
    layers: Sequence[tuple[int, nn.Module]],
    *,
    group_count: int,
    tile_rows: int,
    tile_cols: int,
) -> CheckpointStaticPagePlan:
    if not layers:
        raise ProgramSharingInvariantError("at least one complete layer is required")
    if tile_rows <= 0 or tile_cols <= 0:
        raise ProgramSharingInvariantError("tile dimensions must be positive")

    provisional: list[tuple[str, int, str, torch.Tensor, int, int, int, int]] = []
    seen_weights: set[int] = set()
    for layer_index, layer in sorted(layers, key=lambda item: int(item[0])):
        for module_name, module in sorted(layer.named_modules(), key=lambda item: item[0]):
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight
            if weight.ndim != 2:
                raise ProgramSharingInvariantError(f"non-matrix linear weight: {module_name}")
            if id(weight) in seen_weights:
                raise ProgramSharingInvariantError("tied linear weight inside selected window")
            seen_weights.add(id(weight))
            parameter_name = f"layers.{int(layer_index)}.{module_name}.weight"
            rows, cols = map(int, weight.shape)
            for row_start in range(0, rows, int(tile_rows)):
                for col_start in range(0, cols, int(tile_cols)):
                    provisional.append(
                        (
                            parameter_name,
                            int(layer_index),
                            module_name,
                            weight,
                            row_start,
                            min(rows, row_start + int(tile_rows)),
                            col_start,
                            min(cols, col_start + int(tile_cols)),
                        )
                    )
    provisional.sort(key=lambda item: (item[1], item[0], item[4], item[6]))
    bindings: list[_TileBinding] = []
    for ordinal, item in enumerate(provisional):
        parameter_name, layer_index, module_name, weight, r0, r1, c0, c1 = item
        elements = int((r1 - r0) * (c1 - c0))
        descriptor = TileDescriptor(
            parameter_name=parameter_name,
            layer_index=layer_index,
            module_name=module_name,
            row_start=r0,
            row_stop=r1,
            col_start=c0,
            col_stop=c1,
            group=ordinal % int(group_count),
            elements=elements,
            raw_bytes=elements * int(weight.element_size()),
        )
        bindings.append(_TileBinding(descriptor=descriptor, weight=weight))
    return CheckpointStaticPagePlan(
        layer_indices=[index for index, _ in layers],
        group_count=group_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        bindings=bindings,
    )


def validate_mask(mask: int, group_count: int) -> None:
    if int(mask) < 0 or int(mask) >= (1 << int(group_count)):
        raise ProgramSharingInvariantError(
            f"mask {mask} outside {group_count}-group program language"
        )


def mask_groups(mask: int, group_count: int) -> tuple[int, ...]:
    validate_mask(mask, group_count)
    return tuple(group for group in range(group_count) if mask & (1 << group))


def gray_kept_masks(group_count: int) -> tuple[int, ...]:
    if group_count <= 0 or group_count > 20:
        raise ProgramSharingInvariantError("exhaustive Gray traversal requires group_count in [1, 20]")
    full = (1 << group_count) - 1
    return tuple(full ^ (index ^ (index >> 1)) for index in range(1 << group_count))


def select_minimal_shared_program(
    exact_masks_by_state: Mapping[str, Iterable[int]],
    build_state_ids: Sequence[str],
    *,
    group_count: int,
    group_bytes: Sequence[int] | None = None,
) -> SharedProgramSelection:
    ids = tuple(str(value) for value in build_state_ids)
    if not ids:
        raise ProgramSharingInvariantError("build state set is empty")
    missing = [state_id for state_id in ids if state_id not in exact_masks_by_state]
    if missing:
        raise ProgramSharingInvariantError(f"missing exact-mask table for {missing}")
    shared: set[int] | None = None
    for state_id in ids:
        masks = {int(mask) for mask in exact_masks_by_state[state_id]}
        for mask in masks:
            validate_mask(mask, group_count)
        shared = masks if shared is None else shared.intersection(masks)
    if not shared:
        raise ProgramSharingInvariantError("no exact program shared by build states")
    if group_bytes is not None:
        if len(group_bytes) != group_count or any(int(value) <= 0 for value in group_bytes):
            raise ProgramSharingInvariantError("group_bytes must contain one positive value per group")

        def program_cost(mask: int) -> tuple[int, int, int]:
            retained = sum(int(group_bytes[group]) for group in mask_groups(mask, group_count))
            return retained, int(mask).bit_count(), int(mask)
    else:

        def program_cost(mask: int) -> tuple[int, int, int]:
            return int(mask).bit_count(), int(mask).bit_count(), int(mask)

    ordered = tuple(sorted(shared, key=program_cost))
    selected = ordered[0]
    return SharedProgramSelection(
        build_state_ids=ids,
        shared_exact_masks=ordered,
        selected_mask=selected,
        selected_retained_groups=int(selected).bit_count(),
    )


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def tensors_bitwise_equal(
    reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]
) -> bool:
    if len(reference) != len(candidate):
        return False
    for lhs, rhs in zip(reference, candidate):
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            return False
        if tensor_bytes(lhs) != tensor_bytes(rhs):
            return False
    return True


def first_tensor_mismatch(
    reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]
) -> dict[str, Any] | None:
    if len(reference) != len(candidate):
        return {"kind": "length", "reference": len(reference), "candidate": len(candidate)}
    for index, (lhs, rhs) in enumerate(zip(reference, candidate)):
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            return {
                "kind": "metadata",
                "tensor_index": index,
                "reference_shape": list(lhs.shape),
                "candidate_shape": list(rhs.shape),
                "reference_dtype": str(lhs.dtype),
                "candidate_dtype": str(rhs.dtype),
            }
        lhs_u8 = lhs.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
        rhs_u8 = rhs.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
        differences = torch.nonzero(lhs_u8 != rhs_u8, as_tuple=False)
        if differences.numel():
            offset = int(differences[0].item())
            return {
                "kind": "bytes",
                "tensor_index": index,
                "byte_offset": offset,
                "reference_byte": int(lhs_u8[offset].item()),
                "candidate_byte": int(rhs_u8[offset].item()),
                "different_bytes": int((lhs_u8 != rhs_u8).sum().item()),
                "total_bytes": int(lhs_u8.numel()),
            }
    return None


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def optimistic_405b_projection(retained_fraction: float) -> dict[str, Any]:
    if retained_fraction < 0.0 or retained_fraction > 1.0:
        raise ProgramSharingInvariantError("retained fraction must be in [0, 1]")
    hot_bytes = int(round(LLAMA405_IDEAL_BF16_BYTES * retained_fraction))
    return {
        "assumption": "all 405B checkpoint linear pages achieve the measured retained fraction; program bytes and non-linear state traffic are free",
        "retained_fraction": float(retained_fraction),
        "ideal_hot_bytes_per_token": hot_bytes,
        "target_hot_bytes_per_token": TARGET_HOT_BYTES_PER_TOKEN,
        "hot_byte_target_passed": hot_bytes <= TARGET_HOT_BYTES_PER_TOKEN,
        "ideal_io_seconds_at_100GBps": hot_bytes / 100_000_000_000.0,
    }
