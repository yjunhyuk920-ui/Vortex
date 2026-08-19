from __future__ import annotations

import gc
import json
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .fixed_public_dynamic_common import (
    ArtifactIntegrityError,
    CompiledArtifact,
    DTYPE_BY_NAME,
    ExecutorInvariantError,
    TensorRecord,
    config_dict,
    directory_bytes,
    dtype_name,
    json_sha256,
    model_tensor_sha256,
    module_parameter_bytes,
    resolved_model_identity,
    rss_bytes,
    sha256_bytes,
    tensor_bytes,
)
from .fixed_public_dynamic_executor import ExactReplacementDecoderLayer


DEFAULT_OUTPUT_TILE_ROWS = 128


def _pack_tile(tensor: torch.Tensor, artifact_dir: Path, logical_name: str) -> TensorRecord:
    raw = tensor_bytes(tensor)
    compressed = zlib.compress(raw, 9)
    codec, payload = ("zlib-9", compressed) if len(compressed) < len(raw) else ("raw", raw)
    relative_path = logical_name.replace(".", "__") + ".bin"
    (artifact_dir / relative_path).write_bytes(payload)
    return TensorRecord(
        name=logical_name,
        shape=tuple(map(int, tensor.shape)),
        dtype=dtype_name(tensor.dtype),
        raw_bytes=len(raw),
        artifact_bytes=len(payload),
        codec=codec,
        relative_path=relative_path,
        raw_sha256=sha256_bytes(raw),
        artifact_sha256=sha256_bytes(payload),
    )


class ExactRowStreamedMLP(nn.Module):
    """Lossless checkpoint MLP evaluated as output-row tiles without splitting a dot-product reduction axis."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        records: Mapping[str, TensorRecord],
        projection_tiles: Mapping[str, tuple[tuple[str, Optional[str]], ...]],
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        pretraining_tp: int,
        output_tile_rows: int,
        verify_integrity: bool = True,
    ) -> None:
        super().__init__()
        if int(pretraining_tp) != 1:
            raise ExecutorInvariantError("row-streamed MLP fails closed unless pretraining_tp == 1")
        if int(output_tile_rows) <= 0:
            raise ExecutorInvariantError("output_tile_rows must be positive")
        self.artifact_dir = str(artifact_dir)
        self.records = dict(records)
        self.projection_tiles = {name: tuple(tiles) for name, tiles in projection_tiles.items()}
        self.act_fn = act_fn
        self.pretraining_tp = int(pretraining_tp)
        self.output_tile_rows = int(output_tile_rows)
        self.verify_integrity = bool(verify_integrity)
        self.reset_stats()

    @classmethod
    def from_reference(
        cls,
        mlp: nn.Module,
        *,
        artifact_dir: Path,
        output_tile_rows: int = DEFAULT_OUTPUT_TILE_ROWS,
        verify_integrity: bool = True,
    ) -> "ExactRowStreamedMLP":
        if int(output_tile_rows) <= 0:
            raise ExecutorInvariantError("output_tile_rows must be positive")
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(mlp, name):
                raise ExecutorInvariantError(f"reference MLP missing {name}")
        records: dict[str, TensorRecord] = {}
        projection_tiles: dict[str, tuple[tuple[str, Optional[str]], ...]] = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name)
            weight = projection.weight.detach()
            bias = projection.bias.detach() if projection.bias is not None else None
            tiles: list[tuple[str, Optional[str]]] = []
            for start in range(0, int(weight.shape[0]), int(output_tile_rows)):
                stop = min(start + int(output_tile_rows), int(weight.shape[0]))
                weight_key = f"{name}.weight.r{start:06d}-{stop:06d}"
                records[weight_key] = _pack_tile(weight[start:stop], artifact_dir, weight_key)
                bias_key: Optional[str] = None
                if bias is not None:
                    bias_key = f"{name}.bias.r{start:06d}-{stop:06d}"
                    records[bias_key] = _pack_tile(bias[start:stop], artifact_dir, bias_key)
                tiles.append((weight_key, bias_key))
            projection_tiles[name] = tuple(tiles)
        return cls(
            artifact_dir=artifact_dir,
            records=records,
            projection_tiles=projection_tiles,
            act_fn=mlp.act_fn,
            pretraining_tp=int(getattr(getattr(mlp, "config", None), "pretraining_tp", 1)),
            output_tile_rows=int(output_tile_rows),
            verify_integrity=verify_integrity,
        )

    def reset_stats(self) -> None:
        self._stats = {
            key: 0
            for key in (
                "cold_bytes",
                "hot_materialized_bytes",
                "pcie_bytes",
                "gpu_bytes",
                "decompression_ns",
                "materialization_ns",
                "projection_calls",
                "integrity_probes",
                "peak_projection_hot_bytes",
            )
        }

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def _materialize(self, logical_name: str, device: torch.device) -> torch.Tensor:
        record = self.records[logical_name]
        start = time.perf_counter_ns()
        payload = (Path(self.artifact_dir) / record.relative_path).read_bytes()
        self._stats["cold_bytes"] += len(payload)
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(payload) != record.artifact_sha256:
                raise ArtifactIntegrityError(f"artifact payload hash mismatch: {logical_name}")
        decompression_start = time.perf_counter_ns()
        if record.codec == "zlib-9":
            raw = zlib.decompress(payload)
        elif record.codec == "raw":
            raw = payload
        else:
            raise ArtifactIntegrityError(f"unknown codec: {record.codec}")
        self._stats["decompression_ns"] += time.perf_counter_ns() - decompression_start
        if len(raw) != record.raw_bytes:
            raise ArtifactIntegrityError(f"raw byte length mismatch: {logical_name}")
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(raw) != record.raw_sha256:
                raise ArtifactIntegrityError(f"raw tensor hash mismatch: {logical_name}")
        dtype = DTYPE_BY_NAME.get(record.dtype)
        if dtype is None:
            raise ExecutorInvariantError(f"unsupported packed dtype: {record.dtype}")
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(record.shape)
        if tensor.device != device:
            tensor = tensor.to(device)
            if device.type == "cuda":
                self._stats["pcie_bytes"] += record.raw_bytes
        self._stats["hot_materialized_bytes"] += record.raw_bytes
        self._stats["peak_projection_hot_bytes"] = max(self._stats["peak_projection_hot_bytes"], record.raw_bytes)
        if device.type == "cuda":
            self._stats["gpu_bytes"] += record.raw_bytes
        self._stats["materialization_ns"] += time.perf_counter_ns() - start
        return tensor

    def _linear_streamed(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.projection_tiles:
            raise ExecutorInvariantError(f"missing streamed projection: {name}")
        outputs: list[torch.Tensor] = []
        for weight_key, bias_key in self.projection_tiles[name]:
            weight = self._materialize(weight_key, x.device)
            bias = self._materialize(bias_key, x.device) if bias_key is not None else None
            self._stats["projection_calls"] += 1
            outputs.append(F.linear(x, weight, bias))
        if not outputs:
            raise ExecutorInvariantError(f"empty streamed projection: {name}")
        return torch.cat(outputs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self._linear_streamed(x, "gate_proj"))
        up = self._linear_streamed(x, "up_proj")
        return self._linear_streamed(gate * up, "down_proj")


def compile_checkpoint_row_streamed(
    reference_model: nn.Module,
    *,
    artifact_dir: Optional[str | Path] = None,
    layer_index: int = 0,
    output_tile_rows: int = DEFAULT_OUTPUT_TILE_ROWS,
    verify_integrity: bool = True,
) -> CompiledArtifact:
    start = time.perf_counter_ns()
    layers = getattr(getattr(reference_model, "model", None), "layers", None)
    if layers is None or not 0 <= int(layer_index) < len(layers):
        raise ExecutorInvariantError("valid model.model.layers and layer index are required")
    model_id, revision = resolved_model_identity(reference_model)
    config_sha = json_sha256(config_dict(getattr(reference_model, "config", None)))
    checkpoint_hash = model_tensor_sha256(reference_model)
    root = Path(artifact_dir) if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="fixed-public-row-stream-"))
    root.mkdir(parents=True, exist_ok=True)
    original = layers[int(layer_index)]
    original_bytes = module_parameter_bytes(original)
    streamed = ExactRowStreamedMLP.from_reference(
        original.mlp,
        artifact_dir=root,
        output_tile_rows=int(output_tile_rows),
        verify_integrity=verify_integrity,
    )
    replacement = ExactReplacementDecoderLayer(original, streamed)
    layers[int(layer_index)] = replacement
    manifest = {
        "format": "fixed-public-dynamic-executor-artifact-v1",
        "model_id": model_id,
        "revision": revision,
        "model_class": reference_model.__class__.__name__,
        "layer_index": int(layer_index),
        "config_sha256": config_sha,
        "checkpoint_tensor_sha256": checkpoint_hash,
        "mechanism": "checkpoint_mlp_output_row_streamed_lossless_existing_isa",
        "streaming": {
            "axis": "output_rows",
            "output_tile_rows": int(output_tile_rows),
            "reduction_axis_partitioned": False,
            "tile_order": "ascending_output_row",
        },
        "tensor_records": [asdict(record) for _, record in sorted(streamed.records.items())],
        "projection_tiles": {
            name: [[weight_key, bias_key] for weight_key, bias_key in tiles]
            for name, tiles in streamed.projection_tiles.items()
        },
        "exactness_contract": {
            "pretraining_tp": streamed.pretraining_tp,
            "verify_integrity": bool(verify_integrity),
            "arithmetic_order": "down(act(gate(x))*up(x))",
            "dot_product_reduction_axis_split": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_path.write_bytes(manifest_bytes)
    artifact = CompiledArtifact(
        reference_model,
        model_id,
        revision,
        int(layer_index),
        str(root),
        str(manifest_path),
        sha256_bytes(manifest_bytes),
        directory_bytes(root),
        0,
        rss_bytes(),
        checkpoint_hash,
        config_sha,
        original_bytes,
        module_parameter_bytes(replacement),
        replacement,
        "checkpoint_mlp_output_row_streamed_lossless_existing_isa",
    )
    artifact.compile_wall_ns = time.perf_counter_ns() - start
    del original
    gc.collect()
    return artifact
