from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .fixed_public_dynamic_common import CompiledArtifact, ExecutorInvariantError, config_dict, directory_bytes, json_sha256, model_tensor_sha256, module_parameter_bytes, resolved_model_identity, rss_bytes, sha256_bytes
from .fixed_public_dynamic_executor import ExactReplacementDecoderLayer


class ExactInductorMLP(nn.Module):
    """Second primitive: lower the actual checkpoint MLP with TorchInductor to existing host ISA."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        if not hasattr(torch, "compile"): raise ExecutorInvariantError("torch.compile unavailable")
        self.compiled = torch.compile(mlp, backend="inductor", fullgraph=True, dynamic=False)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {k: 0 for k in ("cold_bytes", "hot_materialized_bytes", "pcie_bytes", "gpu_bytes", "decompression_ns", "materialization_ns", "integrity_probes", "peak_projection_hot_bytes")}
        self._stats["projection_calls"] = 3

    def stats_snapshot(self) -> dict[str, int]: return dict(self._stats)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.compiled(x)


def compile_checkpoint_inductor(reference_model: nn.Module, *, artifact_dir: Optional[str | Path] = None, layer_index: int = 0) -> CompiledArtifact:
    start = time.perf_counter_ns()
    layers = getattr(getattr(reference_model, "model", None), "layers", None)
    if layers is None or not 0 <= int(layer_index) < len(layers): raise ExecutorInvariantError("valid model.model.layers and layer index required")
    model_id, revision = resolved_model_identity(reference_model)
    config_sha, checkpoint_hash = json_sha256(config_dict(getattr(reference_model, "config", None))), model_tensor_sha256(reference_model)
    root = (Path(artifact_dir) if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="fixed-public-inductor-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inductor_cache = (root / "torchinductor").resolve()
    inductor_cache.mkdir(parents=True, exist_ok=True)
    # TorchInductor invokes the C++ compiler from a temporary cwd. A relative cache path
    # therefore makes generated source paths invalid at compile time. Pin the cache to an
    # absolute path so generated C++/shared objects are ordinary existing-ISA artifacts.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    original = layers[int(layer_index)]; original_bytes = module_parameter_bytes(original)
    replacement = ExactReplacementDecoderLayer(original, ExactInductorMLP(original.mlp)); layers[int(layer_index)] = replacement
    manifest = {"format": "fixed-public-dynamic-executor-artifact-v1", "model_id": model_id, "revision": revision, "model_class": reference_model.__class__.__name__, "layer_index": int(layer_index), "config_sha256": config_sha, "checkpoint_tensor_sha256": checkpoint_hash, "mechanism": "checkpoint_mlp_torchinductor_existing_isa", "compile": {"backend": "inductor", "fullgraph": True, "dynamic": False, "cache_dir": str(inductor_cache)}}
    manifest_path = root / "manifest.json"; data = json.dumps(manifest, sort_keys=True, indent=2).encode(); manifest_path.write_bytes(data)
    return CompiledArtifact(reference_model, model_id, revision, int(layer_index), str(root), str(manifest_path), sha256_bytes(data), directory_bytes(root), time.perf_counter_ns() - start, rss_bytes(), checkpoint_hash, config_sha, original_bytes, module_parameter_bytes(replacement), replacement, "checkpoint_mlp_torchinductor_existing_isa")


def refresh_artifact_bytes(artifact: CompiledArtifact) -> int:
    artifact.artifact_bytes = directory_bytes(artifact.artifact_dir)
    return artifact.artifact_bytes
