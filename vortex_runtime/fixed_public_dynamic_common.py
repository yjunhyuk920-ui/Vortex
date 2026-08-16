from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3.1-405B-Instruct"
TARGET_REVISION = "f9801cba95a53242b3cc928a4a418d12571d1c5f"
DEV_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEV_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
PINNED_TORCH = "2.5.1+cpu"
PINNED_TRANSFORMERS = "4.46.3"
PINNED_SAFETENSORS = "0.4.5"


class ExecutorInvariantError(RuntimeError):
    pass


class ArtifactIntegrityError(ExecutorInvariantError):
    pass


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    raw_bytes: int
    artifact_bytes: int
    codec: str
    relative_path: str
    raw_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class RNGState:
    mode: str = "greedy"
    generator_state: bytes = b""
    seed: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @staticmethod
    def greedy() -> "RNGState":
        return RNGState(mode="greedy")

    @staticmethod
    def sampled(seed: int, *, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> "RNGState":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return RNGState("sample", bytes(generator.get_state().tolist()), int(seed), float(temperature), int(top_k), float(top_p))


@dataclass
class CompiledState:
    past_key_values: Any
    position: int
    step: int = 0
    rng_state: RNGState = field(default_factory=RNGState.greedy)
    prefix_sha256: str = ""


@dataclass
class ResourceTrace:
    step: int
    wall_ns: int
    cold_bytes: int
    hot_materialized_bytes: int
    pcie_bytes: int
    gpu_bytes: int
    kv_bytes: int
    artifact_bytes: int
    resident_model_parameter_bytes: int
    compiled_layer_resident_parameter_bytes: int
    replaced_layer_reference_parameter_bytes: int
    decompression_ns: int
    materialization_ns: int
    projection_calls: int
    integrity_probes: int
    algorithmic_flops: int
    cpu_rss_bytes: Optional[int]
    peak_vram_bytes: Optional[int]
    isa_instruction_count: Optional[int] = None
    isa_status: str = "NOT_MEASURED"
    register_count: Optional[int] = None
    local_memory_spill_bytes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledArtifact:
    model: nn.Module
    model_id: str
    revision: str
    layer_index: int
    artifact_dir: str
    manifest_path: str
    manifest_sha256: str
    artifact_bytes: int
    compile_wall_ns: int
    compile_peak_rss_bytes: Optional[int]
    checkpoint_tensor_sha256: str
    config_sha256: str
    replaced_layer_reference_parameter_bytes: int
    compiled_layer_resident_parameter_bytes: int
    runtime_layer: Optional[nn.Module] = None
    mechanism: str = "cold_packed_projection_sequential_materialization"


DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "int64": torch.int64, "int32": torch.int32,
    "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool,
}


def dtype_name(dtype: torch.dtype) -> str:
    value = str(dtype)
    if not value.startswith("torch."):
        raise ExecutorInvariantError(f"unsupported dtype representation: {value}")
    return value[6:]


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def module_parameter_bytes(module: nn.Module) -> int:
    return sum(tensor_nbytes(p) for p in module.parameters())


def rss_bytes() -> Optional[int]:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return {k: v for k, v in vars(config).items() if not k.startswith("_")} if hasattr(config, "__dict__") else {"repr": repr(config)}


def model_tensor_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(dtype_name(tensor.dtype).encode()); h.update(tensor_bytes(tensor))
    return h.hexdigest()


def resolved_model_identity(model: nn.Module) -> tuple[str, str]:
    config = getattr(model, "config", None)
    model_id = str(getattr(config, "_name_or_path", "") or getattr(model, "name_or_path", "") or "UNKNOWN")
    revision = str(getattr(config, "_commit_hash", "") or getattr(model, "_vortex_revision", "") or "UNKNOWN")
    return model_id, revision


def directory_bytes(path: str | os.PathLike[str]) -> int:
    root = Path(path)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def percentile_ns(values: Sequence[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return int(round(ordered[lo] * (1 - fraction) + ordered[hi] * fraction))


def environment_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "cpu_count": os.cpu_count(),
        "pinned": {"torch": PINNED_TORCH, "transformers": PINNED_TRANSFORMERS, "safetensors": PINNED_SAFETENSORS},
    }
