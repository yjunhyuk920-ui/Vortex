from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def quadratic_feature_count(rank: int) -> int:
    rank = int(rank)
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    return 1 + rank + rank * (rank + 1) // 2


def quadratic_features(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must be [rows,rank]")
    rows, rank = values.shape
    columns = [
        torch.ones((rows, 1), dtype=values.dtype, device=values.device),
        values,
    ]
    products = [
        values[:, i : i + 1] * values[:, j : j + 1]
        for i in range(rank)
        for j in range(i, rank)
    ]
    if products:
        columns.append(torch.cat(products, dim=1))
    return torch.cat(columns, dim=1)


def deterministic_design(rows: int, rank: int, *, seed: int) -> torch.Tensor:
    rows, rank = int(rows), int(rank)
    if rows <= 0 or rank < 0:
        raise ValueError("invalid design shape")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((rows, rank), generator=generator, dtype=torch.float64)


@dataclass(frozen=True)
class WholeSwiGLUIR:
    gate_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    down_shape: tuple[int, ...]
    gate_sha256: str
    up_sha256: str
    down_sha256: str
    module_sha256: str
    source_parameter_bytes: int
    reference_macs_per_vector: int

    @classmethod
    def from_module(cls, module: nn.Module) -> "WholeSwiGLUIR":
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(module, name):
                raise ValueError(f"module missing {name}")
        gate = module.gate_proj.weight.detach()
        up = module.up_proj.weight.detach()
        down = module.down_proj.weight.detach()
        gate_sha = _tensor_sha256(gate)
        up_sha = _tensor_sha256(up)
        down_sha = _tensor_sha256(down)
        payload = json.dumps(
            {
                "gate_shape": list(gate.shape),
                "up_shape": list(up.shape),
                "down_shape": list(down.shape),
                "gate_sha256": gate_sha,
                "up_sha256": up_sha,
                "down_sha256": down_sha,
                "activation": module.act_fn.__class__.__qualname__,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        tensors = (gate, up, down)
        return cls(
            gate_shape=tuple(map(int, gate.shape)),
            up_shape=tuple(map(int, up.shape)),
            down_shape=tuple(map(int, down.shape)),
            gate_sha256=gate_sha,
            up_sha256=up_sha,
            down_sha256=down_sha,
            module_sha256=hashlib.sha256(payload).hexdigest(),
            source_parameter_bytes=sum(
                int(tensor.numel() * tensor.element_size()) for tensor in tensors
            ),
            reference_macs_per_vector=sum(int(tensor.numel()) for tensor in tensors),
        )


def exact_bfloat16_report(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError("shape mismatch")
    if reference.dtype != torch.bfloat16 or candidate.dtype != torch.bfloat16:
        raise ValueError("both tensors must be bfloat16")
    if reference.ndim < 1:
        raise ValueError("expected at least one vector axis")
    flat_reference = reference.reshape(reference.shape[0], -1)
    flat_candidate = candidate.reshape(candidate.shape[0], -1)
    mismatch_counts = (flat_reference != flat_candidate).sum(dim=1)
    return {
        "vector_exact": [bool(value) for value in (mismatch_counts == 0).tolist()],
        "mismatched_rows": [int(value) for value in mismatch_counts.tolist()],
        "total_mismatches": int(mismatch_counts.sum().item()),
    }


def consecutive_true_prefix(values: list[bool]) -> int:
    count = 0
    for value in values:
        if not value:
            break
        count += 1
    return count
