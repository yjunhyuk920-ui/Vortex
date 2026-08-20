from __future__ import annotations

import hashlib
import math
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


P50_WHOLE_MODEL_FRACTION = 1.2 * 4.0 / 405.0
P95_WHOLE_MODEL_FRACTION = 1.5 * 4.0 / 405.0
LLAMA405_PARAMETERS = 405_849_243_648
LLAMA405_Q4_GIB = 188.98828125
LLAMA405_BF16_GIB = LLAMA405_PARAMETERS * 2 / (1024**3)
TARGET_ROOT_FREE_GIB = 97.6183
TARGET_PCIE_CEILING_GIB_S = 7.4506
TARGET_M5000_FP32_PEAK_TFLOP_S = 4.3
LLAMA405_ONE_MLP_PROJECTION_PARAMETERS = 109_924_319_232
LLAMA405_MLP_PARAMETER_SHARE = (
    3.0 * LLAMA405_ONE_MLP_PROJECTION_PARAMETERS / LLAMA405_PARAMETERS
)


class LosslessEntropyStationaryError(RuntimeError):
    """Fail-closed error for the lossless entropy-stationary compiler."""


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def byte_entropy_bits(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / len(payload)
    return float(-(probabilities * np.log2(probabilities)).sum())


def word16_entropy_bits(payload: bytes) -> float:
    if len(payload) % 2:
        raise LosslessEntropyStationaryError("BF16 payload must have even byte length")
    if not payload:
        return 0.0
    words = np.frombuffer(payload, dtype=np.uint16)
    _unique, counts = np.unique(words, return_counts=True)
    probabilities = counts.astype(np.float64) / words.size
    return float(-(probabilities * np.log2(probabilities)).sum())


@dataclass(frozen=True)
class EncodedTile:
    raw_bytes: int
    encoded_bytes: int
    raw_sha256: str
    encoded_sha256: str
    codec: str
    payload: bytes


@dataclass
class EncodedTensor:
    shape: tuple[int, ...]
    dtype: str
    raw_bytes: int
    encoded_bytes: int
    raw_sha256: str
    byte_entropy_bits: float
    word16_entropy_bits: float
    shannon_word16_bytes: float
    tiles: tuple[EncodedTile, ...]
    encode_ns: int

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(1, self.encoded_bytes)

    @property
    def shannon_ratio(self) -> float:
        return self.raw_bytes / max(1.0, self.shannon_word16_bytes)

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("tiles")
        result["compression_ratio"] = self.compression_ratio
        result["shannon_ratio"] = self.shannon_ratio
        result["tile_count"] = len(self.tiles)
        return result


def encode_tensor_zlib(
    tensor: torch.Tensor, *, tile_bytes: int = 1 << 20
) -> EncodedTensor:
    if tile_bytes <= 0:
        raise LosslessEntropyStationaryError("tile_bytes must be positive")
    raw = tensor_bytes(tensor)
    start = time.perf_counter_ns()
    tiles: list[EncodedTile] = []
    for offset in range(0, len(raw), int(tile_bytes)):
        chunk = raw[offset : offset + int(tile_bytes)]
        compressed = zlib.compress(chunk, 9)
        codec, payload = (
            ("zlib-9", compressed) if len(compressed) < len(chunk) else ("raw", chunk)
        )
        tiles.append(
            EncodedTile(
                raw_bytes=len(chunk),
                encoded_bytes=len(payload),
                raw_sha256=sha256_bytes(chunk),
                encoded_sha256=sha256_bytes(payload),
                codec=codec,
                payload=payload,
            )
        )
    encode_ns = time.perf_counter_ns() - start
    word_entropy = (
        word16_entropy_bits(raw)
        if tensor.element_size() == 2
        else byte_entropy_bits(raw) * tensor.element_size()
    )
    symbols = max(1, tensor.numel())
    shannon_bytes = word_entropy * symbols / 8.0
    return EncodedTensor(
        shape=tuple(map(int, tensor.shape)),
        dtype=str(tensor.dtype).replace("torch.", ""),
        raw_bytes=len(raw),
        encoded_bytes=sum(tile.encoded_bytes for tile in tiles),
        raw_sha256=sha256_bytes(raw),
        byte_entropy_bits=byte_entropy_bits(raw),
        word16_entropy_bits=word_entropy,
        shannon_word16_bytes=shannon_bytes,
        tiles=tuple(tiles),
        encode_ns=encode_ns,
    )


def decode_tensor_zlib(
    encoded: EncodedTensor, *, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    start = time.perf_counter_ns()
    chunks: list[bytes] = []
    for tile in encoded.tiles:
        if sha256_bytes(tile.payload) != tile.encoded_sha256:
            raise LosslessEntropyStationaryError("encoded tile checksum mismatch")
        if tile.codec == "zlib-9":
            raw = zlib.decompress(tile.payload)
        elif tile.codec == "raw":
            raw = tile.payload
        else:
            raise LosslessEntropyStationaryError(f"unsupported codec {tile.codec}")
        if len(raw) != tile.raw_bytes or sha256_bytes(raw) != tile.raw_sha256:
            raise LosslessEntropyStationaryError("decoded tile mismatch")
        chunks.append(raw)
    raw = b"".join(chunks)
    if len(raw) != encoded.raw_bytes or sha256_bytes(raw) != encoded.raw_sha256:
        raise LosslessEntropyStationaryError("decoded tensor mismatch")
    tensor = (
        torch.frombuffer(bytearray(raw), dtype=dtype)
        .reshape(encoded.shape)
        .clone()
    )
    return tensor, time.perf_counter_ns() - start


@dataclass
class BlockLinearReport:
    block_size: int
    vector_width: int
    output_width: int
    sequential_ns: int
    block_ns: int
    block_speedup: float
    mismatch_count: int
    vector_exact_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def block_linear_report(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    repeats: int = 3,
) -> BlockLinearReport:
    if (
        inputs.ndim != 2
        or weight.ndim != 2
        or inputs.shape[1] != weight.shape[1]
    ):
        raise LosslessEntropyStationaryError("invalid block linear shapes")
    if inputs.dtype != weight.dtype:
        raise LosslessEntropyStationaryError("input/weight dtype mismatch")
    repeats = max(1, int(repeats))
    with torch.inference_mode():
        reference = torch.cat(
            [F.linear(inputs[i : i + 1], weight) for i in range(inputs.shape[0])],
            dim=0,
        )
        candidate = F.linear(inputs, weight)
        sequential_samples: list[int] = []
        block_samples: list[int] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _ = torch.cat(
                [
                    F.linear(inputs[i : i + 1], weight)
                    for i in range(inputs.shape[0])
                ],
                dim=0,
            )
            sequential_samples.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            _ = F.linear(inputs, weight)
            block_samples.append(time.perf_counter_ns() - start)
    mismatch_by_vector = (reference != candidate).reshape(inputs.shape[0], -1).any(dim=1)
    sequential_ns = min(sequential_samples)
    block_ns = min(block_samples)
    return BlockLinearReport(
        block_size=int(inputs.shape[0]),
        vector_width=int(inputs.shape[1]),
        output_width=int(weight.shape[0]),
        sequential_ns=int(sequential_ns),
        block_ns=int(block_ns),
        block_speedup=float(sequential_ns / max(1, block_ns)),
        mismatch_count=int(torch.count_nonzero(reference != candidate).item()),
        vector_exact_fraction=float((~mismatch_by_vector).float().mean().item()),
    )


def full_mlp_block_exactness(
    inputs: torch.Tensor,
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_fn: Any,
) -> dict[str, Any]:
    with torch.inference_mode():
        sequential = torch.cat(
            [
                F.linear(
                    act_fn(F.linear(inputs[i : i + 1], gate_weight))
                    * F.linear(inputs[i : i + 1], up_weight),
                    down_weight,
                )
                for i in range(inputs.shape[0])
            ],
            dim=0,
        )
        block = F.linear(
            act_fn(F.linear(inputs, gate_weight)) * F.linear(inputs, up_weight),
            down_weight,
        )
    mismatch = sequential != block
    by_vector = mismatch.reshape(inputs.shape[0], -1).any(dim=1)
    return {
        "block_size": int(inputs.shape[0]),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "vector_exact_fraction": float((~by_vector).float().mean().item()),
    }


@dataclass(frozen=True)
class TargetSurfaceRow:
    source_format: str
    source_gib: float
    compression_ratio: float
    block_size: int
    io_ms_per_token: float
    dense_compute_floor_ms_per_token: float
    total_roofline_floor_ms_per_token: float
    required_native_4b_p50_ms: float
    compressed_checkpoint_gib: float
    fits_registered_root_free_space: bool
    traffic_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_surface(
    *,
    compression_ratios: Iterable[float],
    block_sizes: Iterable[int],
) -> list[TargetSurfaceRow]:
    compute_ms = (
        2.0
        * LLAMA405_PARAMETERS
        / (TARGET_M5000_FP32_PEAK_TFLOP_S * 1e12)
        * 1000.0
    )
    rows: list[TargetSurfaceRow] = []
    for source_format, source_gib in (
        ("ideal_q4", LLAMA405_Q4_GIB),
        ("bf16", LLAMA405_BF16_GIB),
    ):
        for ratio in compression_ratios:
            ratio = float(ratio)
            if not math.isfinite(ratio) or ratio <= 0:
                raise LosslessEntropyStationaryError("invalid compression ratio")
            compressed = source_gib / ratio
            for block_size in block_sizes:
                block_size = int(block_size)
                if block_size <= 0:
                    raise LosslessEntropyStationaryError("invalid block size")
                io_ms = (
                    source_gib
                    / ratio
                    / TARGET_PCIE_CEILING_GIB_S
                    / block_size
                    * 1000.0
                )
                roofline = max(io_ms, compute_ms)
                rows.append(
                    TargetSurfaceRow(
                        source_format=source_format,
                        source_gib=float(source_gib),
                        compression_ratio=ratio,
                        block_size=block_size,
                        io_ms_per_token=float(io_ms),
                        dense_compute_floor_ms_per_token=float(compute_ms),
                        total_roofline_floor_ms_per_token=float(roofline),
                        required_native_4b_p50_ms=float(roofline / 1.2),
                        compressed_checkpoint_gib=float(compressed),
                        fits_registered_root_free_space=bool(
                            compressed <= TARGET_ROOT_FREE_GIB
                        ),
                        traffic_fraction=float(1.0 / (ratio * block_size)),
                    )
                )
    return rows
