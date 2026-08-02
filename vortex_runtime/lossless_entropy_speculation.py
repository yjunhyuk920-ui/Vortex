from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2
import zlib

import numpy as np

from vortex_runtime.feasibility import GIB, ModelSpec


_TRANSFORMS = ("raw", "byte_shuffle", "xor16_byte_shuffle")


@dataclass(frozen=True)
class EncodedLosslessTile:
    transform: str
    raw_size: int
    payload: bytes

    @property
    def encoded_size(self) -> int:
        # One byte is sufficient to identify the transform in a real stream.
        return len(self.payload) + 1


@dataclass(frozen=True)
class LosslessTileStats:
    values: int
    raw_bytes: int
    encoded_bytes: int
    transform: str
    bits_per_value: float
    byte_plane_entropy_bits_per_value: float
    symbol_entropy_bits_per_value: float
    xor_symbol_entropy_bits_per_value: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class EntropySpeculationBudget:
    compressed_bits_per_weight: float
    exact_source_bits: int
    compressed_stream_gib: float
    uncompressed_stream_gib: float
    resident_limit_gib: float
    compressed_stream_fits_resident: bool
    verified_positions: int
    committed_tokens: int
    verification_expansion: float
    transfer_seconds_per_pass: float
    decompression_seconds_per_pass: float
    compute_seconds_per_pass: float
    ideal_seconds_per_committed_token: float
    serialized_seconds_per_committed_token: float
    baseline_seconds_per_token: float
    target_ratio: float
    ideal_pass: bool
    serialized_pass: bool
    minimum_straight_commit_ideal: int | None
    minimum_straight_commit_serialized: int | None

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def _require_even(data: bytes) -> None:
    if len(data) % 2:
        raise ValueError("16-bit weight tile must contain an even number of bytes")


def _byte_shuffle(data: bytes) -> bytes:
    _require_even(data)
    source = np.frombuffer(data, dtype=np.uint8)
    return source[0::2].tobytes() + source[1::2].tobytes()


def _byte_unshuffle(data: bytes) -> bytes:
    _require_even(data)
    source = np.frombuffer(data, dtype=np.uint8)
    half = len(source) // 2
    restored = np.empty_like(source)
    restored[0::2] = source[:half]
    restored[1::2] = source[half:]
    return restored.tobytes()


def _xor16(data: bytes) -> bytes:
    _require_even(data)
    values = np.frombuffer(data, dtype="<u2")
    if values.size == 0:
        return data
    transformed = np.empty_like(values)
    transformed[0] = values[0]
    if values.size > 1:
        transformed[1:] = np.bitwise_xor(values[1:], values[:-1])
    return transformed.astype("<u2", copy=False).tobytes()


def _unxor16(data: bytes) -> bytes:
    _require_even(data)
    encoded = np.frombuffer(data, dtype="<u2")
    if encoded.size == 0:
        return data
    restored = np.empty_like(encoded)
    restored[0] = encoded[0]
    for index in range(1, encoded.size):
        restored[index] = np.bitwise_xor(restored[index - 1], encoded[index])
    return restored.astype("<u2", copy=False).tobytes()


def transform_tile(data: bytes, transform: str) -> bytes:
    if transform == "raw":
        return data
    if transform == "byte_shuffle":
        return _byte_shuffle(data)
    if transform == "xor16_byte_shuffle":
        return _byte_shuffle(_xor16(data))
    raise ValueError(f"unsupported transform: {transform}")


def inverse_transform_tile(data: bytes, transform: str) -> bytes:
    if transform == "raw":
        return data
    if transform == "byte_shuffle":
        return _byte_unshuffle(data)
    if transform == "xor16_byte_shuffle":
        return _unxor16(_byte_unshuffle(data))
    raise ValueError(f"unsupported transform: {transform}")


def encode_lossless_tile(
    data: bytes,
    *,
    compression_level: int = 6,
) -> EncodedLosslessTile:
    _require_even(data)
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0, 9]")
    candidates: list[EncodedLosslessTile] = []
    for transform in _TRANSFORMS:
        transformed = transform_tile(data, transform)
        candidates.append(
            EncodedLosslessTile(
                transform=transform,
                raw_size=len(data),
                payload=zlib.compress(transformed, level=compression_level),
            )
        )
    return min(candidates, key=lambda item: item.encoded_size)


def decode_lossless_tile(tile: EncodedLosslessTile) -> bytes:
    transformed = zlib.decompress(tile.payload)
    restored = inverse_transform_tile(transformed, tile.transform)
    if len(restored) != tile.raw_size:
        raise RuntimeError("lossless tile decoded to an unexpected byte length")
    return restored


def _entropy_bits(values: np.ndarray, alphabet_size: int) -> float:
    if values.size == 0:
        return 0.0
    counts = np.bincount(values.astype(np.int64), minlength=alphabet_size)
    nonzero = counts[counts > 0].astype(np.float64)
    probabilities = nonzero / values.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def measure_lossless_tile(
    data: bytes,
    *,
    compression_level: int = 6,
) -> LosslessTileStats:
    _require_even(data)
    encoded = encode_lossless_tile(data, compression_level=compression_level)
    if decode_lossless_tile(encoded) != data:
        raise RuntimeError("lossless tile round-trip failed")

    byte_values = np.frombuffer(data, dtype=np.uint8)
    low = byte_values[0::2]
    high = byte_values[1::2]
    symbols = np.frombuffer(data, dtype="<u2")
    xor_symbols = np.frombuffer(_xor16(data), dtype="<u2")
    values = len(data) // 2
    byte_plane_entropy = _entropy_bits(low, 256) + _entropy_bits(high, 256)
    symbol_entropy = _entropy_bits(symbols, 65_536)
    xor_entropy = _entropy_bits(xor_symbols, 65_536)
    return LosslessTileStats(
        values=values,
        raw_bytes=len(data),
        encoded_bytes=encoded.encoded_size,
        transform=encoded.transform,
        bits_per_value=encoded.encoded_size * 8 / max(values, 1),
        byte_plane_entropy_bits_per_value=byte_plane_entropy,
        symbol_entropy_bits_per_value=symbol_entropy,
        xor_symbol_entropy_bits_per_value=xor_entropy,
    )


def entropy_speculation_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    compressed_bits_per_weight: float,
    exact_source_bits: int,
    verified_positions: int,
    committed_tokens: int,
    host_to_device_gib_s: float = 24.0,
    decompression_output_gib_s: float = 1_000.0,
    target_tensor_tflops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_tensor_tflops: float = 40.0,
    resident_limit_gib: float = 8.0,
    target_ratio: float = 1.2,
) -> EntropySpeculationBudget:
    """Project exact lossless target verification with speculative amortization.

    Weights are transferred and decompressed once per target pass while all
    speculative positions are evaluated as a batch. ``verified_positions`` may
    exceed ``committed_tokens`` for a token tree; the ratio is charged directly
    to target compute. Ideal overlap and fully serialized bounds are both
    reported.
    """

    positive = (
        compressed_bits_per_weight,
        exact_source_bits,
        verified_positions,
        committed_tokens,
        host_to_device_gib_s,
        decompression_output_gib_s,
        target_tensor_tflops,
        baseline_memory_gib_s,
        baseline_tensor_tflops,
        resident_limit_gib,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("budget values must be positive")
    if committed_tokens > verified_positions:
        raise ValueError("committed_tokens cannot exceed verified_positions")
    if compressed_bits_per_weight > exact_source_bits:
        raise ValueError("lossless compressed bit rate cannot exceed source bits here")

    compressed_gib = target.parameters * compressed_bits_per_weight / 8 / GIB
    uncompressed_gib = target.parameters * exact_source_bits / 8 / GIB
    transfer_seconds = compressed_gib / host_to_device_gib_s
    decompression_seconds = uncompressed_gib / decompression_output_gib_s
    operations_per_position = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    compute_seconds = (
        operations_per_position * verified_positions
        / (target_tensor_tflops * 1e12)
    )
    ideal_pass_seconds = max(
        transfer_seconds,
        decompression_seconds,
        compute_seconds,
    )
    serialized_pass_seconds = (
        transfer_seconds + decompression_seconds + compute_seconds
    )

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_operations = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_operations / (
        baseline_tensor_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_seconds = target_ratio * baseline_seconds

    target_compute_per_straight_token = operations_per_position / (
        target_tensor_tflops * 1e12
    )
    stream_ideal = max(transfer_seconds, decompression_seconds)
    minimum_ideal = (
        ceil(stream_ideal / allowed_seconds)
        if target_compute_per_straight_token <= allowed_seconds
        else None
    )
    serialized_stream = transfer_seconds + decompression_seconds
    minimum_serialized = (
        ceil(serialized_stream / (allowed_seconds - target_compute_per_straight_token))
        if target_compute_per_straight_token < allowed_seconds
        else None
    )

    ideal_per_token = ideal_pass_seconds / committed_tokens
    serialized_per_token = serialized_pass_seconds / committed_tokens
    return EntropySpeculationBudget(
        compressed_bits_per_weight=compressed_bits_per_weight,
        exact_source_bits=exact_source_bits,
        compressed_stream_gib=compressed_gib,
        uncompressed_stream_gib=uncompressed_gib,
        resident_limit_gib=resident_limit_gib,
        compressed_stream_fits_resident=compressed_gib <= resident_limit_gib,
        verified_positions=verified_positions,
        committed_tokens=committed_tokens,
        verification_expansion=verified_positions / committed_tokens,
        transfer_seconds_per_pass=transfer_seconds,
        decompression_seconds_per_pass=decompression_seconds,
        compute_seconds_per_pass=compute_seconds,
        ideal_seconds_per_committed_token=ideal_per_token,
        serialized_seconds_per_committed_token=serialized_per_token,
        baseline_seconds_per_token=baseline_seconds,
        target_ratio=target_ratio,
        ideal_pass=ideal_per_token <= allowed_seconds,
        serialized_pass=serialized_per_token <= allowed_seconds,
        minimum_straight_commit_ideal=minimum_ideal,
        minimum_straight_commit_serialized=minimum_serialized,
    )


def maximum_resident_bits_per_weight(
    *,
    parameters: int,
    resident_gib: float,
) -> float:
    if parameters <= 0 or resident_gib <= 0:
        raise ValueError("parameters and resident_gib must be positive")
    return resident_gib * GIB * 8 / parameters
