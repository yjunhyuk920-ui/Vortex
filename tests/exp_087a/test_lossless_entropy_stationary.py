from __future__ import annotations

import pytest
import torch

from vortex_runtime.lossless_entropy_stationary import (
    EncodedTensor,
    EncodedTile,
    LLAMA405_BF16_GIB,
    LosslessEntropyStationaryError,
    block_linear_report,
    byte_entropy_bits,
    decode_tensor_zlib,
    encode_tensor_zlib,
    full_mlp_block_exactness,
    target_surface,
    tensor_bytes,
    word16_entropy_bits,
)


def test_constant_entropy_is_zero() -> None:
    assert byte_entropy_bits(bytes([7]) * 1024) == pytest.approx(0.0)
    assert word16_entropy_bits((b"\x34\x12") * 512) == pytest.approx(0.0)


def test_zlib_roundtrip_is_bit_exact() -> None:
    torch.manual_seed(9)
    source = torch.randn(33, 17, dtype=torch.float32).to(torch.bfloat16)
    encoded = encode_tensor_zlib(source, tile_bytes=128)
    decoded, _ns = decode_tensor_zlib(encoded, dtype=torch.bfloat16)
    assert tensor_bytes(decoded) == tensor_bytes(source)
    assert encoded.compression_ratio > 0


def test_corrupted_encoded_tile_fails_closed() -> None:
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8).to(torch.bfloat16)
    encoded = encode_tensor_zlib(source, tile_bytes=32)
    first = encoded.tiles[0]
    payload = bytearray(first.payload)
    payload[0] ^= 0x01
    corrupted_first = EncodedTile(
        raw_bytes=first.raw_bytes,
        encoded_bytes=first.encoded_bytes,
        raw_sha256=first.raw_sha256,
        encoded_sha256=first.encoded_sha256,
        codec=first.codec,
        payload=bytes(payload),
    )
    corrupted = EncodedTensor(
        shape=encoded.shape,
        dtype=encoded.dtype,
        raw_bytes=encoded.raw_bytes,
        encoded_bytes=encoded.encoded_bytes,
        raw_sha256=encoded.raw_sha256,
        byte_entropy_bits=encoded.byte_entropy_bits,
        word16_entropy_bits=encoded.word16_entropy_bits,
        shannon_word16_bytes=encoded.shannon_word16_bytes,
        tiles=(corrupted_first, *encoded.tiles[1:]),
        encode_ns=encoded.encode_ns,
    )
    with pytest.raises(LosslessEntropyStationaryError):
        decode_tensor_zlib(corrupted, dtype=torch.bfloat16)


def test_block_linear_matches_sequential_reference() -> None:
    torch.manual_seed(10)
    x = torch.randn(16, 12, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(20, 12, dtype=torch.float32).to(torch.bfloat16)
    report = block_linear_report(x, w, repeats=1)
    assert report.mismatch_count == 0
    assert report.vector_exact_fraction == 1.0


def test_full_mlp_block_matches_sequential_reference() -> None:
    torch.manual_seed(11)
    x = torch.randn(8, 4, dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(6, 4, dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(6, 4, dtype=torch.float32).to(torch.bfloat16)
    down = torch.randn(4, 6, dtype=torch.float32).to(torch.bfloat16)
    report = full_mlp_block_exactness(
        x,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        act_fn=torch.nn.SiLU(),
    )
    assert report["mismatch_count"] == 0
    assert report["vector_exact_fraction"] == 1.0


def test_target_surface_has_expected_monotonicity() -> None:
    rows = target_surface(compression_ratios=[1.0, 2.0], block_sizes=[16, 32])
    q4 = {
        (row.compression_ratio, row.block_size): row
        for row in rows
        if row.source_format == "ideal_q4"
    }
    assert q4[(2.0, 16)].io_ms_per_token < q4[(1.0, 16)].io_ms_per_token
    assert q4[(1.0, 32)].io_ms_per_token < q4[(1.0, 16)].io_ms_per_token
    assert q4[(2.0, 32)].dense_compute_floor_ms_per_token == pytest.approx(
        q4[(1.0, 16)].dense_compute_floor_ms_per_token
    )
    assert LLAMA405_BF16_GIB > 700
