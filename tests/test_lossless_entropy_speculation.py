from __future__ import annotations

import numpy as np

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.lossless_entropy_speculation import (
    decode_lossless_tile,
    encode_lossless_tile,
    entropy_speculation_budget,
    maximum_resident_bits_per_weight,
    measure_lossless_tile,
)


def _model(parameters: int, *, weight_bits: int) -> ModelSpec:
    return ModelSpec(
        parameters=parameters,
        layers=4,
        hidden_size=128,
        intermediate_size=256,
        attention_heads=8,
        kv_heads=2,
        vocab_size=1024,
        context_tokens=64,
        weight_bits=weight_bits,
        kv_bits=4,
    )


def test_lossless_codec_round_trips_exact_float16_bits() -> None:
    generator = np.random.default_rng(25001)
    values = generator.normal(size=16_384).astype(np.float16)
    raw = values.tobytes()
    encoded = encode_lossless_tile(raw, compression_level=6)
    restored = decode_lossless_tile(encoded)
    assert restored == raw


def test_byte_shuffle_compresses_repeated_exponent_patterns() -> None:
    values = np.linspace(-1.0, 1.0, 32_768, dtype=np.float16)
    stats = measure_lossless_tile(values.tobytes(), compression_level=6)
    assert stats.values == values.size
    assert stats.encoded_bytes < stats.raw_bytes
    assert stats.bits_per_value < 16.0
    assert stats.transform in {"raw", "byte_shuffle", "xor16_byte_shuffle"}


def test_speculation_budget_rewards_long_exact_commit() -> None:
    target = _model(405_000_000_000, weight_bits=16)
    baseline = _model(4_000_000_000, weight_bits=4)
    short = entropy_speculation_budget(
        target=target,
        baseline=baseline,
        compressed_bits_per_weight=1.0,
        exact_source_bits=16,
        verified_positions=16,
        committed_tokens=16,
    )
    long = entropy_speculation_budget(
        target=target,
        baseline=baseline,
        compressed_bits_per_weight=1.0,
        exact_source_bits=16,
        verified_positions=512,
        committed_tokens=512,
    )
    assert long.ideal_seconds_per_committed_token < short.ideal_seconds_per_committed_token
    assert long.minimum_straight_commit_ideal is not None
    assert long.compressed_stream_gib > 40.0


def test_tree_expansion_charges_extra_target_compute() -> None:
    target = _model(405_000_000_000, weight_bits=16)
    baseline = _model(4_000_000_000, weight_bits=4)
    straight = entropy_speculation_budget(
        target=target,
        baseline=baseline,
        compressed_bits_per_weight=0.5,
        exact_source_bits=16,
        verified_positions=128,
        committed_tokens=128,
    )
    tree = entropy_speculation_budget(
        target=target,
        baseline=baseline,
        compressed_bits_per_weight=0.5,
        exact_source_bits=16,
        verified_positions=512,
        committed_tokens=128,
    )
    assert tree.verification_expansion == 4.0
    assert tree.compute_seconds_per_pass > straight.compute_seconds_per_pass
    assert tree.ideal_seconds_per_committed_token >= straight.ideal_seconds_per_committed_token


def test_8gib_requires_sub_bit_exact_representation() -> None:
    bits = maximum_resident_bits_per_weight(
        parameters=405_000_000_000,
        resident_gib=6.0,
    )
    assert bits < 0.13
