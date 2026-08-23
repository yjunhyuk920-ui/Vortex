from __future__ import annotations

import numpy as np

from vortex_runtime.native_rounding_page_certificate import (
    bf16_round,
    certify_page_termwise_noop,
    float32_to_bf16_bits,
    native_separate_mul_add,
    simulate_row_page_certificate,
    traffic_ledger,
)


def test_bf16_round_is_idempotent() -> None:
    values = np.array([-3.14159, -0.0, 0.0, 1.0, 2.0 ** -130, 65504.0], dtype=np.float32)
    once = bf16_round(values)
    twice = bf16_round(once)
    assert np.array_equal(once.view(np.uint32), twice.view(np.uint32))


def test_certificate_positive_control_is_sound() -> None:
    accumulator = np.float32(2.0 ** 24)
    weights = bf16_round(np.full(64, 2.0 ** -20, dtype=np.float32)).astype(np.float32)
    activations = bf16_round(np.ones(64, dtype=np.float32)).astype(np.float32)
    exponent_code = int((((float32_to_bf16_bits(weights) >> 7) & 0xFF).max()))
    assert certify_page_termwise_noop(accumulator, exponent_code, 1.0)
    candidate = accumulator
    for weight, activation in zip(weights, activations, strict=True):
        candidate = native_separate_mul_add(candidate, weight, activation)
    assert candidate.view(np.uint32) == accumulator.view(np.uint32)


def test_all_ones_never_skips_and_matches_reference() -> None:
    ones = np.ones(4096, dtype=np.float32)
    result = simulate_row_page_certificate(ones, ones, 128)
    assert result["exact_match"] is True
    assert result["page_read_fraction"] == 1.0
    assert result["term_execution_fraction"] == 1.0


def test_metadata_floor_requires_over_99_percent_page_skip() -> None:
    for page_size in (64, 128, 256, 512, 1024):
        ledger = traffic_ledger(page_size, 1, 0.011851851851851851)
        assert ledger.required_skip_fraction > 0.98
        assert ledger.metadata_fraction < 0.01


def test_zero_weights_can_skip_after_nonzero_seed() -> None:
    weights = np.concatenate((np.array([2.0 ** 24], dtype=np.float32), np.zeros(1024, dtype=np.float32)))
    activations = np.ones_like(weights)
    result = simulate_row_page_certificate(weights, activations, 128)
    assert result["exact_match"] is True
    assert result["pages_skipped"] > 0


def test_certificate_never_claims_zero_accumulator_for_nonzero_products() -> None:
    weights = bf16_round(np.array([1.0, -1.0], dtype=np.float32)).astype(np.float32)
    exponent_code = int((((float32_to_bf16_bits(weights) >> 7) & 0xFF).max()))
    assert not certify_page_termwise_noop(np.float32(0.0), exponent_code, 1.0)
