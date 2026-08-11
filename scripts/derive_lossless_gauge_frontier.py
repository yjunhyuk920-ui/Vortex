r"""Derive the finite E0 frontier for lossless streaming and attention gauges.

This script deliberately grants each candidate more power than a deployable
VORTEX implementation.  It asks only whether the candidate can survive the
first target-scale byte/parameter ceiling before any model or hardware work.

Run:

    python scripts/derive_lossless_gauge_frontier.py \
        --output-dir results/e0_lossless_gauge_frontier
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


PARAMETERS = 405_849_243_648
LAYERS = 126
HIDDEN_SIZE = 16_384
QUERY_HEADS = 128
KV_HEADS = 8
HEAD_DIM = 128

SHANNON_BF16_BITS = Fraction(53, 5)  # 10.6 bits/parameter
ZIPSERV_BF16_BITS = Fraction(113, 10)  # 11.3 bits/parameter
FAST_LINK_BYTES_PER_SECOND = 32_000_000_000
TARGET_LINK_BYTES_PER_SECOND = 8_000_000_000
TOKEN_SECONDS = Fraction(1, 50)
REFERENCE_BLOCK_TOKENS = 32
ALLOWED_DENSE_FRACTION = Fraction(118_270_517, 10_000_000_000)


def as_float(value: Fraction) -> float:
    return value.numerator / value.denominator


def build_summary() -> dict[str, object]:
    shannon_bytes = Fraction(PARAMETERS) * SHANNON_BF16_BITS / 8
    zipserv_bytes = Fraction(PARAMETERS) * ZIPSERV_BF16_BITS / 8

    def link_case(bytes_per_second: int) -> dict[str, float | int]:
        sweep_seconds = shannon_bytes / bytes_per_second
        minimum_tokens = math.ceil(sweep_seconds / TOKEN_SECONDS)
        block_ms_per_token = (
            sweep_seconds / REFERENCE_BLOCK_TOKENS * 1_000
        )
        return {
            "bytes_per_second": bytes_per_second,
            "sweep_seconds": as_float(sweep_seconds),
            "minimum_zero_compute_tokens_for_20ms": minimum_tokens,
            "zero_compute_ms_per_token_at_32_tokens": as_float(
                block_ms_per_token
            ),
        }

    q_parameters_per_layer = HIDDEN_SIZE * HIDDEN_SIZE
    k_parameters_per_layer = KV_HEADS * HEAD_DIM * HIDDEN_SIZE
    v_parameters_per_layer = k_parameters_per_layer
    o_parameters_per_layer = HIDDEN_SIZE * HIDDEN_SIZE
    all_attention_parameters = LAYERS * (
        q_parameters_per_layer
        + k_parameters_per_layer
        + v_parameters_per_layer
        + o_parameters_per_layer
    )
    all_attention_fraction = Fraction(all_attention_parameters, PARAMETERS)
    required_elimination = 1 - ALLOWED_DENSE_FRACTION

    # A deliberately generous gauge count: full GL(head_dim) freedom in both
    # Q/K and V/O for every query head and every layer.  RoPE and GQA only
    # shrink the legal group, so this is an upper ceiling, not a claim about
    # the actual Llama gauge dimension.
    generous_gauge_dimensions = (
        2 * LAYERS * QUERY_HEADS * HEAD_DIM * HEAD_DIM
    )

    summary: dict[str, object] = {
        "contract": {
            "parameters": PARAMETERS,
            "allowed_dense_fraction": as_float(ALLOWED_DENSE_FRACTION),
            "required_elimination_fraction": as_float(required_elimination),
            "token_seconds": as_float(TOKEN_SECONDS),
        },
        "lossless_streaming": {
            "shannon_bf16_bits_per_parameter": as_float(
                SHANNON_BF16_BITS
            ),
            "zipserv_bf16_bits_per_parameter": as_float(
                ZIPSERV_BF16_BITS
            ),
            "shannon_model_bytes": as_float(shannon_bytes),
            "zipserv_model_bytes": as_float(zipserv_bytes),
            "fast_link": link_case(FAST_LINK_BYTES_PER_SECOND),
            "target_link_favorable_ceiling": link_case(
                TARGET_LINK_BYTES_PER_SECOND
            ),
        },
        "attention_reparameterization": {
            "generous_continuous_gauge_dimensions": generous_gauge_dimensions,
            "generous_gauge_fraction_of_parameters": generous_gauge_dimensions
            / PARAMETERS,
            "all_attention_parameters": all_attention_parameters,
            "all_attention_fraction_of_parameters": as_float(
                all_attention_fraction
            ),
            "remaining_fraction_after_free_attention_deletion": as_float(
                1 - all_attention_fraction
            ),
        },
        "decisions": [
            "REJECT_OMEGA_CUTSUM_AS_EXACT_BILINEAR_RELABELING",
            "REJECT_OMEGA_ZIPWAVE_AS_STANDALONE_CORE",
            "REJECT_ATTENTION_ONLY_GAUGE_FUSION_AT_FAVORABLE_CEILING",
            "KEEP_GENERAL_NONLINEAR_ADAPTIVE_RANK_ONE_PROBE_GAP_OPEN",
            "NO_SURVIVING_CANDIDATE",
        ],
    }

    assert ZIPSERV_BF16_BITS >= SHANNON_BF16_BITS
    assert summary["lossless_streaming"]["fast_link"][  # type: ignore[index]
        "minimum_zero_compute_tokens_for_20ms"
    ] > REFERENCE_BLOCK_TOKENS
    assert all_attention_fraction < required_elimination
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = build_summary()
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            rendered, encoding="utf-8"
        )
    print(rendered, end="")


if __name__ == "__main__":
    main()
