from __future__ import annotations

import json
import math
from pathlib import Path

GIB = 1024**3


def matrix_vtx_bytes(out_features: int, in_features: int, tile_cols: int = 128) -> dict[str, int]:
    """Storage/read layout implemented by vortex_runtime.vtx_linear.

    Important: quant values are stored as torch.int8 regardless of base_bits.
    Residual and four metadata arrays (scale + three bounds) are float32.
    """
    elements = out_features * in_features
    tiles = math.ceil(in_features / tile_cols)
    rows_tiles = out_features * tiles
    return {
        "elements": elements,
        "int8_base": elements,
        "fp32_residual": elements * 4,
        "fp32_scales": rows_tiles * 4,
        "fp32_bounds": rows_tiles * 3 * 4,
    }


def add(target: dict[str, int], source: dict[str, int], multiplier: int = 1) -> None:
    for key, value in source.items():
        if key != "elements":
            target[key] = target.get(key, 0) + value * multiplier


def gib(value: int | float) -> float:
    return float(value) / GIB


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = json.loads((root / "validation_results.json").read_text(encoding="utf-8"))

    hidden = 16_384
    intermediate = 53_248
    kv_width = 1_024
    vocab = 128_256
    layers = 126
    tile_cols = 128

    matrices = {
        "q_proj": (hidden, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }

    internal = {}
    per_matrix = {}
    for name, (out_features, in_features) in matrices.items():
        layout = matrix_vtx_bytes(out_features, in_features, tile_cols)
        per_matrix[name] = {key: gib(value) for key, value in layout.items() if key != "elements"}
        add(internal, layout, layers)

    lm_head = matrix_vtx_bytes(vocab, hidden, tile_cols)
    all_progressive = dict(internal)
    add(all_progressive, lm_head)

    no_residual_read = (
        all_progressive["int8_base"]
        + all_progressive["fp32_scales"]
        + all_progressive["fp32_bounds"]
    )
    full_vtx_storage = sum(all_progressive.values())

    native_4b_q4 = 4_000_000_000 * 0.5
    minimum_amortization = no_residual_read / native_4b_q4
    jacobi_mean = float(validation["jacobi"]["mean_committed_block"])

    tiny_vocab = 257
    tiny_hidden = 64
    tiny_intermediate = 160
    tiny_layers = 4
    tiny_kv_width = 16
    tiny_per_layer = (
        tiny_hidden * tiny_hidden
        + 2 * tiny_kv_width * tiny_hidden
        + tiny_hidden * tiny_hidden
        + 2 * tiny_intermediate * tiny_hidden
        + tiny_hidden * tiny_intermediate
        + 2 * tiny_hidden
    )
    tiny_global = 2 * tiny_vocab * tiny_hidden + tiny_hidden
    tiny_params = tiny_layers * tiny_per_layer + tiny_global

    kv_bytes_per_token = layers * 2 * kv_width * 2

    report = {
        "audit_date": "2026-08-02",
        "source_revision": "3c486625ecea4e4692e1e5e4bcc22d3a59bf6e15",
        "verified_local_tests": 7,
        "prototype_test_model": {
            "parameters": tiny_params,
            "parameters_million": tiny_params / 1e6,
            "scale_gap_to_405_85b": 405_849_243_648 / tiny_params,
            "layers": tiny_layers,
            "hidden_size": tiny_hidden,
            "vocab_size": tiny_vocab,
        },
        "current_vtx_format": {
            "quant_storage": "int8 even when base_bits is 4, 5, or 6",
            "residual_storage": "float32",
            "scale_storage": "float32 per output-row/input-tile",
            "bound_storage": "three float32 values per output-row/input-tile",
            "tile_cols": tile_cols,
        },
        "projected_405b_vtx_if_applied_to_all_internal_linears_and_lm_head": {
            "int8_base_gib": gib(all_progressive["int8_base"]),
            "fp32_residual_gib": gib(all_progressive["fp32_residual"]),
            "fp32_scales_gib": gib(all_progressive["fp32_scales"]),
            "fp32_bounds_gib": gib(all_progressive["fp32_bounds"]),
            "total_vtx_storage_gib_excluding_embedding": gib(full_vtx_storage),
            "minimum_base_plus_metadata_read_per_target_pass_gib": gib(no_residual_read),
            "minimum_read_assumes_zero_residual_tiles": True,
        },
        "traffic_gate": {
            "native_4b_q4_weight_bytes_gib": gib(native_4b_q4),
            "minimum_tokens_needed_per_405b_weight_stream": minimum_amortization,
            "observed_tiny_jacobi_mean_committed_block": jacobi_mean,
            "amortization_gap_factor": minimum_amortization / jacobi_mean,
            "note": "Compute, KV, residual reads, and I/O inefficiency are excluded, so this is optimistic.",
        },
        "kv_cache_bf16": {
            "bytes_per_token": kv_bytes_per_token,
            "gib_2k": gib(kv_bytes_per_token * 2048),
            "gib_4k": gib(kv_bytes_per_token * 4096),
            "gib_8k": gib(kv_bytes_per_token * 8192),
            "gib_16k": gib(kv_bytes_per_token * 16384),
            "gib_128k": gib(kv_bytes_per_token * 131072),
        },
        "per_matrix_per_layer_vtx_gib": per_matrix,
        "engineering_verdict": {
            "8gb_residency_plumbing": "prototype only; CPU simulation, not measured CUDA residency",
            "405b_end_to_end": "not implemented",
            "4b_class_wall_clock": "not demonstrated and cannot be reached by scaling the current full-base VTX format unchanged",
            "required_architecture_change": "avoid reading the complete base matrix on normal tokens; add a smaller cached operator path plus measured exact fallback and block amortization",
        },
    }

    output = root / "current_architecture_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
