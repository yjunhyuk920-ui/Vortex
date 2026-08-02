from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import require_transformers
from vortex_runtime.feasibility import default_specs
from vortex_runtime.lossless_entropy_speculation import (
    entropy_speculation_budget,
    maximum_resident_bits_per_weight,
    measure_lossless_tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact FP16 checkpoint tile compressibility and project the "
            "speculative commit length required to amortize one lossless 405B pass."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tile-values", type=int, default=65_536)
    parser.add_argument("--sample-values", type=int, default=8_388_608)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--host-to-device-gib-s", type=float, default=24.0)
    parser.add_argument("--decompression-output-gib-s", type=float, default=1_000.0)
    parser.add_argument("--target-tensor-tflops", type=float, default=160.0)
    parser.add_argument("--baseline-memory-gib-s", type=float, default=300.0)
    parser.add_argument("--target-ratio", type=float, default=1.2)
    parser.add_argument("--measured-candidate-depth", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lossless_entropy_speculation.json"),
    )
    return parser.parse_args()


def unique_matrix_weights(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2:
            continue
        identity = (
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append((name, parameter))
    if not result:
        raise RuntimeError("model exposed no unique matrix weights")
    return result


def sampled_segments(
    tensor: torch.Tensor,
    *,
    maximum_values: int,
) -> list[torch.Tensor]:
    flat = tensor.detach().reshape(-1)
    count = min(maximum_values, flat.numel())
    if count <= 0:
        return []
    if count == flat.numel():
        return [flat]
    segments: list[torch.Tensor] = []
    segment_count = 3
    base = count // segment_count
    remainder = count - base * segment_count
    lengths = [base, base, base + remainder]
    available = flat.numel()
    starts = [
        0,
        max(0, (available - lengths[1]) // 2),
        max(0, available - lengths[2]),
    ]
    for start, length in zip(starts, lengths):
        if length:
            segments.append(flat[start : start + length])
    return segments


def main() -> None:
    args = parse_args()
    if min(args.tile_values, args.sample_values, args.measured_candidate_depth) <= 0:
        raise SystemExit("tile, sample and candidate-depth values must be positive")
    if not 0 <= args.compression_level <= 9:
        raise SystemExit("compression level must be in [0, 9]")

    AutoModelForCausalLM, _ = require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    model.eval()
    weights = unique_matrix_weights(model)
    allocation = max(1, args.sample_values // len(weights))

    started = time.perf_counter()
    total_values = 0
    total_raw_bytes = 0
    total_encoded_bytes = 0
    weighted_byte_entropy = 0.0
    weighted_symbol_entropy = 0.0
    weighted_xor_entropy = 0.0
    transforms: Counter[str] = Counter()
    per_tensor: list[dict[str, int | float | str]] = []

    for name, parameter in weights:
        tensor_values = 0
        tensor_raw = 0
        tensor_encoded = 0
        tensor_transforms: Counter[str] = Counter()
        for segment in sampled_segments(parameter, maximum_values=allocation):
            source = segment.to("cpu", torch.float16).contiguous()
            for start in range(0, source.numel(), args.tile_values):
                end = min(start + args.tile_values, source.numel())
                tile = source[start:end].contiguous().numpy().tobytes()
                stats = measure_lossless_tile(
                    tile,
                    compression_level=args.compression_level,
                )
                total_values += stats.values
                total_raw_bytes += stats.raw_bytes
                total_encoded_bytes += stats.encoded_bytes
                weighted_byte_entropy += (
                    stats.byte_plane_entropy_bits_per_value * stats.values
                )
                weighted_symbol_entropy += (
                    stats.symbol_entropy_bits_per_value * stats.values
                )
                weighted_xor_entropy += (
                    stats.xor_symbol_entropy_bits_per_value * stats.values
                )
                transforms[stats.transform] += 1
                tensor_values += stats.values
                tensor_raw += stats.raw_bytes
                tensor_encoded += stats.encoded_bytes
                tensor_transforms[stats.transform] += 1
        if tensor_values:
            per_tensor.append(
                {
                    "name": name,
                    "sampled_values": tensor_values,
                    "bits_per_value": tensor_encoded * 8 / tensor_values,
                    "compression_ratio": tensor_raw / tensor_encoded,
                    "dominant_transform": tensor_transforms.most_common(1)[0][0],
                }
            )

    if total_values == 0:
        raise RuntimeError("no checkpoint values were sampled")
    measured_bits = total_encoded_bytes * 8 / total_values
    measured_ratio = total_raw_bytes / total_encoded_bytes
    byte_entropy = weighted_byte_entropy / total_values
    symbol_entropy = weighted_symbol_entropy / total_values
    xor_entropy = weighted_xor_entropy / total_values

    target, baseline = default_specs()
    commit_lengths = [12, 32, 64, 128, 256, 512, 1_024]
    expansion_factors = [1, 2, 4]
    measured_frontier: list[dict[str, object]] = []
    for expansion in expansion_factors:
        for committed in commit_lengths:
            budget = entropy_speculation_budget(
                target=target,
                baseline=baseline,
                compressed_bits_per_weight=min(measured_bits, 16.0),
                exact_source_bits=16,
                verified_positions=committed * expansion,
                committed_tokens=committed,
                host_to_device_gib_s=args.host_to_device_gib_s,
                decompression_output_gib_s=args.decompression_output_gib_s,
                target_tensor_tflops=args.target_tensor_tflops,
                baseline_memory_gib_s=args.baseline_memory_gib_s,
                target_ratio=args.target_ratio,
            )
            measured_frontier.append(budget.to_dict())

    theoretical_frontier: list[dict[str, object]] = []
    for bits_per_weight in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        budget = entropy_speculation_budget(
            target=target,
            baseline=baseline,
            compressed_bits_per_weight=bits_per_weight,
            exact_source_bits=16,
            verified_positions=args.measured_candidate_depth,
            committed_tokens=args.measured_candidate_depth,
            host_to_device_gib_s=args.host_to_device_gib_s,
            decompression_output_gib_s=args.decompression_output_gib_s,
            target_tensor_tflops=args.target_tensor_tflops,
            baseline_memory_gib_s=args.baseline_memory_gib_s,
            target_ratio=args.target_ratio,
        )
        theoretical_frontier.append(budget.to_dict())

    measured_depth_budget = entropy_speculation_budget(
        target=target,
        baseline=baseline,
        compressed_bits_per_weight=min(measured_bits, 16.0),
        exact_source_bits=16,
        verified_positions=args.measured_candidate_depth,
        committed_tokens=args.measured_candidate_depth,
        host_to_device_gib_s=args.host_to_device_gib_s,
        decompression_output_gib_s=args.decompression_output_gib_s,
        target_tensor_tflops=args.target_tensor_tflops,
        baseline_memory_gib_s=args.baseline_memory_gib_s,
        target_ratio=args.target_ratio,
    )
    resident_bits_limit = maximum_resident_bits_per_weight(
        parameters=target.parameters,
        resident_gib=6.0,
    )

    required_serialized = measured_depth_budget.minimum_straight_commit_serialized
    best_tested = min(
        measured_frontier,
        key=lambda item: item["serialized_seconds_per_committed_token"],
    )
    qualifies = bool(
        best_tested["serialized_pass"]
        and measured_bits <= resident_bits_limit
    )
    payload = {
        "evidence_level": "E2 sampled pretrained lossless tile entropy; E0 405B projection",
        "model": args.model,
        "source_precision_bits": 16,
        "unique_matrix_tensors": len(weights),
        "sampled_values": total_values,
        "sampled_raw_bytes": total_raw_bytes,
        "sampled_encoded_bytes": total_encoded_bytes,
        "measured_bits_per_weight": measured_bits,
        "measured_compression_ratio": measured_ratio,
        "empirical_zero_order_bounds": {
            "byte_plane_bits_per_weight": byte_entropy,
            "uint16_symbol_bits_per_weight": symbol_entropy,
            "xor_uint16_symbol_bits_per_weight": xor_entropy,
        },
        "transform_tiles": dict(transforms),
        "per_tensor": sorted(
            per_tensor,
            key=lambda item: item["bits_per_value"],
        ),
        "hardware": {
            "host_to_device_gib_s": args.host_to_device_gib_s,
            "decompression_output_gib_s": args.decompression_output_gib_s,
            "target_tensor_tflops": args.target_tensor_tflops,
            "baseline_memory_gib_s": args.baseline_memory_gib_s,
            "target_ratio": args.target_ratio,
        },
        "projected_405b": {
            "resident_6gib_max_bits_per_weight": resident_bits_limit,
            "measured_candidate_depth": args.measured_candidate_depth,
            "measured_depth_budget": measured_depth_budget.to_dict(),
            "minimum_straight_commit_serialized_at_measured_rate": required_serialized,
            "measured_frontier": measured_frontier,
            "theoretical_bit_rate_frontier_at_measured_depth": theoretical_frontier,
            "best_tested_point": best_tested,
        },
        "contract": (
            "Every sampled tile is bit-exact after decode. The 405B projection "
            "charges compressed host transfer, exact FP16 decompression output, "
            "all verified target positions and only actually committed tokens."
        ),
        "qualifies": qualifies,
        "decision": (
            "advance lossless resident speculation"
            if qualifies
            else "retain entropy codec as a systems optimization, not a Gate 0 solution"
        ),
        "next_candidate_if_rejected": (
            "measure a real resident drafter's accepted straight-run length; lossless "
            "compression can only promote when that length exceeds the calculated "
            "minimum without a large verification tree expansion"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
