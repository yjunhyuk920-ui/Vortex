from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.feasibility import default_specs
from vortex_runtime.weight_stationary_block import (
    StreamedBlockHardware,
    streamed_exact_block_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map the best-case one-pass exact weight-streaming frontier under "
            "the 8 GiB and native-4B latency contracts."
        )
    )
    parser.add_argument("--maximum-block", type=int, default=8192)
    parser.add_argument("--target-ratio", type=float, default=1.2)
    parser.add_argument(
        "--host-bandwidths",
        default="24,32,64,128",
        help="comma-separated GiB/s values",
    )
    parser.add_argument(
        "--target-throughputs",
        default="80,120,160,240",
        help="comma-separated effective tensor TFLOP/s values",
    )
    parser.add_argument("--operator-tile-gib", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("weight_stationary_gate0.json"))
    return parser.parse_args()


def parse_values(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("frontier values must be positive")
    return values


def main() -> None:
    args = parse_args()
    if args.maximum_block <= 0:
        raise SystemExit("maximum block must be positive")
    target, baseline = default_specs()
    bandwidths = parse_values(args.host_bandwidths)
    throughputs = parse_values(args.target_throughputs)

    points: list[dict[str, object]] = []
    for bandwidth in bandwidths:
        for throughput in throughputs:
            hardware = StreamedBlockHardware(
                host_to_device_gib_s=bandwidth,
                target_tensor_tflops=throughput,
                operator_tile_gib=args.operator_tile_gib,
            )
            memory_maximum = 0
            ideal_blocks: list[int] = []
            serialized_blocks: list[int] = []
            representative = None
            for block in range(1, args.maximum_block + 1):
                budget = streamed_exact_block_budget(
                    target=target,
                    baseline=baseline,
                    draft_positions=block,
                    committed_tokens=block,
                    target_passes=1,
                    hardware=hardware,
                    target_ratio=args.target_ratio,
                )
                if budget.memory_pass:
                    memory_maximum = block
                    representative = budget
                else:
                    break
                if budget.ideal_pass:
                    ideal_blocks.append(block)
                if budget.serialized_pass:
                    serialized_blocks.append(block)
            if representative is None:
                raise RuntimeError("even a one-token block exceeds device memory")
            points.append(
                {
                    "host_to_device_gib_s": bandwidth,
                    "target_tensor_tflops": throughput,
                    "maximum_memory_block": memory_maximum,
                    "ideal_feasible_range": (
                        None
                        if not ideal_blocks
                        else [min(ideal_blocks), max(ideal_blocks)]
                    ),
                    "serialized_feasible_range": (
                        None
                        if not serialized_blocks
                        else [min(serialized_blocks), max(serialized_blocks)]
                    ),
                    "memory_at_maximum_block": {
                        "peak_device_gib": representative.peak_device_gib,
                        "weight_buffer_gib": representative.weight_buffer_gib,
                        "kv_cache_gib": representative.kv_cache_gib,
                        "activation_workspace_gib": (
                            representative.activation_workspace_gib
                        ),
                    },
                    "one_pass_requirements_at_maximum_block": {
                        "required_target_tensor_tflops": (
                            representative.required_target_tensor_tflops_at_full_commit
                        ),
                        "required_host_to_device_gib_s": (
                            representative.required_host_to_device_gib_s_at_full_commit
                        ),
                    },
                }
            )

    default_point = next(
        point
        for point in points
        if point["host_to_device_gib_s"] == 24.0
        and point["target_tensor_tflops"] == 80.0
    )
    any_serialized = any(point["serialized_feasible_range"] for point in points)
    payload = {
        "evidence_level": "E0 exact dense full-stream lower bound",
        "target": {
            "parameters": target.parameters,
            "weight_bits": target.weight_bits,
            "weight_gib": target.weight_bytes / (1024**3),
            "context_tokens": target.context_tokens,
        },
        "baseline": {
            "parameters": baseline.parameters,
            "weight_bits": baseline.weight_bits,
        },
        "device_memory_gib": 8.0,
        "operator_tile_gib": args.operator_tile_gib,
        "target_ratio": args.target_ratio,
        "points": points,
        "default_hardware": default_point,
        "decision": (
            "exact full-weight streaming has a serialized feasible region in the tested grid"
            if any_serialized
            else "reject serialized exact full-weight streaming in the tested hardware grid"
        ),
        "architecture_implication": (
            "If the default point has no ideal range, a target pass must avoid "
            "either full checkpoint traffic or full dense arithmetic; improving "
            "the token proposal alone cannot close Gate 0."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
