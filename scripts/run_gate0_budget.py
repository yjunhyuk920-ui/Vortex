from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.gate0_budget import (
    BaselineMeasurement,
    ProjectedCapsuleCandidate,
    calculate_gate0_certificate,
    llama_31_405b_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the VORTEX Architecture Gate 0 certificate."
    )
    parser.add_argument("--output", type=Path, default=Path("gate0_budget.json"))
    parser.add_argument("--target-amortization", type=float, default=512.0)
    parser.add_argument("--baseline-traffic-gib", type=float, default=2.0)
    parser.add_argument("--baseline-compute-gflops", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = BaselineMeasurement(
        name="native-4B-Q4 proxy",
        traffic_gib_per_token=args.baseline_traffic_gib,
        compute_gflops_per_token=args.baseline_compute_gflops,
        source=(
            "conservative_proxy_not_measured"
            if args.baseline_traffic_gib == 2.0
            and args.baseline_compute_gflops == 8.0
            else "user_supplied"
        ),
    )
    candidate = ProjectedCapsuleCandidate(
        target_amortized_tokens_per_full_stream=args.target_amortization
    )
    certificate = calculate_gate0_certificate(
        llama_31_405b_geometry(), candidate, baseline
    )
    payload = certificate.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
