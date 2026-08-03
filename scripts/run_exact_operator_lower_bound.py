from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.exact_operator_lower_bound import (
    exact_operator_information_budget,
    exhaustive_single_skip_adversaries,
)


TARGET_PARAMETERS = 405_849_243_648


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an exact dense-operator information certificate and "
            "exhaustive skipped-coordinate top-1 adversaries."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exact_operator_lower_bound.json"),
    )
    parser.add_argument(
        "--resident-gib",
        type=float,
        default=8.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resident_gib < 0:
        raise ValueError("resident memory must be nonnegative")

    budgets = {
        name: exact_operator_information_budget(
            parameter_count=TARGET_PARAMETERS,
            bits_per_parameter=bits,
            resident_gib=args.resident_gib,
        ).to_dict()
        for name, bits in (("q4", 4), ("q8", 8), ("fp16", 16))
    }

    shapes = [(2, 4), (3, 5), (4, 7), (8, 8)]
    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    total_coordinates = 0
    passing_coordinates = 0
    for rows, columns in shapes:
        summary, cases = exhaustive_single_skip_adversaries(
            rows=rows,
            columns=columns,
        )
        summaries.append(summary.to_dict())
        total_coordinates += summary.total_coordinates
        passing_coordinates += summary.passing_coordinates
        examples.append(cases[-1].to_dict())

    coverage = passing_coordinates / max(total_coordinates, 1)
    q4 = budgets["q4"]
    exact_output_bound_pass = bool(
        q4["exact_information_exceeds_resident"]
        and q4["minimum_external_information_gib"] > 0
    )
    adversary_pass = bool(
        coverage == 1.0
        and all(item["passes"] for item in summaries)
    )

    payload = {
        "experiment": "040_exact_operator_lower_bound",
        "evidence_level": "E1 formal and executable worst-case certificate",
        "target": {
            "parameter_count": TARGET_PARAMETERS,
            "resident_gib": args.resident_gib,
            "native_baseline_parameter_count": 4_000_000_000,
        },
        "exact_output_information_theorem": {
            "checkpoint_cardinality_log2": {
                name: budget["exact_information_bits"]
                for name, budget in budgets.items()
            },
            "minimum_lossless_representation_bits_equals_parameter_codes": True,
            "proof_method": (
                "injectivity/cardinality: 2^(N*b) arbitrary checkpoints require "
                "at least N*b bits in the worst-case exact representation"
            ),
            "budgets": budgets,
            "passes": exact_output_bound_pass,
        },
        "skipped_coordinate_top1_adversary": {
            "model": (
                "coordinate-query runtime with one checkpoint coordinate neither "
                "inspected nor represented by available checkpoint metadata"
            ),
            "shapes": summaries,
            "tested_coordinates": total_coordinates,
            "passing_coordinates": passing_coordinates,
            "coverage": coverage,
            "example_cases": examples,
            "passes": adversary_pass,
        },
        "scope_separation": {
            "exact_output_nb_bit_lower_bound_proven": True,
            "every_unrepresented_coordinate_can_flip_top1_proven": True,
            "metadata_aware_top1_nb_bit_lower_bound_proven": False,
            "measured_gpu_wall_clock_proven": False,
            "real_405b_execution_performed": False,
        },
        "conclusion": {
            "certificate_passes": bool(exact_output_bound_pass and adversary_pass),
            "arbitrary_dense_exact_output_with_only_8gib_information_compatible": False,
            "arbitrary_coordinate_omission_safe_for_universal_exact_top1": False,
            "top1_only_with_checkpoint_specific_compressed_metadata": (
                "not closed by this gate; requires a metadata-aware decision-function "
                "information bound or a constructive exact representation"
            ),
            "q4_minimum_external_information_gib": q4[
                "minimum_external_information_gib"
            ],
            "q4_dense_compute_gflop": q4["dense_compute_gflop"],
            "q4_compute_ratio_to_4b": q4["compute_ratio_to_baseline"],
            "fixed_target_status": (
                "contradicted for arbitrary dense exact operator output; exact "
                "top-1-only remains conditionally open only through a charged "
                "checkpoint-specific decision representation"
            ),
        },
        "next_proof_obligation": (
            "derive a metadata-aware information lower bound for the complete "
            "top-1 decision function of quantized dense transformers, or narrow "
            "universality to a measured structured checkpoint family"
        ),
    }

    if not payload["conclusion"]["certificate_passes"]:
        raise RuntimeError("lower-bound certificate did not pass its own checks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
