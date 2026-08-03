from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.top1_function_information import (
    enumerate_top1_function_family,
    llama_operator_collection_bound,
    top1_family_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate injective exact top-1 classifier families and project "
            "their metadata lower bound to Llama-405B operator shapes."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/top1_function_information_bound.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enumeration_shapes = [(2, 2), (4, 4), (4, 5), (6, 6)]
    enumerations = [
        enumerate_top1_function_family(
            rows=rows,
            columns=columns,
            maximum_bits=10,
        )
        for rows, columns in enumeration_shapes
    ]
    if not all(item.passes for item in enumerations):
        raise RuntimeError("an exhaustive top-1 family failed injectivity")

    square = top1_family_shape(rows=16_384, columns=16_384)
    collection = llama_operator_collection_bound()
    payload = {
        "experiment": "041_top1_function_information_bound",
        "evidence_level": "E1 metadata-aware exact top-1 function certificate",
        "construction": {
            "family": "selector/payload row-pair classifier",
            "decision_bits_formula": "K = p * (d - p)",
            "pair_formula": "p = min(floor(m/2), floor(d/2))",
            "distinct_functions": "2^K",
            "minimum_exact_metadata_bits": "K",
            "proof_reason": (
                "each encoded bit is recovered by one selector/payload query; "
                "different bit tables produce different complete top-1 signatures"
            ),
        },
        "exhaustive_enumerations": [item.to_dict() for item in enumerations],
        "square_hidden_classifier": square.to_dict(),
        "llama_405b_operator_collection": collection.to_dict(),
        "scope_separation": {
            "direct_dense_classifier_top1_metadata_bound_proven": True,
            "independently_callable_operator_collection_bound_proven": True,
            "full_transformer_final_token_bound_proven": False,
            "real_405b_execution_performed": False,
            "measured_gpu_wall_clock_proven": False,
        },
        "conclusion": {
            "certificate_passes": bool(
                all(item.passes for item in enumerations)
                and collection.exceeds_resident_limit
                and not collection.full_transformer_top1_bound_proven
            ),
            "direct_classifier_minimum_metadata_bits": (
                "p*(d-p) for the constructed family"
            ),
            "operator_collection_total_gib": collection.total_gib,
            "operator_collection_exceeds_8gib": collection.exceeds_resident_limit,
            "operator_collection_excess_gib": (
                collection.total_gib - collection.resident_limit_gib
            ),
            "fixed_target_status": (
                "metadata-aware exact top-1 compression is lower-bounded above "
                "8 GiB for the independently callable Llama-shaped operator "
                "collection; an end-to-end Transformer routing theorem remains open"
            ),
        },
        "next_proof_obligation": (
            "construct a Llama-like residual/attention/MLP routing family whose "
            "final token winners independently expose the layerwise selector/"
            "payload bits, or show why architectural composition prevents the "
            "operator-collection bounds from adding"
        ),
    }
    if not payload["conclusion"]["certificate_passes"]:
        raise RuntimeError("top-1 function information certificate failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
