from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.causal_residual_atlas import derive_atlas_budget


def derive() -> dict[str, object]:
    scenarios = [
        derive_atlas_budget(
            rank=rank,
            requested_cold_fraction=cold_fraction,
            page_columns=64,
            capsule_scalar_bytes=2,
            metadata_bytes_per_block=4,
            service_tokens=64,
        ).to_dict()
        for rank, cold_fraction in (
            (8, "0.002"),
            (8, "0.005"),
            (16, "0.002"),
            (16, "0.005"),
            (32, "0.002"),
        )
    ]
    return {
        "classification": "E0_DERIVED_INFORMATION_SOURCE_SCREEN_ONLY",
        "candidate": "Causal Residual Atlas",
        "capsule_semantics": (
            "exact committed-prefix input/image subspace plus a fully "
            "retained residual"
        ),
        "scenarios": scenarios,
        "unverified": [
            "real causal residual concentration",
            "end-to-end output certificate coverage",
            "capsule numerical enclosure",
            "cold-page physical latency",
            "complete 8 GiB branch peak",
            "405B execution and quality",
        ],
    }


def main() -> None:
    print(json.dumps(derive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
