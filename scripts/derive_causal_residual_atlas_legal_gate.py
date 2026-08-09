#!/usr/bin/env python3
"""Print the frozen legal-pair/outward-bound Gate without loading a model."""

from __future__ import annotations

import json

from vortex_runtime.causal_residual_atlas_legal_gate import (
    derive_legal_pair_outward_gate,
)


def main() -> None:
    print(
        json.dumps(
            derive_legal_pair_outward_gate().to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
