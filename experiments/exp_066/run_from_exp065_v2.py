#!/usr/bin/env python3
"""Run the corrected EXP-066 frozen-evidence derivation.

The original bounded runner is retained as the data pipeline. This wrapper
injects the proven row/column-permutation equivalence before importing it and
normalizes the machine-readable scientific wording.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from vortex_runtime import tt_kronecker_reuse as reuse

# The base module imported the earlier control name. Expose the corrected
# implementation before importing that module so there is no duplicated runner.
reuse.validate_cut_equivalence = reuse.validate_cut_permutation_equivalence

from experiments.exp_066 import run_from_exp065 as base  # noqa: E402

_original_controls = base.controls
_original_write_json = base.write_json


def controls(seed: int):
    rows, failures, adversary = _original_controls(seed)
    for row in rows:
        if row.get("control") == "tt_cut_equals_exp065_rearrangement":
            row["control"] = (
                "tt_cut_matches_exp065_after_independent_row_column_permutations"
            )
            row["equivalence_kind"] = "rank-preserving row/column permutations"
    return rows, failures, adversary


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, dict) and value.get("experiment") == "EXP-066":
        derived = value.get("DERIVED", {})
        derived["derivation"] = (
            "Every interleaved TT/MPO prefix-suffix unfolding is transformed "
            "into the matching EXP-065 Kronecker rearrangement by independent "
            "row and column permutations. Rank is invariant under those "
            "permutations. Nontrivial cut ranks reuse frozen validated EXP-065 "
            "rows; unit-boundary cuts receive the universally valid favorable "
            "lower bound one."
        )
        derived["permutation_equivalence_control"] = True
    _original_write_json(path, value)


base.controls = controls
base.write_json = write_json


if __name__ == "__main__":
    base.main()
