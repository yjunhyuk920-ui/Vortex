#!/usr/bin/env python3
"""Fail-closed integrity check for evidence imported by the E0 audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_audit_module(repo_root: Path) -> Any:
    path = repo_root / "experiments/e0_three_invention_validation/run_validation.py"
    spec = importlib.util.spec_from_file_location("three_invention_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify(repo_root: Path) -> dict[str, str]:
    audit = load_audit_module(repo_root)
    evidence = audit.EVIDENCE
    loaded: dict[str, Any] = {}
    verified: dict[str, str] = {}

    for key, row in evidence.items():
        path = repo_root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        blob_sha = git_blob_sha(path)
        if blob_sha != row["blob_sha"]:
            raise AssertionError(
                f"{key} blob drift: expected {row['blob_sha']} got {blob_sha}"
            )
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        verified[key] = blob_sha

    exp086 = loaded["exp_086a"]["aggregates"]["coordinate_k128"]
    assert exp086["input_residual_nonzero_fraction_p50"] == evidence["exp_086a"][
        "coordinate_k128_input_residual_nonzero_p50"
    ]
    assert exp086["down_residual_nonzero_fraction_p50"] == evidence["exp_086a"][
        "coordinate_k128_down_residual_nonzero_p50"
    ]
    assert exp086["projected_whole_model_operation_fraction_p50"] == evidence[
        "exp_086a"
    ]["coordinate_k128_operation_fraction_p50"]
    assert exp086["projected_whole_model_weight_fraction_p50"] == evidence[
        "exp_086a"
    ]["coordinate_k128_weight_fraction_p50"]
    assert exp086["reconstruction_mismatches"] == 0

    exp069 = loaded["exp_069"]["MEASURED"]
    assert exp069["p50_mandatory_operation_fraction_lower_bound"] == evidence[
        "exp_069"
    ]["p50_mandatory_operation_fraction"]
    assert exp069["p90_mandatory_operation_fraction_lower_bound"] == evidence[
        "exp_069"
    ]["p90_mandatory_operation_fraction"]
    assert exp069["total_verified_exact_replay_hits"] == evidence["exp_069"][
        "verified_exact_replay_hits"
    ]

    exp049 = loaded["exp_049"]["MEASURED"]
    assert exp049["triangular_one_position_per_round_barrier_observed"] is True
    assert exp049["triangular_transcript_indistinguishability_observed"] is True
    assert exp049["oracle_best_s1s2_p50_matching_prefix"] == evidence["exp_049"][
        "oracle_p50_matching_prefix_after_four_passes"
    ]
    assert exp049["oracle_best_s1s2_max_matching_prefix"] == evidence["exp_049"][
        "oracle_max_matching_prefix_after_four_passes"
    ]

    exp080 = loaded["exp_080a"]["DERIVED"]
    assert exp080["best_constructive_row"]["constructive_ratio"] == evidence[
        "exp_080a"
    ]["best_constructive_arithmetic_fraction"]
    assert exp080["best_constructive_row"]["block_length"] == evidence["exp_080a"][
        "best_constructive_block_length"
    ]
    assert exp080["constructive_joint_pass_count"] == evidence["exp_080a"][
        "constructive_joint_pass_count"
    ]

    shortcut = loaded["native_shortcut"]
    assert shortcut["temporal_identity"][
        "unlimited_same_coordinate_history_hit_fraction"
    ]["max"] == evidence["native_shortcut"][
        "unlimited_same_coordinate_history_hit_fraction"
    ]
    assert shortcut["exact_product_reuse"][
        "best_combined_local_elimination_upper"
    ] == evidence["native_shortcut"]["best_combined_local_elimination_upper"]
    assert shortcut["exact_product_reuse"]["unique_product_fraction"]["p50"] == evidence[
        "native_shortcut"
    ]["unique_product_fraction_p50"]
    return verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    verified = verify(args.repo_root.resolve())
    print(json.dumps(verified, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
