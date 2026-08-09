"""Post-hoc, read-only decision-direction screen for the frozen EXP-083B row.

This script performs no Transformer forward and writes no evidence.  It uses
the tied BF16 embedding/LM-head tensor directly from the pinned safetensors
shard, preserving the existing result bundle as the authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


PROMPT_ID = "legal_holdout_english_01"
TIED_HEAD_TENSOR = "model.language_model.embed_tokens.weight"
FROZEN_PROJECTION_RADIUS = 23.420520066618046


def _bf16_memmap(path: Path, tensor_name: str) -> np.memmap:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    metadata = header[tensor_name]
    if metadata["dtype"] != "BF16":
        raise ValueError("registered tied head must be BF16")
    shape = tuple(int(value) for value in metadata["shape"])
    start, _ = (int(value) for value in metadata["data_offsets"])
    return np.memmap(
        path,
        mode="r",
        dtype="<u2",
        offset=8 + header_length + start,
        shape=shape,
    )


def _bf16_to_float32(value: np.ndarray) -> np.ndarray:
    words = np.asarray(value, dtype=np.uint16).astype(np.uint32)
    return (words << 16).view(np.float32)


def _direction_screen(
    weight_words: np.ndarray,
    gain: np.ndarray,
    candidate: np.ndarray,
    native: np.ndarray,
    *,
    chunk_rows: int = 4096,
) -> dict[str, object]:
    rows = int(weight_words.shape[0])
    candidate_scores = np.empty(rows, dtype=np.float64)
    native_scores = np.empty(rows, dtype=np.float64)
    effective_norms = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        weight = _bf16_to_float32(weight_words[start:stop])
        effective = weight * gain
        candidate_scores[start:stop] = effective @ candidate
        native_scores[start:stop] = effective @ native
        effective_norms[start:stop] = np.linalg.norm(
            effective.astype(np.float64), axis=1
        )

    candidate_winner = int(np.argmax(candidate_scores))
    native_winner = int(np.argmax(native_scores))
    actual_radius = float(
        np.linalg.norm(native.astype(np.float64) - candidate.astype(np.float64))
    )
    margin = candidate_scores[candidate_winner] - candidate_scores
    screens: dict[str, object] = {}
    for name, radius in (
        ("actual_candidate_native_pre_norm_distance", actual_radius),
        ("frozen_projection_radius", FROZEN_PROJECTION_RADIUS),
    ):
        upper = (
            effective_norms[candidate_winner] + effective_norms
        ) * radius
        unresolved = margin <= upper
        unresolved[candidate_winner] = False
        screens[name] = {
            "radius": radius,
            "unresolved_competitors": int(np.count_nonzero(unresolved)),
            "competitors": rows - 1,
            "unresolved_fraction": float(np.count_nonzero(unresolved) / (rows - 1)),
        }

    top_two_candidate = np.partition(candidate_scores, -2)[-2:]
    top_two_native = np.partition(native_scores, -2)[-2:]
    return {
        "candidate_linear_winner": candidate_winner,
        "native_linear_winner": native_winner,
        "candidate_linear_top_two_gap": float(
            top_two_candidate.max() - top_two_candidate.min()
        ),
        "native_linear_top_two_gap": float(
            top_two_native.max() - top_two_native.min()
        ),
        "actual_pre_norm_distance": actual_radius,
        "effective_winner_row_norm": float(effective_norms[candidate_winner]),
        "effective_row_norm_minimum": float(effective_norms.min()),
        "effective_row_norm_maximum": float(effective_norms.max()),
        "screens": screens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/exp_083b"))
    parser.add_argument(
        "--model-shard",
        type=Path,
        default=Path(
            ".deps/exp076-model/model.safetensors-00001-of-00001.safetensors"
        ),
    )
    args = parser.parse_args()

    prompt_path = args.result_dir / "raw" / "prompt_arrays" / f"{PROMPT_ID}.npz"
    static_path = args.result_dir / "raw" / "static_arrays.npz"
    summary_path = args.result_dir / "summary.json"
    prompt = np.load(prompt_path)
    static = np.load(static_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    candidate = prompt["candidate_pre_norm"].astype(np.float32)
    native = prompt["native_pre_norm"].astype(np.float32)
    gain = static["final_norm_gain"].astype(np.float32)
    weights = _bf16_memmap(args.model_shard, TIED_HEAD_TENSOR)
    if weights.shape[1] != candidate.size or gain.shape != candidate.shape:
        raise ValueError("tied head, gain, and prompt evidence shapes do not match")

    audit = _direction_screen(weights, gain, candidate, native)
    audit.update(
        {
            "scope": "POST_HOC_DIAGNOSTIC_NOT_A_FROZEN_GATE",
            "experiment": "EXP-083B",
            "prompt_id": PROMPT_ID,
            "deterministic_core_sha256": summary["MEASURED"][
                "deterministic_core_sha256"
            ],
            "candidate_bf16_winner": int(np.argmax(prompt["candidate_logits"])),
            "native_bf16_winner": int(np.argmax(prompt["native_logits"])),
            "model_forward_calls": 0,
        }
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
