from __future__ import annotations

import argparse
from pathlib import Path

from experiments.exp_092a import update_ledgers as base


def update(result_path: Path, source_commit: str) -> None:
    base.update(result_path, source_commit)
    replacements = {
        "The exact AR target and teacher-forced block are oracle controls only.": "The exact incremental AR block is an oracle control only.",
        "exact teacher-forced block": "exact incremental-reference block",
        "teacher-forced block": "incremental-reference block",
        "teacher-forced oracle": "incremental-reference oracle",
        "DEV-W guessed/true operator-input blocks, delayed AR equality": "DEV-W guessed complete-block inputs, true incremental-reference operator-input blocks",
        "| Teacher-forced block | 128-token delayed AR equality |": "| Incremental reference block | 128 official one-token captures |",
    }
    paths = [
        "docs/research/EXP_092A_LATEST_RESULT.md",
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "FAILED_APPROACHES.md",
        "FAILED_APPROACHES_RECENT.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
        "ARCHITECTURE.md",
        "HARDWARE_VALIDATION_PLAN.md",
        "REPRODUCIBILITY.md",
    ]
    for relative in paths:
        path = base.ROOT / relative
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    update(args.result, args.source_commit)
