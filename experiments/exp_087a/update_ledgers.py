from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def replace_marker(path: Path, marker: str, body: str) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = f"{start}\n{body.rstrip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def body(result: dict[str, Any]) -> str:
    decision = result["authoritative_decision"]
    if decision == "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION":
        return (
            "## EXP-087A lossless entropy-stationary block Gate\n\n"
            f"- Decision: `{decision}`\n"
            f"- Error: `{result.get('error')}`\n"
        )
    global_row = result["global_compression"]
    k128 = result["aggregate_by_k"][
        str(max(map(int, result["aggregate_by_k"].keys())))
    ]
    return (
        "## EXP-087A lossless entropy-stationary perfect-block Gate\n\n"
        f"- Decision: `{decision}`\n"
        f"- Exact weight roundtrip mismatches: `{result['roundtrip_mismatches']}`\n"
        f"- Native block/sequential mismatches: `{result['block_mismatches']}`\n"
        f"- DEV-W zlib ratio: `{global_row['zlib_ratio']:.6f}x`\n"
        f"- DEV-W empirical Shannon ratio: `{global_row['shannon_ratio']:.6f}x`\n"
        f"- K=128 zlib traffic fraction: `{k128['zlib_weight_fraction']:.9%}`\n"
        f"- K=128 full-MLP exact-vector p50: `{k128['full_mlp_vector_exact_p50']:.9%}`\n"
        f"- Deterministic core: `{result['deterministic_core_sha256']}`\n\n"
        "This Gate preserves exact weights on a small public checkpoint and uses "
        "perfect future activations. Target 405B entropy, target native 4B baseline, "
        "target effective compute, CUDA, 8 GiB, and latency remain `NOT TESTED`.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    text = body(result)
    replace_marker(
        ROOT / "docs/research/EXP_087A_LATEST_RESULT.md", "EXP087A_RESULT", text
    )
    for name in (
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "FAILED_APPROACHES_RECENT.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
    ):
        replace_marker(ROOT / name, "EXP087A_RESULT", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
