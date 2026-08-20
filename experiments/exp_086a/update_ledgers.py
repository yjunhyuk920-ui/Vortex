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


def result_body(result: dict[str, Any]) -> str:
    decision = result["authoritative_decision"]
    decisive = result.get("decisive_aggregate", {})
    if decision == "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION":
        return (
            "## EXP-086A exact state-axis lifting\n\n"
            f"- Decision: `{decision}`\n"
            f"- Error: `{result.get('error', 'unknown')}`\n"
            "- Scientific mechanism verdict: not produced.\n"
        )
    return (
        "## EXP-086A exact state-axis lifting microprogram Gate\n\n"
        f"- Decision: `{decision}`\n"
        f"- Official target calls: `{result['target_forward_calls']}`\n"
        f"- Decisive block size: `{decisive['block_size']}`\n"
        f"- Favorable whole-model operation p50/p95: "
        f"`{decisive['projected_whole_model_operation_fraction_p50']:.9%}` / "
        f"`{decisive['projected_whole_model_operation_fraction_p95']:.9%}`\n"
        f"- Favorable whole-model weight p50/p95: "
        f"`{decisive['projected_whole_model_weight_fraction_p50']:.9%}` / "
        f"`{decisive['projected_whole_model_weight_fraction_p95']:.9%}`\n"
        f"- Joint p50/p95: `{decisive['projected_joint_fraction_p50']:.9%}` / "
        f"`{decisive['projected_joint_fraction_p95']:.9%}`\n"
        f"- Reconstruction mismatches: `{decisive['reconstruction_mismatches']}`\n"
        f"- Deterministic core: `{result['deterministic_core_sha256']}`\n\n"
        "This is a non-deployable perfect-future, exact-real favorable Gate. Native "
        "FP32 reduction order, causal drafting, complete-layer state, CUDA, 405B, "
        "8 GiB, and target latency remain `NOT TESTED`.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    body = result_body(result)
    replace_marker(
        ROOT / "docs/research/EXP_086A_LATEST_RESULT.md", "EXP086A_RESULT", body
    )
    for name in (
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "FAILED_APPROACHES_RECENT.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
    ):
        replace_marker(ROOT / name, "EXP086A_RESULT", body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
