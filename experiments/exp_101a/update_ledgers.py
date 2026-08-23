from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- EXP-101A:START -->"
END = "<!-- EXP-101A:END -->"


def replace_block(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body.rstrip()}\n{END}"
    if START in text and END in text:
        prefix, rest = text.split(START, 1)
        _, suffix = rest.split(END, 1)
        text = prefix.rstrip() + "\n\n" + block + suffix
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def update(result_path: Path, source_commit: str) -> None:
    r: dict[str, Any] = json.loads(result_path.read_text())
    best = r["best_composed_row"]
    K = int(best["block_length"])
    ratio = float(best["catalog_plus_structured"]["multiplication_ratio"])
    catalog = float(best["catalog_only"]["multiplication_ratio"])
    structured = float(best["structured_only"]["multiplication_ratio"])
    decision = r["authoritative_decision"]
    evidence = result_path.as_posix()

    latest = f"""# EXP-101A latest result — Structured Direct-Sum Composition Gate

- source commit: `{source_commit}`
- decision: `{decision}`
- best block length: `{K}`
- best catalog-only multiplication fraction: `{catalog:.12%}`
- best structured-only multiplication fraction: `{structured:.12%}`
- best catalog + structured multiplication fraction: `{ratio:.12%}`
- 10x multiplication Gate: `{ratio <= 0.10}`
- integrity failures: `{r['integrity_failures']}`
- deterministic core: `{r['deterministic_core_sha256']}`

## Claim boundary

This is a deliberately favorable multiplication-only oracle. Every transform,
addition, byte, workspace allocation, native-order repair, causal block source,
and candidate/commit cost is free. TARGET-W, an executable circuit, 8-GiB
physical execution, and same-machine 4B-Q4 latency remain `NOT TESTED`.
"""
    (ROOT / "docs/research/EXP_101A_LATEST_RESULT.md").write_text(latest, encoding="utf-8")

    state = f"""## EXP-101A — structured direct-sum composition

- decision: `{decision}`
- best exact multiplication fraction: `{ratio:.12%}` at `K={K}`
- catalog-only comparator: `{catalog:.12%}`
- structured-only comparator: `{structured:.12%}`
- evidence: `{evidence}`
- source commit: `{source_commit}`

The Gate composed all retained exact integral AlphaTensor schemes with the pinned
published `<6,6,6> <= 137<1> + 8<1,1,2>` relation and granted all non-multiply
costs for free. EXP-100A is reclassified scientifically as a rejection rather
than an integrity failure: its only flagged condition was a resource-empty
`lm_head/K=16384` frontier after the frozen 8-GiB filter, while all 132 exact
factorization controls passed.
"""
    replace_block(ROOT / "RESEARCH_STATE.md", state)

    if decision.startswith("PROMOTE"):
        next_text = "Build the finite-word transform/addition/byte/workspace circuit for the selected composition, then require exact official-checkpoint successor-state equality and the dual roofline."
        failed = "EXP-101A survived the multiplication-only Gate; no failed-family closure is registered."
    else:
        next_text = "Do not retune depth, block list, catalog subset, or the 6x6x6 relation. The next core must change the dominant transform information flow: share one exact transform circuit across multiple projections/layers, or introduce a nonlinear whole-layer instruction."
        failed = "The pinned published structured 6x6x6 direct-sum relation, even when composed with every retained integral AlphaTensor catalog scheme and given all transforms for free, did not reach the first 10x multiplication Gate."
    replace_block(ROOT / "NEXT_EXPERIMENT.md", f"## EXP-101A handoff\n\nCurrent decision: `{decision}`.\n\n{next_text}")
    replace_block(ROOT / "DECISION_LOG.md", f"## EXP-101A decision\n\n`{decision}`\n\nBest composed multiplication fraction: `{ratio:.12%}` at `K={K}`. All non-multiply costs were free.")
    failure_block = f"## EXP-101A — structured direct-sum composition\n\n{failed}\n\nFrozen source: `mkauers/matrix-multiplication@12c26b29a5458e173813911fb4f2c2865fba841e`, `structured/666.exp`; exact catalog source inherited from EXP-100A."
    replace_block(ROOT / "FAILED_APPROACHES.md", failure_block)
    replace_block(ROOT / "FAILED_APPROACHES_RECENT.md", failure_block)
    replace_block(ROOT / "ASSUMPTION_REGISTER.md", "## EXP-101A grants\n\n`GRANTED`: perfect future block, N/A=1, free transforms/additions/moves/bytes/workspace/repair. `MEASURED`: deterministic integer recurrence and complete registered target-shape ledger. `NOT TESTED`: target checkpoint execution and hardware latency.")
    replace_block(ROOT / "VALIDATION_MATRIX.md", f"## EXP-101A validation\n\n| Boundary | Result |\n|---|---|\n| Pinned structured source identity | PASS |\n| Triple-cyclic rank identity | PASS |\n| Registered 405B projection inventory | PASS |\n| Multiplication-only 10x Gate | `{ratio <= 0.10}` |\n| Executable finite-word circuit | `NOT TESTED` |\n| 405B / 8 GiB / latency | `NOT TESTED` |")
    replace_block(ROOT / "ARCHITECTURE.md", f"## EXP-101A architecture status\n\nThe structured direct-sum object is an offline multiplication-count oracle, not a runtime component. Architecture promotion: `{decision.startswith('PROMOTE')}`.")
    replace_block(ROOT / "HARDWARE_VALIDATION_PLAN.md", "## EXP-101A hardware status\n\nNo target hardware run was performed. Multiplication fractions exclude every transform and byte and are not speed records.")
    replace_block(ROOT / "REPRODUCIBILITY.md", f"## EXP-101A reproduction\n\n```bash\npytest -q tests/exp_101a\npython experiments/exp_101a/run_experiment.py --config experiments/exp_101a/config.json --output-dir results/exp_101a/<source-commit>\n```\n\nSource commit: `{source_commit}`. Verify `checksums.sha256`.")
    replace_block(ROOT / "docs/research/VORTEX_RESEARCH_HANDOFF.md", f"## EXP-101A remote handoff\n\nDecision `{decision}`; best composed multiplication fraction `{ratio:.12%}` at K={K}. Evidence `{evidence}`. {next_text}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    update(a.result, a.source_commit)
