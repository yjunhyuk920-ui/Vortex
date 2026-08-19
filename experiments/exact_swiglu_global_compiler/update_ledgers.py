from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MARKER_PREFIX = "<!-- EXACT_SWIGLU_GLOBAL_COMPILER_RESULT:"


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text() if path.exists() else ""
    if marker in text:
        return
    separator = "" if not text or text.endswith("\n") else "\n"
    path.write_text(text + separator + "\n" + marker + "\n" + block.rstrip() + "\n")


def result_block(result: dict[str, Any]) -> str:
    aggregate = result["DERIVED"]["aggregate"]
    abi = result["MEASURED"]["joint_gate_up_abi_probe"]
    return f"""## EXP-085A exact whole-SwiGLU compiler result

- source SHA: `{result['github'].get('sha')}`
- decision: `{result['authoritative_decision']}`
- public checkpoint: `{result['pins']['DEV-W']['model_id']}@{result['pins']['DEV-W']['revision']}`
- layers audited: `{result['MEASURED']['layers_audited']}`
- strict operation fraction: `{aggregate['strict_operation_fraction']:.12%}`
- optimistic algebraic operation fraction: `{aggregate['optimistic_algebraic_operation_fraction']:.12%}`
- final p50 target fraction: `{aggregate['target_fraction']:.12%}`
- optimistic miss factor: `{result['DERIVED']['distance_to_target']:.6f}x`
- exact duplicate gate rows: `{aggregate['duplicate_gate_rows']}`
- exact duplicate up rows: `{aggregate['duplicate_up_rows']}`
- exact duplicate nonlinear atoms: `{aggregate['duplicate_gate_up_atoms']}`
- antipodal gate pairs: `{aggregate['antipodal_gate_pairs']}`
- joint gate/up ABI probe: `{abi['exact']}` (`{abi['mismatch_rows']}` mismatching rows)
- deterministic core: `{result['deterministic_core_sha256']}`

The compiler treats `Wd[silu(Wg x) * (Wu x)]` as one functional DAG and grants
identical nonlinear atoms a free exact-real down-column merge in the decisive
bound. A structural failure rejects exact equality/sign/CSE as the missing
global nonlinear core before a full executor. It does not reject a materially
different nonlinear representation that derives new query-time information
without relying on exact atom equivalence.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    source = str(result.get("github", {}).get("sha") or "UNKNOWN")
    marker = f"{MARKER_PREFIX}{source} -->"
    block = result_block(result)
    for relative in (
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
        "FAILED_APPROACHES_RECENT.md",
        "docs/research/EXPERIMENT_085A_EXACT_SWIGLU_GLOBAL_COMPILER_GATE.md",
    ):
        append_once(ROOT / relative, marker, block)


if __name__ == "__main__":
    main()
