from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREREG_MARKER = "<!-- EXP086A_PREREGISTERED -->"
RESULT_MARKER_PREFIX = "<!-- EXP086A_RESULT:"


def append_once(path: Path, marker: str, block: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return
    separator = "" if not current or current.endswith("\n") else "\n"
    path.write_text(current + separator + "\n" + marker + "\n" + block.rstrip() + "\n", encoding="utf-8")


def preregister() -> None:
    block = """## Active EXP-086A — Causal Functional Microprogram Sharing Gate

The frozen Gate compiles the complete layer-0 SwiGLU function of the official SmolLM2-135M checkpoint into four non-state-keyed low-rank affine microprograms. Ninety-six causal build states and ninety-six disjoint causal evaluation states are fixed. An impossible favorable oracle and a nearest-centroid causal router are measured separately. Promotion requires zero BF16 vector mismatch, repeated evaluation-state reuse per program, a projected FP32 sidecar no larger than 4 GiB, and compiled-MLP work no greater than the registered p50 fraction. The Gate does not authorize attention, full-layer, 405B, or hardware claims."""
    append_once(ROOT / "NEXT_EXPERIMENT.md", PREREG_MARKER, block)
    append_once(ROOT / "ASSUMPTION_REGISTER.md", PREREG_MARKER, """## EXP-086A frozen assumptions

- `A-086A-1`: a small library of whole-SwiGLU functional maps is exactly reusable across disjoint causal states. **UNVERIFIED**.
- `A-086A-2`: an affine low-rank program is a materially different cross-page function compiler, not a state lookup or fixed-weight low-rank decomposition. **FIXED SCOPE**.
- `A-086A-3`: FP32 program coefficients can preserve official BF16 MLP outputs exactly. **TO BE TESTED**.
- `A-086A-4`: failure of the impossible oracle closes only this affine program language and frozen resource plan. **FIXED SCOPE**.""")
    append_once(ROOT / "VALIDATION_MATRIX.md", PREREG_MARKER, """## EXP-086A validation entry

| Gate | Required evidence | Status before run |
|---|---|---|
| Official checkpoint | pinned SmolLM2 official loader | NOT RUN |
| Build/evaluation separation | six disjoint prompt pairs, 96/96 states | FROZEN |
| Numerical integrity | FP64 build interpolation bit-exact | NOT RUN |
| Oracle reuse | FP32 oracle vector exact 100%, >=4 states/used program | NOT RUN |
| Causal router | nearest-centroid FP32 vector exact 100% | NOT RUN |
| Resource | target sidecar <=4 GiB, compiled MLP fraction <=1.185185% | DERIVED BY RUN |
| 405B/8 GiB hardware | actual target execution | NOT TESTED |""")


def record_result(result_path: Path) -> None:
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    source = str(result.get("source_sha", "UNKNOWN"))
    marker = f"{RESULT_MARKER_PREFIX}{source} -->"
    decision = str(result.get("authoritative_decision", "UNKNOWN"))
    measured = result.get("MEASURED", {})
    queries = measured.get("queries", {})
    oracle = queries.get("evaluation_oracle_fp32", {}).get("exact_report", {})
    router = queries.get("evaluation_router_fp32", {}).get("exact_report", {})
    resources = result.get("DERIVED", {}).get("target_resources", {})
    block = f"""## EXP-086A result — `{source}`

- Decision: `{decision}`.
- Evaluation oracle FP32 vector exact: `{oracle.get('vector_exact_fraction', 'NOT_AVAILABLE')}`.
- Evaluation router FP32 vector exact: `{router.get('vector_exact_fraction', 'NOT_AVAILABLE')}`.
- Oracle row exact p50: `{oracle.get('row_exact_fraction_p50', 'NOT_AVAILABLE')}`.
- Router row exact p50: `{router.get('row_exact_fraction_p50', 'NOT_AVAILABLE')}`.
- Projected target sidecar: `{resources.get('sidecar_gib', 'NOT_AVAILABLE')} GiB`.
- Compiled MLP operation fraction: `{resources.get('compiled_mlp_operation_fraction', 'NOT_AVAILABLE')}`.
- Whole-model fraction with only MLP replaced: `{resources.get('whole_model_operation_fraction_if_only_mlp_replaced', 'NOT_AVAILABLE')}`.
- Complete Transformer layer, 405B, 8-GiB GPU, and physical p50/p95 remain `NOT TESTED`."""
    for name in (
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "FAILED_APPROACHES_RECENT.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
    ):
        append_once(ROOT / name, marker, block)
    latest = ROOT / "docs/research/EXP_086A_LATEST_RESULT.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(marker + "\n" + block + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preregister",), default=None)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister()
    elif args.result is not None:
        record_result(args.result)
    else:
        parser.error("use --mode preregister or --result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
