from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def append_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + text.rstrip() + "\n", encoding="utf-8")


def preregister() -> None:
    append_once(ROOT / "NEXT_EXPERIMENT.md", "EXP086A_GLOBAL_BF16_SUCCESSIVE_REFINEMENT", """
<!-- EXP086A_GLOBAL_BF16_SUCCESSIVE_REFINEMENT -->
## Active EXP-086A — Global BF16 Successive-Refinement Oracle Gate

EXP-085A rejected page-separable exact SwiGLU refinement after both the
reference-aided oracle and sound selector consumed every fused macro-page.
EXP-086A changes the information unit from pages to a global BF16
sign/exponent/MSB-mantissa stream.

The official SmolLM2-135M layer-0 Gate grants one global gate/up precision and
a reference-aided independently refinable down precision for every output row.
It accepts only a complete bitwise-equal BF16 MLP output. The frozen promotion
threshold is zero-order prefix-entropy fraction p50/p95 <=20%/25% and required
perfect block acceptance p50/p95 <=16/32. This is a favorable cold-information
Gate, not an operation or latency claim.

Authority: `docs/research/EXPERIMENT_086A_GLOBAL_BF16_SUCCESSIVE_REFINEMENT_GATE.md`.
""")
    append_once(ROOT / "ASSUMPTION_REGISTER.md", "A-086A-1", """
<!-- EXP086A_GLOBAL_BF16_SUCCESSIVE_REFINEMENT -->
## EXP-086A frozen assumptions

- `A-086A-1`: low BF16 mantissa streams can often remain unread while the
  complete unchanged SwiGLU output stays bitwise exact. **UNVERIFIED**.
- `A-086A-2`: one global gate/up precision plus row-adaptive down precision is a
  favorable ceiling for the fixed MSB-prefix source. **FROZEN**.
- `A-086A-3`: zero-order prefix entropy grants ideal coding with coder, framing,
  random access, decode, metadata and kernel costs free. **NOT A RUNTIME**.
- `A-086A-4`: rejection closes only the fixed significance-prefix source; a
  cross-weight context model or nonlinear exact generator is different.
""")


def latest_markdown(result: dict[str, Any]) -> str:
    measured = result.get("MEASURED", {})
    return f"""# EXP-086A latest result

```text
decision: {result.get('authoritative_decision', 'INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION')}
source: {result.get('source_sha', 'UNKNOWN')}
evidence: {result.get('evidence_level', 'NOT_AVAILABLE')}
deterministic core: {result.get('deterministic_core_sha256', 'NOT_AVAILABLE')}
```

```text
activations: {measured.get('activation_count', 'NOT_COMPLETED')}
full precision control: {measured.get('full_precision_control_passed', 'NOT_COMPLETED')}
global uniform entropy p50/p95: {measured.get('global_uniform_entropy_fraction_p50', 'NOT_COMPLETED')} / {measured.get('global_uniform_entropy_fraction_p95', 'NOT_COMPLETED')}
row-adaptive entropy p50/p95: {measured.get('row_adaptive_entropy_fraction_p50', 'NOT_COMPLETED')} / {measured.get('row_adaptive_entropy_fraction_p95', 'NOT_COMPLETED')}
minimum perfect acceptance p50/p95: {measured.get('row_adaptive_minimum_perfect_acceptance_p50', 'NOT_COMPLETED')} / {measured.get('row_adaptive_minimum_perfect_acceptance_p95', 'NOT_COMPLETED')}
```

This is a favorable E1 cold-information observation. Standard-ISA operation
reduction, complete layer replacement, successor-state equality, 405B, 8 GiB,
and target latency remain NOT TESTED.
"""


def record_result(result_path: Path) -> None:
    preregister()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    decision = str(result["authoritative_decision"])
    measured = result.get("MEASURED", {})
    core = result.get("deterministic_core_sha256", "NOT_AVAILABLE")
    (ROOT / "docs/research/EXP_086A_LATEST_RESULT.md").write_text(latest_markdown(result), encoding="utf-8")
    common = f"""
<!-- EXP086A_RESULT -->
## EXP-086A — Global BF16 successive refinement

```text
decision                                    {decision}
official checkpoint                         HuggingFaceTB/SmolLM2-135M
actual causal activations                   {measured.get('activation_count', 'NOT_COMPLETED')}
full-precision control                      {measured.get('full_precision_control_passed', 'NOT_COMPLETED')}
row-adaptive entropy fraction p50/p95       {measured.get('row_adaptive_entropy_fraction_p50', 'NOT_COMPLETED')} / {measured.get('row_adaptive_entropy_fraction_p95', 'NOT_COMPLETED')}
minimum perfect acceptance p50/p95          {measured.get('row_adaptive_minimum_perfect_acceptance_p50', 'NOT_COMPLETED')} / {measured.get('row_adaptive_minimum_perfect_acceptance_p95', 'NOT_COMPLETED')}
deterministic core SHA-256                   {core}
```

The Gate grants ideal zero-order coding, reference-aided per-output-row down
precision, free non-MLP traffic, and free selector/decompression/kernel costs.
It is not a physical speed result.
"""
    for filename in ("RESEARCH_STATE.md", "DECISION_LOG.md", "VALIDATION_MATRIX.md"):
        append_once(ROOT / filename, "EXP086A_RESULT", common)
    if decision.startswith("REJECT_"):
        append_once(ROOT / "FAILED_APPROACHES_RECENT.md", "EXP086A_RESULT", common + """
Permanent scope restriction: do not tune or rename the global
sign/exponent/MSB-mantissa prefix. Reopening requires a new cross-weight
information source or causal execution dependency.
""")
    next_text = """
<!-- EXP086A_POST_RESULT -->
## Post EXP-086A next Gate

The fixed MSB-prefix mechanism passed its favorable information Gate. The next
authorized step is a complete real-MLP packed/fused kernel Gate charging prefix
decoding, residual addressing, instructions and physical bytes.
""" if decision.startswith("PROMOTE_") else """
<!-- EXP086A_POST_RESULT -->
## Post EXP-086A next Gate

The fixed global BF16 MSB-prefix information source is closed under its frozen
favorable oracle. The next candidate must change the checkpoint information
source: a cross-weight context model or nonlinear exact generator with fully
charged residual code must first pass a no-forward construction Gate.
"""
    append_once(ROOT / "NEXT_EXPERIMENT.md", "EXP086A_POST_RESULT", next_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preregister"])
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister()
    elif args.result is not None:
        record_result(args.result)
    else:
        parser.error("provide --mode preregister or --result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
