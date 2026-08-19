from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = "<!-- EXP088A_CROSS_LAYER_TEMPLATE_CODE -->"
RESULT_PREFIX = "<!-- EXP088A_RESULT:"


def append_once(path: Path, marker: str, block: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current:
        return
    separator = "" if not current or current.endswith("\n") else "\n"
    path.write_text(
        current + separator + "\n" + marker + "\n" + block.rstrip() + "\n",
        encoding="utf-8",
    )


def preregister() -> None:
    append_once(
        ROOT / "NEXT_EXPERIMENT.md",
        PREREG,
        """## Active EXP-088A — Cross-Layer Template and Recurrence Code Gate

EXP-087A found no useful aligned within-layer context. EXP-088A changes the information source to the complete ordered layer stack. It grants each gate/up/down role an oracle choice among exact first/second-order XOR and modulo-2^16 recurrences, one global bit-majority template, and four contiguous depth templates. All 30 official SmolLM2 decoder MLP layers are included with zero forwards. Promotion requires total/p50/p95 exact information fractions <=20%/20%/25%.""",
    )
    append_once(
        ROOT / "ASSUMPTION_REGISTER.md",
        PREREG,
        """## EXP-088A frozen assumptions

- `A-088A-1`: trained layer stacks may be generated from a fixed number of full-shape templates or low-order exact recurrences with low-entropy residuals. **UNVERIFIED**.
- `A-088A-2`: role-wise oracle generator choice and ideal per-stream entropy coding make the Gate favorable. **FROZEN GRANTS**.
- `A-088A-3`: piecewise repeated-layer positive controls demonstrate that the language can cross 20%. **CONTROL REQUIREMENT**.
- `A-088A-4`: rejection closes the frozen recurrence/template language; a new generator must be sublinear in both layer count and matrix area, or change the causal dependency. **FIXED SCOPE**.""",
    )
    append_once(
        ROOT / "VALIDATION_MATRIX.md",
        PREREG,
        """## EXP-088A validation entry

| Gate | Required evidence | Before run |
|---|---|---|
| Official checkpoint | pinned SmolLM2 official loader | NOT RUN |
| Population | all decoder-layer gate/up/down matrices | FROZEN |
| Exact reconstruction | every recurrence/template plus residual | NOT RUN |
| Positive control | four piecewise templates total fraction <20% | UNIT TEST |
| Information | total/p50/p95 <=20%/20%/25% | NOT RUN |
| Runtime/state/hardware | later Gate only | NOT TESTED |""",
    )


def record(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source = str(result.get("source_sha", "UNKNOWN"))
    marker = f"{RESULT_PREFIX}{source} -->"
    aggregate = result.get("DERIVED", {}).get("aggregate", {})
    decision = str(result.get("authoritative_decision", "UNKNOWN"))
    block = f"""## EXP-088A result — `{source}`

```text
decision                          {decision}
layer count                       {aggregate.get('layer_count','NOT_AVAILABLE')}
role-wise best candidates         {aggregate.get('role_wise_best_candidate_names','NOT_AVAILABLE')}
total information fraction        {aggregate.get('total_information_fraction','NOT_AVAILABLE')}
effective layer fraction p50      {aggregate.get('effective_layer_fraction_p50','NOT_AVAILABLE')}
effective layer fraction p95      {aggregate.get('effective_layer_fraction_p95','NOT_AVAILABLE')}
all exact reconstructions         {aggregate.get('all_reconstructions_exact','NOT_AVAILABLE')}
```

This is a favorable no-forward checkpoint-description Gate. Dense operation replacement, complete transition/state, 405B, 8 GiB, and physical latency remain `NOT TESTED`."""
    for name in (
        "RESEARCH_STATE.md",
        "NEXT_EXPERIMENT.md",
        "DECISION_LOG.md",
        "FAILED_APPROACHES_RECENT.md",
        "ASSUMPTION_REGISTER.md",
        "VALIDATION_MATRIX.md",
    ):
        append_once(ROOT / name, marker, block)
    latest = ROOT / "docs/research/EXP_088A_LATEST_RESULT.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(marker + "\n" + block + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preregister",))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        preregister()
    elif args.result is not None:
        record(args.result)
    else:
        parser.error("use --mode preregister or --result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
