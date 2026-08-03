#!/usr/bin/env python3
"""Idempotently finalize EXP-052 and register EXP-053."""

from __future__ import annotations

from pathlib import Path

MARKER = "<!-- EXP-052-AUTHORITATIVE-FINAL -->"
AUTH = (
    "`results/exp_052/summary.json`; workflow `30811429049`; source head "
    "`d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; "
    "ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`."
)
DECISION = "REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY"
RESULT = (
    "1,152 exact warm states and 36 leave-one-family-out rows produced zero wrong hits "
    "and zero build/evaluation leakage, but P0 prefix and S0 KV-state held-out hit rates "
    "were 0% in every family. Fallback was 100%, natural exact reuse median/max was 1/1, "
    "and p90 fully-accounted target fraction was 6.0 (600%). Same-state replay was 100% "
    "exact and required at least 85 repetitions. Under 8 GiB hot index plus 1 TiB cold "
    "advice, combined coverage of 2^48 independent states was 6.357828752356909e-7, "
    "leaving fallback 0.9999993642171248."
)


def append(path: str, heading: str, text: str) -> None:
    file = Path(path)
    body = file.read_text(encoding="utf-8")
    if MARKER in body:
        return
    file.write_text(
        body.rstrip() + f"\n\n{MARKER}\n## {heading}\n\n{text.strip()}\n",
        encoding="utf-8",
    )


append(
    "RESEARCH_STATE.md",
    "EXP-052 authoritative result and EXP-053 frontier",
    f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`. Exact tables are auxiliary only. "
    "The active frontier is `EXP-053 — Automatic Bit-Exact Decision-Circuit Compiler Gate`, "
    "which compiles bounded quantized operators from weights rather than enumerating states. "
    "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.",
)
append(
    "DECISION_LOG.md",
    "D-022/D-023/D-024 — Close EXP-051/052 and select EXP-053",
    "D-022 rejects EXP-051 layer-tail skipping: median suffix-stable depth 25%, p90 37.5%, "
    "median/p90 favorable traffic 82.2069%/99.8011%, with a final-layer adversary.\n\n"
    f"D-023 records EXP-052 authority {AUTH} {RESULT} Decision: `{DECISION}`.\n\n"
    "D-024 requires the next mechanism to be non-enumerative and weight-derived; EXP-053 is active.",
)
append(
    "FAILED_APPROACHES.md",
    "F-019/F-020 — Tail exit and enumerative exact advice",
    "F-019 rejects fixed/oracle layer-finalization tail skipping as core. F-020 rejects "
    "enumerative exact prefix/KV advice as core. Forbidden rescues include larger copies of "
    "the same table, hash-width changes presented as compression, replay presented as held-out "
    "generalization, and uncharged build/fallback amortization.",
)
append(
    "ASSUMPTION_REGISTER.md",
    "A-026/A-027/A-028 — Advice coverage, reuse, circuit compilation",
    "A-026 enumerative exact advice generalizes across unseen families: CONTRADICTED (0% held-out hits). "
    "A-027 natural exact states repeat at least 85 times: CONTRADICTED on the corpus (median/max 1/1). "
    "A-028 a non-enumerative bit-exact weight-derived circuit remains compact: ACTIVE UNVERIFIED for EXP-053.",
)
append(
    "VALIDATION_MATRIX.md",
    "EXP-051/052 addendum",
    "| Claim | Evidence | Verdict |\n|---|---:|---|\n"
    "| EXP-051 p90 favorable traffic <=25% | E1: 99.8011% | REJECTED |\n"
    "| EXP-052 exact table integrity | E1: wrong hits 0 | AUXILIARY |\n"
    "| EXP-052 held-out hit >=98.8148% | E1: 0% all P0/S0 families | REJECTED |\n"
    "| EXP-052 natural reuse >=85 | E1: median/max 1/1 | REJECTED |\n"
    "| EXP-052 p90 fraction <=1.185185% | E1: 600% | REJECTED |\n"
    "| EXP-052 budget fallback <=1.185185% | E1: 99.9999364% | REJECTED |\n"
    "| EXP-053 exact circuit compiler | E0 | NEXT GATE |",
)
append(
    "ARCHITECTURE.md",
    "Closed EXP-052 and active EXP-053 architecture",
    "EXP-052 exact witnessed tables are `exact hit OR exact target fallback` auxiliary memoization. "
    "EXP-053 compiles bounded quantized target weights and exact arithmetic semantics into a "
    "structurally hashed bit-vector/AIG decision circuit. Compile time, nodes, bytes, query touches, "
    "reduction, and fallback are mandatory costs.",
)
append(
    "HARDWARE_VALIDATION_PLAN.md",
    "EXP-052/053 hardware boundary",
    "EXP-052 has no Phase-D core route. EXP-053 hardware work is forbidden until a real small-checkpoint "
    "operation is replaced exactly and circuit bytes/query work close the 8 GiB and 1.185185% equations. "
    "Phase D remains NOT TESTED.",
)
append(
    "REPRODUCIBILITY.md",
    "EXP-052 frozen authority",
    f"Authority: {AUTH}\n\nVerify with `cd results/exp_052 && sha256sum -c checksums.sha256`. "
    "Original workflow hashes are preserved in `results/exp_052/artifacts/workflow_checksums.sha256`; "
    "the original workflow summary is `results/exp_052/raw/workflow_summary.json`.",
)
append(
    "docs/research/EXPERIMENT_052_EXACT_ADVICE_TRADEOFF.md",
    "Final authoritative result",
    f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`.",
)
for optional in ("docs/RESEARCH_PROGRESS_LEDGER.md", "docs/SESSION_HANDOFF.md"):
    if Path(optional).exists():
        append(
            optional,
            "EXP-052 handoff",
            "Enumerative exact advice is rejected. Read `results/exp_052/summary.json` and "
            "`NEXT_EXPERIMENT.md`; continue with EXP-053 or a materially new mechanism only.",
        )

Path("NEXT_EXPERIMENT.md").write_text(
    f"""# Next Experiment

## Closed Gate — EXP-052

Authority: {AUTH}

{RESULT}

Decision:

```text
{DECISION}
```

## EXP-053 — Automatic Bit-Exact Decision-Circuit Compiler Gate

### Mechanism change

Compile a bounded quantized target operator directly from immutable weights and exact arithmetic semantics into a reduced Boolean/arithmetic decision circuit. Runtime states may not be stored as the representation.

### Conditions

```text
Q0 independent bit-exact arithmetic reference
Q1 weight-derived bit-vector/AIG compiler
Q2 structural hashing and exact reduction
Q3 structured sparse/low-rank controls
Q4 adversarial random dense and late-bit operators
Q5 exact circuit query and exact fallback
```

### Initial domains

```text
input bits 8, 12, 16, 20
output classes 2, 4, 8
accumulator widths 8, 12, 16
structured and dense-random operator families
```

### Contract

- no training, future generated token, or state truth table as the compiled representation;
- compiler input is weights/config/arithmetic semantics only;
- exhaustive small-domain enumeration is validation only;
- bit-exact equality is mandatory;
- compile time, nodes, bytes, reduction, query touches/bytes, and fallback are charged;
- structured success cannot erase adversarial/random scaling failure.

### Early rejection Gate

```text
exact mismatch >0
hidden truth-table representation
p50 query node/byte fraction >10%
p90 query node/byte fraction >25%
1 TiB projection exceeded before target scale
adversarial node growth doubling exponent >1.5 per added input bit
compile cost not amortizable under measured reuse
random dense cases require near-full original arithmetic
```

Promotion still requires real small-checkpoint operation replacement and p90 fully-accounted fraction <=1.185185%.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement bit-vector arithmetic and an exact AIG evaluator;
2. compile dense linear top-1 decisions from weights without state enumeration;
3. add structural hashing and reduction;
4. use exhaustive small-domain evaluation only for equivalence validation;
5. measure structured versus adversarial node/query scaling;
6. freeze circuits, checksums, scaling fits, and decision.
""",
    encoding="utf-8",
)

workflow = Path(".github/workflows/exp_052_gate.yml")
lines = workflow.read_text(encoding="utf-8").splitlines()
start = lines.index("on:")
end = lines.index("concurrency:")
workflow.write_text(
    "\n".join(lines[:start] + ["on:", "  workflow_dispatch:", ""] + lines[end:]) + "\n",
    encoding="utf-8",
)
