#!/usr/bin/env python3
"""Idempotently finalize EXP-053 and preregister EXP-054."""

from __future__ import annotations

from pathlib import Path

MARKER = "<!-- EXP-053-AUTHORITATIVE-FINAL -->"
AUTH = (
    "`results/exp_053/summary.json`; workflow `30814648709`; source head "
    "`325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge "
    "`4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; "
    "ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`."
)
DECISION = "REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY"
RESULT = (
    "24 weight-derived circuits were exhaustively checked over 4,506,624 inputs with "
    "zero output-bit mismatch and no truth-table representation. Structural hashing left "
    "p50/p90 reachable fractions 0.84168345/0.94107229; dense-random p50 was 0.92452096. "
    "The maximum 405B source-parameter circuit projection was 255.5966 TiB. Late-bit "
    "controls simplified to zero AND nodes, but sparse controls still retained 65–78% of "
    "the exact bit-blast and projected 3.17–7.45 TiB. Growth and compile-amortization Gates "
    "passed; node, byte, storage, and random-dense Gates failed."
)


def append(path: str, heading: str, body: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        return
    target.write_text(
        text.rstrip() + f"\n\n{MARKER}\n## {heading}\n\n{body.strip()}\n",
        encoding="utf-8",
    )


append(
    "RESEARCH_STATE.md",
    "EXP-053 authoritative result and EXP-054 frontier",
    f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`. The exact AIG compiler, evaluator, binary format, and exhaustive validator remain E1 auxiliary reference machinery. The active frontier is `EXP-054 — Exact Reduced Ordered Decision-Diagram Gate`, which replaces all-gate AIG evaluation with one exact input-adaptive decision path. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.",
)
append(
    "DECISION_LOG.md",
    "D-025/D-026 — Reject AIG structural hashing and select EXP-054",
    f"D-025 records EXP-053 authority {AUTH} {RESULT} Decision: `{DECISION}`.\n\nD-026 selects exact reduced ordered decision diagrams as the next representation class. They must be compiled from weights/residual arithmetic states, not a stored truth table, and must charge compile-state visits, unique nodes, bytes, path probes, variable-order search, and fallback.",
)
append(
    "FAILED_APPROACHES.md",
    "F-021 — Structurally hashed bit-exact AIG as core",
    "Bit-exact AIG compilation preserved all registered finite-domain decisions, but p50/p90 query work remained 84.17%/94.11% of the same unreduced exact bit-blast, dense-random p50 was 92.45%, and projected storage reached 255.60 TiB. Forbidden rescues are reporting only late-bit controls, relabeling raw bit blasting as compression, hiding circuit bytes, or extrapolating E1 synthetic exactness into a real Transformer claim.",
)
append(
    "ASSUMPTION_REGISTER.md",
    "A-029/A-030 — AIG reduction and input-adaptive exact diagrams",
    "A-029 structural hashing alone makes exact dense arithmetic cheap: CONTRADICTED for the registered bounded operators. A-030 a weight-derived reduced ordered decision diagram can trade compile/storage for a short exact input-adaptive query path without exponential growth: ACTIVE UNVERIFIED for EXP-054.",
)
append(
    "VALIDATION_MATRIX.md",
    "EXP-053 addendum",
    "| Claim | Evidence | Verdict |\n|---|---:|---|\n| EXP-053 exact circuit equality | E1: 0 mismatch / 4,506,624 inputs | PASS reference |\n| EXP-053 no truth-table representation | E1: 0 cases | PASS |\n| EXP-053 p50 query fraction <=10% | E1: 84.1683% | REJECTED |\n| EXP-053 p90 query fraction <=25% | E1: 94.1072% | REJECTED |\n| EXP-053 dense-random p50 <=25% | E1: 92.4521% | REJECTED |\n| EXP-053 projected storage <=1 TiB | PROJECTED: max 255.5966 TiB | REJECTED |\n| EXP-054 reduced decision diagram | E0 | NEXT GATE |",
)
append(
    "ARCHITECTURE.md",
    "Closed EXP-053 and active EXP-054 architecture",
    "EXP-053 exact AIGs are auxiliary bit-level reference machinery. EXP-054 compiles immutable weights into a reduced ordered multi-terminal decision diagram using Shannon branching, exact residual arithmetic states, unique-table reduction, and a fixed weight-derived variable order. Runtime evaluates one root-to-terminal path; compile-state visits, nodes, bytes, query probes, order-search cost, and fallback are mandatory.",
)
append(
    "HARDWARE_VALIDATION_PLAN.md",
    "EXP-053/054 hardware boundary",
    "EXP-053 has no Phase-D promotion route as core. EXP-054 hardware work is forbidden until a real small-checkpoint operation is replaced exactly and both decision-diagram storage and path probes close the 8 GiB and 1.185185% equations. Phase D remains NOT TESTED.",
)
append(
    "REPRODUCIBILITY.md",
    "EXP-053 frozen authority",
    f"Authority: {AUTH}\n\nVerify with `cd results/exp_053 && sha256sum -c checksums.sha256`. Original workflow hashes are preserved in `results/exp_053/artifacts/workflow_checksums.sha256`; original summary is `results/exp_053/raw/workflow_summary.json`; all 24 binary AIGs are under `results/exp_053/raw/circuits/`.",
)
append(
    "docs/research/EXPERIMENT_053_BIT_EXACT_DECISION_CIRCUIT.md",
    "Final authoritative result",
    f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`.",
)
for optional in ("docs/RESEARCH_PROGRESS_LEDGER.md", "docs/SESSION_HANDOFF.md"):
    if Path(optional).exists():
        append(
            optional,
            "EXP-053 handoff",
            "Bit-exact AIG structural hashing is rejected as core. Read `results/exp_053/summary.json` and continue with EXP-054 reduced decision diagrams or a materially new mechanism only.",
        )

Path("NEXT_EXPERIMENT.md").write_text(
    f"""# Next Experiment

## Closed Gate — EXP-053

Authority: {AUTH}

{RESULT}

Decision:

```text
{DECISION}
```

## EXP-054 — Exact Reduced Ordered Decision-Diagram Gate

### Mechanism change

Compile the same bounded signed modular top-1 operators into a reduced ordered multi-terminal decision diagram (ROMTDD/ROBDD-like representation). Unlike EXP-053 AIGs, runtime evaluates one variable-dependent root-to-terminal path rather than every reachable gate.

The compiler may use exact residual score states and Shannon decomposition, but it may not store an explicit input-to-output truth table.

### Conditions

```text
D0 independent arithmetic reference
D1 natural input-bit variable order
D2 deterministic weight-magnitude variable order
D3 exact unique-table reduction: low==high elimination and (var,low,high) sharing
D4 sparse/low-rank controls
D5 dense-random and late-bit adversaries
D6 exact path evaluator and exhaustive finite-domain equivalence
```

For each operator, compile both registered variable orders. A fixed weight-derived selector may retain the smaller diagram, but compile visits/time for both orders are charged.

### Registered domains

Use the EXP-053 operator families and scaling matrix, with an early safety ceiling:

```text
input bits 8, 12, 16, 20
classes 2, 4, 8
accumulator widths 8, 12, 16
maximum compile states/nodes per order 2,000,000
```

A ceiling hit is a scientific failure row with exact fallback, not an infrastructure crash.

### Accounting

```text
recursive compile-state visits
memoized residual states
unique decision nodes
terminal count
serialized bytes
both-order compile time/bytes
p50/p90 root-to-terminal input probes
query probe fraction = path probes / input bits
fallback on ceiling or corruption
405B source-parameter storage projection
node growth per added input bit
```

### Early rejection Gate

```text
exact mismatch >0
explicit truth table stored as representation
p50 query probe fraction >10%
p90 query probe fraction >25%
any dense-random 20-bit case exceeds 2,000,000 compile states/nodes
projected diagram storage >1 TiB
adversarial node-growth multiplier >1.5 per added input bit
variable-order search cost cannot be amortized within 1,000,000 queries
fallback/ceiling rate >0
```

Failure decision:

```text
REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY
```

### Promotion boundary

Synthetic success still requires real small-checkpoint operation replacement, exact output agreement, p90 fully-accounted target fraction <=1.185185%, non-degrading scale, 8 GiB hot-state closure, and Phase-D measurement.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement exact reduced multi-terminal decision diagrams from residual arithmetic states;
2. add natural and weight-magnitude variable orders;
3. enforce compile-state/node ceilings with exact fallback;
4. exhaustively validate all completed finite domains;
5. measure path probes, storage, growth, and order-selection cost;
6. freeze diagrams, checksums, raw rows, and decision.
""",
    encoding="utf-8",
)

workflow = Path(".github/workflows/exp_053_gate.yml")
lines = workflow.read_text(encoding="utf-8").splitlines()
start = lines.index("on:")
end = lines.index("concurrency:")
workflow.write_text(
    "\n".join(lines[:start] + ["on:", "  workflow_dispatch:", ""] + lines[end:]) + "\n",
    encoding="utf-8",
)
