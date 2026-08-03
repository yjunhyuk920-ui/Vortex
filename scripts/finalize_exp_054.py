#!/usr/bin/env python3
"""Idempotently finalize EXP-054 and preregister EXP-055."""
from __future__ import annotations
from pathlib import Path

MARKER = "<!-- EXP-054-AUTHORITATIVE-FINAL -->"
AUTH = "`results/exp_054/summary.json`; workflow `30816333096`; source head `2c63da85050afcedad6a00698a6f8fddd3bc99d2`; artifact `8856906303`; ZIP SHA-256 `0dc642f306cea99ce01095758a5f49151092d530efb94d36985553e408596edf`."
DECISION = "REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY"
RESULT = "24 operators were compiled in natural and weight-magnitude orders: 48 completed diagrams, zero ceiling/fallback, zero mismatches across 9,013,248 validations, and zero truth-table representations. Selected global p50/p90 path fractions were 35%/95%. Dense-random growth was 1.6872587x per added input bit, maximum projected storage was 202.2479 TiB, and maximum order-search amortization was 1,185,055 queries. Late-bit controls reached 5–12.5% paths, but dense, low-rank, and sparse families failed the universal Gate."

def append(path: str, heading: str, body: str) -> None:
    target=Path(path); text=target.read_text(encoding="utf-8")
    if MARKER in text: return
    target.write_text(text.rstrip()+f"\n\n{MARKER}\n## {heading}\n\n{body.strip()}\n",encoding="utf-8")

append("RESEARCH_STATE.md","EXP-054 authoritative result and EXP-055 frontier",f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`. Exact reduced diagrams remain E1 auxiliary reference machinery. The active frontier is `EXP-055 — Exact Column-Signature Popcount Aggregation Gate`, a word-level compiler that groups identical or sign-related weight columns and computes group activation counts rather than evaluating bit-level gates or paths. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.")
append("DECISION_LOG.md","D-027/D-028 — Reject reduced diagrams and select EXP-055",f"D-027 records EXP-054 authority {AUTH} {RESULT} Decision: `{DECISION}`.\n\nD-028 forbids variable-order-only continuation and selects exact word-level column-signature/popcount aggregation as EXP-055.")
append("FAILED_APPROACHES.md","F-022 — Exact reduced ordered decision diagrams as core","Reduced diagrams were exact and avoided compile ceilings, but global p50/p90 paths were 35%/95%, dense-random node growth was 1.6873x per input bit, storage projection reached 202.25 TiB, and order-search amortization exceeded one million queries. Do not continue by trying only more variable orders, reporting late-bit controls alone, or hiding both-order compile cost.")
append("ASSUMPTION_REGISTER.md","A-031/A-032 — Decision paths and word-level column aggregation","A-031 an exact reduced ordered diagram yields a universally short path: CONTRADICTED for the registered operator family. A-032 quantized target weight columns contain enough exact repeated/sign-related structure for word-level grouped popcount aggregation to close query and storage costs: ACTIVE UNVERIFIED for EXP-055.")
append("VALIDATION_MATRIX.md","EXP-054 addendum","| Claim | Evidence | Verdict |\n|---|---:|---|\n| EXP-054 exact equality | E1: 0 mismatch / 9,013,248 | PASS reference |\n| EXP-054 no truth table | E1: 0 cases | PASS |\n| EXP-054 p50 path <=10% | E1: 35% | REJECTED |\n| EXP-054 p90 path <=25% | E1: 95% | REJECTED |\n| EXP-054 no ceiling/fallback | E1: 0/48 | PASS |\n| EXP-054 storage <=1 TiB | PROJECTED: 202.2479 TiB | REJECTED |\n| EXP-054 adversarial growth <=1.5 | E1: 1.6873x/bit | REJECTED |\n| EXP-055 word-level grouping | E0 | NEXT GATE |")
append("ARCHITECTURE.md","Closed EXP-054 and active EXP-055 architecture","EXP-054 ROMTDD/ROBDD-like diagrams are auxiliary exact decision references. EXP-055 keeps signed modular score arithmetic at word level: compile input columns into exact vector signatures, group identical and optional exact-negated signatures, compute group popcounts, and add scaled score vectors. Group build, bit scans/popcounts, vector arithmetic, bytes, selector metadata, and fallback are mandatory.")
append("HARDWARE_VALIDATION_PLAN.md","EXP-054/055 hardware boundary","EXP-054 has no Phase-D route as core. EXP-055 hardware work is forbidden until a real small-checkpoint linear operation is replaced exactly and grouped signature bytes/operations close the 8 GiB and 1.185185% equations. Phase D remains NOT TESTED.")
append("REPRODUCIBILITY.md","EXP-054 frozen authority",f"Authority: {AUTH}\n\nVerify with `cd results/exp_054 && sha256sum -c checksums.sha256`. Original workflow hashes are preserved under `results/exp_054/artifacts/`; all completed binary diagrams are under `results/exp_054/raw/diagrams/`.")
append("docs/research/EXPERIMENT_054_REDUCED_DECISION_DIAGRAM.md","Final authoritative result",f"Authority: {AUTH}\n\n{RESULT}\n\nDecision: `{DECISION}`.")
for optional in ("docs/RESEARCH_PROGRESS_LEDGER.md","docs/SESSION_HANDOFF.md"):
    if Path(optional).exists(): append(optional,"EXP-054 handoff","Reduced decision diagrams are rejected as core. Continue with EXP-055 word-level column-signature/popcount aggregation or a materially new mechanism only.")

Path("NEXT_EXPERIMENT.md").write_text(f"""# Next Experiment

## Closed Gate — EXP-054

Authority: {AUTH}

{RESULT}

Decision:

```text
{DECISION}
```

## EXP-055 — Exact Column-Signature Popcount Aggregation Gate

### Mechanism change

For a binary activation vector and signed modular multi-class linear decision, represent each input column as the exact vector of class weights. Compile identical column signatures into groups and compute one activation count per group:

```text
score_vector = bias_vector + sum_g popcount(active bits in group g) * column_signature_g
```

Optional exact-negated grouping may use a canonical signature plus signed count only when modular equality is proved. Runtime states are not enumerated.

### Conditions

```text
G0 independent signed modular top-1 reference
G1 exact identical-column grouping
G2 exact sign-canonical grouping with proved modular reconstruction
G3 scalar and packed group-popcount evaluator
G4 sparse/repeated/low-rank structured controls
G5 dense-random and unique-column adversaries
G6 exact complete-domain validation
```

### Registered domains

Use binary inputs n=8/12/16/20/32/64, classes C=2/4/8, accumulator widths 8/12/16, and structured repeated-column, sparse, low-rank, dense-random, and forced-unique families.

### Accounting

```text
source columns and scalar weights
group count and signature bytes
group membership/index bytes
input bits scanned or popcount words
group popcount operations
scaled vector-add operations
p50/p90 logical scalar-operation fraction
p50/p90 logical bytes touched
compile time and storage
405B source-parameter projection
exact mismatch and fallback
```

Baseline is the exact dense scalar operation `C*n` signed conditional additions plus source weight reads. Bit scanning/popcount and all grouped vector operations are charged.

### Early rejection Gate

```text
exact mismatch >0
runtime state table used as representation
p50 operation fraction >10%
p90 operation fraction >25%
p50 byte fraction >10%
p90 byte fraction >25%
dense-random or unique-column p50 fraction >25%
projected grouped storage >1 TiB
non-degrading savings fail as n/classes grow
compile cost cannot be amortized within 1,000,000 queries
```

Failure decision:

```text
REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY
```

### Promotion boundary

Synthetic success still requires real checkpoint weight-column extraction and operation replacement, exact token/logit agreement, p90 fully-accounted target fraction <=1.185185%, non-degrading scaling, 8 GiB closure, and Phase-D evidence.

### Evidence boundary

```text
Phase A/B; evidence ceiling E1
real Transformer operation replacement NOT TESTED
405B / 8 GiB / CUDA / PCIe / SSD / TTFT / tokens/sec NOT TESTED
```

### Next exact action

1. implement exact column grouping and signed modular grouped evaluator;
2. add repeated, sign-related, sparse, low-rank, dense-random, and unique-column generators;
3. exhaustively validate small domains and use deterministic larger-domain controls;
4. measure grouped operations/bytes and storage projections;
5. freeze raw groups, manifests, checksums, scaling, and decision.
""",encoding="utf-8")
