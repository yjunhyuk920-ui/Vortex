# VORTEX Decision Log

All architecture decisions are append-only. Later corrections reference prior entries rather than deleting them.

## D-001 — Final target remains fixed

Date: 2026-08-03

Decision: arbitrary public unmodified Hugging Face dense model, runtime replacement only, 405B flagship, <=8 GiB VRAM, original contract preserved, 4B-class user experience, independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

Date: 2026-08-03

Decision: GitHub Actions CPU results cannot be labeled as target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, or peak-VRAM measurements.

Status: ACTIVE, NON-NEGOTIABLE. Phase D is NOT TESTED.

## D-003 — Adopt Phase A/B/C/D validation separation

Date: 2026-08-03

Status: ACTIVE.

## D-004 — Adopt E0–E7 evidence scale

Date: 2026-08-03

Decision: E0 idea, E1 synthetic/reference, E2 small real-model replacement, E3 held-out causal generalization, E4 accessible representative hardware, E5 medium/large scaling, E6 target under 8 GiB, E7 405B at 4B-class performance.

Status: ACTIVE; supersedes older E0–E4-only scale.

## D-005 — Separate MEASURED, DERIVED, PROJECTED, UNVERIFIED

Date: 2026-08-03

Status: ACTIVE.

## D-006 — Reclassify decision-index replay work as auxiliary

Date: 2026-08-03

Evidence: PR #50, #52, #54.

Decision: valid auxiliary representation/replay components, not the unseen-prompt operation-skipping principle.

Status: ACTIVE.

## D-007 — Reject raw prefix enumeration as a core runtime

Date: 2026-08-03

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED CORE MECHANISM.

## D-008 — Accept exact future-DAG only as body compression

Date: 2026-08-03

Evidence: 64 records ->38 nodes, causal held-out start coverage 0%.

Status: ACCEPTED AUXILIARY COMPONENT.

## D-009 — Make unseen-prompt causal operation skipping the primary filter

Date: 2026-08-03

Decision: core work must answer the twelve questions in `RESEARCH_STATE.md`.

Status: ACTIVE.

## D-010 — Select EXP-047 CPTC

Date: 2026-08-03

Decision: test sampling-without-replacement finite-population certificates with exact fallback as a materially different response to failed deterministic signed residual bounds.

Status: EXECUTED.

## D-011 — Accept EXP-047 correctness primitive at E1

Date: 2026-08-03

Authoritative evidence:

```text
PR: #56
workflow: 30791851508
source head: d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b
cases: 525
wrong accepts: 0
fallback mismatches: 0
independent-bound mismatches: 0
adversarial exact fallback: 15/15
```

Decision:

The implemented alpha-spending Serfling interval, causal randomized order, deterministic replay, validation faults, and exact fallback satisfy the Phase-B correctness Gate.

Status: ACCEPTED E1 PRIMITIVE ONLY.

## D-012 — Do not promote global-range CPTC-v1 as the core executor

Date: 2026-08-03

MEASURED:

```text
certified: 4/525
fallback: 521/525 = 99.238%
mean evaluated fraction N=1024: 98.294%
positive control: 107/1024 = 10.449%
Python optimized/reference time ratio: about 8.8–9.1x in measured buckets
```

PROJECTED:

```text
simple target evaluated-weight fraction before overhead: 1.185%
positive-control fraction / target fraction: 8.817x
```

Decision:

The pre-registered positive control passed, but broad skip performance failed to establish a plausible execution path. Do not build a full backend or call this Phase-C performance success.

Status: REVISE CORE ARCHITECTURE.

## D-013 — Next decisive Gate is oracle-tight/stratified real-checkpoint audit

Date: 2026-08-03

Decision:

Use held-out states from available unmodified small checkpoints to distinguish intrinsic statistical difficulty from loose global ranges. Compare current bounds, non-deployable oracle-tight ranges, and deployable stratified checkpoint-derived bounds before actual operation replacement.

Rule:

Offline full-contribution analysis remains below E2. If oracle-tight ranges still require high tile fractions, reject range-only CPTC rather than tuning it.

Status: ACTIVE NEXT GATE.
