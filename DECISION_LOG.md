# VORTEX Decision Log

Append-only architecture decisions. Corrections explicitly supersede prior run identity without deleting scientific history.

## D-001 — Final target remains fixed

Date: 2026-08-03

Arbitrary public unmodified Hugging Face dense model; runtime replacement only; real 405B; <=8 GiB VRAM; original contract preserved; 4B-class user experience; independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

GitHub Actions CPU results cannot be labeled target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, or peak-VRAM measurements.

Status: ACTIVE. Phase D NOT TESTED.

## D-003 — Adopt Phase A/B/C/D validation separation

Status: ACTIVE.

## D-004 — Adopt E0–E7 evidence scale

E0 idea, E1 synthetic/reference, E2 small real-model operation replacement, E3 held-out causal generalization, E4 accessible representative hardware, E5 medium/large scaling, E6 target under 8 GiB, E7 real 405B at 4B-class performance.

Status: ACTIVE.

## D-005 — Separate MEASURED, DERIVED, PROJECTED, UNVERIFIED

Status: ACTIVE.

## D-006 — Reclassify mmap/index/DAG work as auxiliary

Evidence: PR #50/#52/#54.

Status: ACTIVE.

## D-007 — Reject raw prefix enumeration as core runtime

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED CORE MECHANISM.

## D-008 — Accept exact future-DAG only as body compression

Evidence: 64 records ->38 nodes; causal held-out start coverage 0%.

Status: ACCEPTED AUXILIARY COMPONENT.

## D-009 — Make unseen-prompt causal operation skipping the primary filter

Core work must answer the twelve questions in `RESEARCH_STATE.md`.

Status: ACTIVE.

## D-010 — Select EXP-047 CPTC

Test causal sampling-without-replacement finite-population certificates with exact fallback as a materially different response to failed deterministic signed-residual bounds.

Status: EXECUTED.

## D-011 — Accept EXP-047 correctness primitive at E1

Final authoritative evidence:

```text
PR: #56
workflow: 30792813542
source implementation SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
committed result head after workflow: 3359371762c004db3532ebb16872b4eee85accf6
cases: 525
wrong accepts: 0
fallback mismatches: 0
independent-bound mismatches: 0
adversarial exact fallback: 15/15
```

Decision: the alpha-spending Serfling implementation, causal randomized order, deterministic replay, validation faults, and exact fallback satisfy Phase B.

Status: ACCEPTED E1 PRIMITIVE ONLY.

## D-012 — Do not promote global-range CPTC-v1 as core executor

MEASURED:

```text
certified: 4/525
fallback: 521/525 = 99.238%
mean evaluated fraction N=1024: 98.294%
positive control: 107/1024 = 10.449%
Python optimized/reference time: about 9.2–9.7x
```

PROJECTED:

```text
required simple target fraction before selector/fallback: 1.185%
positive-control fraction / target fraction: 8.817x
```

Decision: primitive Gate passed, architecture performance did not. No full backend or Phase-C performance claim.

Status: REVISE CORE ARCHITECTURE.

## D-013 — Next Gate is oracle-tight/stratified real-checkpoint audit

Use held-out current-token states from available unmodified small checkpoints to distinguish intrinsic statistical difficulty from loose global ranges. Compare current global, non-deployable oracle-tight, and deployable stratified bounds before actual operation replacement.

Offline analysis remains below E2. If oracle-tight bounds still require high tile fractions, reject range-only CPTC rather than tune it.

Status: ACTIVE NEXT GATE.

## D-014 — Freeze measurement workflow boundary

Only implementation/config/tests/workflow changes rerun EXP-047. Result interpretation and root-document changes do not regenerate raw timing. Workflow commits complete raw evidence to the source branch after Gate success.

Status: ACTIVE REPRODUCIBILITY RULE.
