# VORTEX Decision Log

Append-only decisions; authoritative run identity is read from committed result JSON when available.

## D-001 — Final target fixed

Arbitrary public unmodified Hugging Face dense model; runtime only; real 405B; <=8 GiB VRAM; original contract preserved; 4B-class user experience; independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

GitHub Actions CPU cannot be labeled target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, or peak-VRAM evidence.

Status: ACTIVE. Phase D NOT TESTED.

## D-003 — Phase A/B/C/D adopted

Status: ACTIVE.

## D-004 — E0–E7 adopted

Status: ACTIVE.

## D-005 — MEASURED/DERIVED/PROJECTED/UNVERIFIED separation adopted

Status: ACTIVE.

## D-006 — mmap/index/DAG components are auxiliary

Evidence: PR #50/#52/#54.

Status: ACTIVE.

## D-007 — Raw prefix enumeration rejected as core

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED.

## D-008 — Exact future-DAG accepted only as body compression

Evidence: 64->38 nodes; causal held-out start coverage 0%.

Status: AUXILIARY.

## D-009 — Core research must skip original operations causally on unseen prompts

Status: ACTIVE.

## D-010 — EXP-047 CPTC selected and executed

Status: EXECUTED.

## D-011 — EXP-047 correctness primitive accepted at E1

Authoritative source:

```text
results/exp_047/summary.json
```

Frozen summary currently records:

```text
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
525 cases
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial fallback 15/15
```

Decision: certificate/reference/fallback implementation passes Phase B under declared assumptions.

Status: ACCEPTED E1 PRIMITIVE ONLY.

## D-012 — Global-range CPTC-v1 not promoted

MEASURED:

```text
certified 4/525
fallback 99.238%
N=1024 mean evaluated 98.294%
positive control 10.449%
Python optimized/reference about 8.6–9.1x
```

PROJECTED:

```text
required simple target fraction before selector/fallback 1.185%
positive-control gap 8.817x
```

Decision: primitive correctness passed; architecture savings failed to establish a plausible path.

Status: REVISE.

## D-013 — Next Gate is oracle-tight/stratified audit

Use held-out current-token states from available unmodified small checkpoints. Compare global, non-deployable oracle-tight, and deployable stratified bounds before actual operation replacement.

Offline analysis remains below E2. If oracle-tight bounds remain high, reject range-only CPTC.

Status: ACTIVE NEXT GATE.

## D-014 — Freeze completed experiment measurements

EXP-047 workflow is manual `workflow_dispatch` only and never commits on reproduction. The frozen committed result directory is authoritative; documentation changes cannot regenerate timing.

Status: ACTIVE REPRODUCIBILITY RULE.
