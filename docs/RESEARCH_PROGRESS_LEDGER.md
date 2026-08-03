# VORTEX Research Progress Ledger

Last updated: 2026-08-03 Asia/Seoul

Compatibility ledger. Current authoritative state is `RESEARCH_STATE.md`; permanent failures and decisions are in `FAILED_APPROACHES.md` and `DECISION_LOG.md`.

## Fixed target

Arbitrary public unmodified Hugging Face dense model, runtime replacement only, real 405B, <=8 GiB VRAM, original contract preserved, and 4B-class user-perceived performance.

Phase D: **NOT TESTED**.

## Governance reset — PR #56

Implemented:

- Phase A/B/C/D;
- E0–E7;
- MEASURED/DERIVED/PROJECTED/UNVERIFIED;
- explicit Phase-D NOT TESTED rule;
- root research state/decision/failure/assumption/validation/hardware/reproducibility files;
- unseen-prompt operation-skipping requirement for core work.

Existing mmap/index/DAG work is auxiliary. Raw exact-prefix scaling is rejected.

## Prior proof and auxiliary milestones

- PR #42: exact dense-operator information lower bound; metadata is not traffic.
- PR #44: metadata-aware direct/operator top-1 bound.
- PR #46: constructed end-to-end Llama final-decision metadata 26.1586 GiB; sparse host access remained open.
- PR #48: near-one host probe/token can still be only a few logical bytes; probe count is not latency.
- PR #50: checksummed atomic mmap exact pointer VM.
- PR #52: bounded TinyLlama compiler replayed 72/72 checked tokens but raw distinct prefixes were 64/64 and held-out start coverage 0%.
- PR #54: exact future-suffix DAG compressed 64 records to 38 nodes but causal held-out start coverage remained 0%.

Detailed permanent numbers remain in Git history and root failure/decision registers.

## EXP-047 — Causal Probabilistic Tile Certificate

Final authoritative identity:

```text
PR: #56
workflow: 30792813542
source SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
evidence head after workflow: 3359371762c004db3532ebb16872b4eee85accf6
phase: A/B
evidence: E1
Phase D: NOT TESTED
```

Mechanism:

- causal random sampling without replacement of decision-relevant linear tiles;
- fixed-step Serfling interval;
- alpha spending `delta_n = delta*6/(pi^2 n^2)` for adaptive stop;
- exact evaluation of all remaining tiles when no certificate closes.

### MEASURED correctness

```text
10 tests passed
525 cases
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial fallback 15/15
future generated tokens used false
```

Decision: E1 reference certificate/fallback primitive accepted.

### MEASURED architecture signal

```text
certified 4/525
fallback 99.238%
mean evaluated fraction N=512 98.519%
mean evaluated fraction N=1024 98.294%
positive control 107/1024 = 10.449%
Python optimized/reference mean time about 9.2–9.7x
```

Decision: one global-range CPTC-v1 is not promoted; core architecture status REVISE.

### PROJECTED target gap

```text
405B Q4 stream 188.593 GiB
1.2x 4B Q4 allowance 2.235 GiB/token
required fraction before selector/fallback 1.185%
positive-control fraction 8.817x above target fraction
```

Not measured on target hardware.

## Infrastructure failures excluded from science

- `30791055142`: eager optional dependency import;
- `30791192434`: missing repository root on `PYTHONPATH`.

Final authoritative success only: `30792813542`.

## Current frontier

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`.

Use held-out current-token states from available unmodified small checkpoints. Compare current global, non-deployable oracle-tight, and deployable stratified bounds. Offline full-contribution analysis remains below E2.

If even oracle-tight bounds require high tile fractions, reject range-only CPTC rather than tune it.

## Current classification

```text
Governance/provenance implemented
Auxiliary mmap/index/DAG bounded evidence retained
EXP-047 correctness E1 PASS
EXP-047 broad savings FAIL/REVISE
Real operation replacement NOT TESTED
70B/405B scaling NOT TESTED
8 GiB target execution NOT TESTED
E6/E7 not achieved
```
