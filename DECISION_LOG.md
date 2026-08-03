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

## D-009 — Core research must skip or amortize original operations causally on unseen prompts

Status: ACTIVE.

## D-010 — EXP-047 CPTC selected and executed

Status: EXECUTED.

## D-011 — EXP-047 correctness primitive accepted at E1

Authoritative source `results/exp_047/summary.json` records workflow `30793232558`, source SHA `74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79`, 525 cases, zero wrong accepts, zero fallback/reference mismatch, zero independent-bound mismatch, and 15/15 adversarial exact fallbacks.

Status: ACCEPTED E1 PRIMITIVE ONLY.

## D-012 — Global-range CPTC-v1 not promoted

MEASURED: 4/525 certified, 99.238% fallback, N=1024 mean evaluated 98.294%, positive control 10.449%, Python reference path about 8.6–9.1x slower than full summation.

PROJECTED: required target fraction before overhead 1.185%; positive control remained 8.817x above it.

Status: REVISE.

## D-013 — Oracle-tight/stratified audit required before further CPTC backend work

Status at registration: ACTIVE NEXT GATE.

## D-014 — Freeze completed experiment measurements

Completed experiment workflows must not silently overwrite frozen evidence. Reproduction writes to isolated output.

Status: ACTIVE REPRODUCIBILITY RULE.

## D-015 — EXP-047R completed with valid E1 evidence

Authoritative source:

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
artifact 8848886335
artifact SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
```

MEASURED:

```text
3 pinned trained dense checkpoints
18 held-out current-token states
wrong accepts 0
bound violations 0
C1 exact-state oracle median 100%
C1 exact-state oracle p90 100%
C2 median 100%
C2 p90 100%
C2 best 99.21875%
```

Status: EXECUTED, E1.

## D-016 — Range-based CPTC rejected as core

The pre-registered C1 oracle threshold was median <=10% and p90 <=25%. The exact realized min/max oracle evaluated 100% at both median and p90. Because C1 is more favorable than any deployable state range, further C2 metadata tuning or C3 variance adaptation cannot rescue the declared range-only core mechanism without changing mechanism class.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

The certificate/fallback primitive remains auxiliary. C3 is not implemented as a continuation of EXP-047R.

Status: REJECTED AS CORE; AUXILIARY PRIMITIVE RETAINED.

## D-017 — Next core Gate changes from weight skipping to block amortization

EXP-048 will test whether one exact full-target weight stream can be amortized across a long causally proposed token block using training-free partial-layer self-speculation and exact prefix verification.

A zero-cost perfect proposal still requires at least 85 accepted tokens per full target pass to meet the projected 1.185% target-equivalent stream fraction. Future-token oracle proposals are upper-bound controls only and never deployable evidence.

Status: ACTIVE NEXT GATE.
