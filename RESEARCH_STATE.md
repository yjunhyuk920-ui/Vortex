# VORTEX Research State

## Fixed target

Arbitrary public, unmodified Hugging Face dense 405B-class checkpoint; executor-only; one 8-GiB GPU; exact declared token and successor-state contract; same-machine native-4B-Q4 warm latency `p50 <=1.2x`, `p95 <=1.5x`.

## Operating mode

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Duplicate GitHub Actions reproduction is not required unless the user explicitly requests it.

## Latest authoritative completed Gate — EXP-104A

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

A full-vocabulary target-derived resident draft was fully accounted at target scale.

- registered population reproduced exactly: `405,849,243,648` parameters;
- realistic Q4 head plus one complete target layer: `2,809,790,464` bytes scanned per proposed token;
- ratio to ideal native-4B-Q4 weight bytes: `1.404895232x`;
- final p50 limit: `1.2x`;
- even metadata-free ideal Q4 head plus one layer: `1.322254336x`;
- a fully charged 8-GiB ledger fits three Q4 layers, but capacity does not rescue the p50 scan floor.

Every proposed autoregressive token requires one resident draft pass, so acceptance length does not amortize this scan. The mechanism is closed as a p50 core before checkpoint execution.

## Previous Gates

```text
EXP-103A REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
EXP-102A REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

## Active next Gate — EXP-105A

`Sub-Full-Head or Multi-Token-Per-Scan Transducer Gate`.

A new candidate must either:

1. avoid scanning all 128,256 LM-head rows with a causal, fully charged mechanism; or
2. produce more than one useful future token per resident weight scan without future target leakage.

It must not reintroduce the closed static top-k selector, Jacobi/Parareal future-state guess, or free HBM scan families.

## Not tested

- complete 405B target execution;
- physical complete 8-GiB allocation;
- target SSD/PCIe/HBM measurements;
- target CUDA/SASS;
- same-machine native-4B-Q4 final p50/p95 acceptance.
