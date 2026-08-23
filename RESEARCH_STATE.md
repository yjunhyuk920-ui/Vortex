# VORTEX Research State

## Fixed target

Arbitrary public, unmodified Hugging Face dense 405B-class checkpoint; executor-only; one 8-GiB GPU; exact declared output and required successor state; same-machine native-4B-Q4 warm latency `p50 <=1.2x`, `p95 <=1.5x`.

## Operating mode

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Duplicate GitHub Actions execution is not required unless explicitly requested.

## Latest authoritative completed Gate — EXP-105A

```text
REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE
NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE
```

EXP-105A implemented an exact query-dependent Q4/Q8 vocabulary tournament. Current activation dimensions were processed in descending magnitude; exact int64 partial logits and exact Cauchy tail bounds eliminated rows only when they could no longer win.

Local validation:

- 8 focused tests passed;
- runtime winner is separate from the dense validation reference;
- 12 random finite-word queries across 6 block sizes, zero winner mismatch;
- byte-identical deterministic rerun;
- checksum ledger passed;
- GitHub Actions not run.

Decisive target-scale result:

- static head + one layer + reserves: `4,330,597,376` bytes, so capacity fits 8 GiB;
- legal late-decision Q4 head requires complete head traffic;
- best charged head query: `1,122,065,408` bytes/token;
- one complete realistic-Q4 target-width layer: `1,693,450,240` bytes/token;
- combined: `2,815,515,648` bytes/token;
- p50 budget: `2,400,000,000` bytes/token;
- ratio: `1.17313152x` the p50 budget.

The head alone fits the byte budget but is not a complete causal transducer and has no measured `A>=339` source. No nonexistent state source is credited.

The one-scan multi-token candidate also fails the registered causal dependency Gate: without guessed branches or a precompiled transition operator, token `t+1` cannot enter the first weight-bearing stage before token `t` head selection.

## Previous authoritative Gates

```text
EXP-104A REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
EXP-103A REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
EXP-102A REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

## Active next Gate — EXP-106A

`Depth-Complete Width-Thin Checkpoint Surrogate Reality Gate`.

The next candidate must supply a concrete causal state source cheaper than one complete 405B-width layer. It will derive a training-free draft-only narrow state from every target layer rather than retaining a few complete layers. Before public-checkpoint execution it must freeze all projected weights, input/output bridges, LM-head/index bytes, KV, workspace, metadata, and per-token operations below the p50 budget. It then must measure real accepted prefix and exact target verification; configured width or K is not evidence.

## `NOT_TESTED`

- complete 405B target execution;
- physical complete 8-GiB target allocation;
- target SSD/PCIe/HBM throughput;
- target CUDA/SASS;
- same-machine native-4B-Q4 final p50/p95 acceptance.
