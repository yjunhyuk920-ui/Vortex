# VORTEX Research State

## Fixed target

Arbitrary public, unmodified Hugging Face dense 405B-class checkpoint;
executor-only; one 8-GiB GPU; exact declared output and required successor
state; same-machine native-4B-Q4 warm latency `p50<=1.2x`, `p95<=1.5x`.

## Operating mode

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Duplicate GitHub Actions execution is not required unless explicitly requested.

## Latest authoritative completed Gate — EXP-106A

```text
REJECT_DEPTH_COMPLETE_WIDTH_THIN_LINEAR_AND_BIT_SLICED_SURROGATES_AS_UNIVERSAL_CAUSAL_CORE
```

A concrete training-free all-layer Q4 width-thin surrogate was fully charged.
Widths 128 through 1,664 fit 8 GiB and keep the draft scan below 2.4 GB/token,
but every such bridge has a nonzero kernel. Exact finite-word controls produced
collided narrow states that a legal target head distinguishes at the first
token. A dense flat-spectrum Hadamard control shows even the best rank-deficient
linear approximation has relative operator-norm error 1.

Restoring full width removes the kernel but costs
`202.377974 GiB` resident and
`214.493273 GB` per proposed token. One
all-checkpoint bitplane alone costs `47.247070312 GiB`.

Local validation:

```text
8 focused tests passed
exact dense-encoder collision confirmed
512-step positive/negative full-depth controls
flat-spectrum dense operator control
byte-identical deterministic rerun
SHA-256 ledger PASS
GitHub Actions not run
```

## Previous authoritative Gate — EXP-105A

```text
REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE
NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE
```

## Active next Gate — EXP-107A

`Exact Distinct-State Shared-Weight-Sweep Gate`.

The next mechanism must share a value-changing weight computation among distinct
causal states rather than compressing state identity. It must charge the branch
matrix, weight-tile reads, arithmetic, nonlinear separation, KV/state,
workspace, actual `N/A`, and exact target verification.

## NOT TESTED

- complete 405B target execution;
- physical complete 8-GiB target allocation;
- target SSD/PCIe/HBM throughput;
- target CUDA/SASS;
- same-machine native-4B-Q4 final p50/p95 acceptance.
