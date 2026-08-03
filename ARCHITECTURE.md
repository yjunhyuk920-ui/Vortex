# VORTEX Architecture

## Mission boundary

VORTEX is a runtime, not a retrained model. It ingests an unmodified supported Hugging Face dense checkpoint and automatically constructs any runtime metadata.

## Target execution stack

```text
Hugging Face checkpoint
        |
        v
checkpoint/shard inspector
        |
        +--> automatic checksummed runtime-format compiler
        |
        v
causal token executor
        |
        +--> proposal or partial original-operation evaluation
        +--> causal certificate
        +--> certified commit OR declared exact/safe fallback
        |
        v
memory scheduler
        - VRAM hot state <=8 GiB
        - RAM/SSD cold state
        - charged transfers and fallback
        |
        v
original-model-compatible output contract
```

## Core requirement

The primary runtime must avoid a substantial fraction of original 405B weight reads and arithmetic on unseen prompts. A component is core only when it participates in that operation-skipping path.

## Mandatory interfaces

### Proposal

Produces a candidate operation result, interval, logit decision, or execution capsule without future generated tokens.

Must report bytes, operations, state, construction cost, and causal inputs.

### Certificate

Declares one explicit contract:

- deterministic exact;
- deterministic top-1;
- bounded-logit error;
- probabilistic top-1 with declared union-accounted `delta`.

Probabilistic certification is not deterministic exactness.

### Fallback

When certification fails, execute all omitted original work needed for the reference contract.

No silent approximation and no uncharged fallback stream.

### Memory virtualization

Separate:

- GPU hot metadata;
- KV cache;
- work buffers;
- repair/fallback tile;
- RAM cache;
- SSD format.

### Evidence

Every run emits phase/evidence/provenance, forward/layer/tile counts, logical/physical bytes where measured, fallback, wrong accepts, memory, timing distribution, and output agreement.

## Current component classification

### Auxiliary accepted

- safetensors discovery/slice access;
- compact40/aligned64 mmap pointer VM;
- atomic/checksummed format builder;
- bounded exact decision-index compiler;
- exact finite-horizon suffix DAG.

These may store metadata or repeated states but are not the operation-skipping principle.

### Rejected core families

See `FAILED_APPROACHES.md`.

### Active core research

EXP-047 statistical tile certification.

## EXP-047 v1 insertion point

```text
linear y = W x
W partitioned into input-dimension tiles
causal random tile permutation
partial scalar decision contributions observed
alpha-spending Serfling interval
accept decision or exact fallback
```

Phase-B correctness passed. Broad skip performance did not.

Authoritative MEASURED result:

```text
certified 4/525
fallback 521/525
N=1024 mean evaluated fraction 98.294%
positive control evaluated fraction 10.449%
wrong accepts 0
```

Architecture decision:

- retain `vortex_runtime/cptc.py` as reference certificate/fallback machinery;
- do not use one global range as the main executor;
- do not build a GPU backend from CPTC-v1 yet.

## Active revision architecture

### Stage R0 — real-checkpoint oracle audit

For current-token small-model states, fully compute exact decision tile contributions only as a non-deployable analysis oracle.

Compare:

- C0 current global range;
- C1 exact per-state min/max oracle range;
- C2 deployable static stratified tile bounds;
- C3 independently proven variance-adaptive finite-population bounds.

This identifies whether failure is intrinsic or caused by loose range metadata.

### Stage R1 — deployable bound compiler

Only if R0 shows a useful upper bound:

- compile checksummed per-layer/per-tile bound metadata from the original checkpoint;
- derive activation-dependent bounds without reading skipped weights;
- charge metadata, selector, randomization, and union budget;
- preserve exact fallback.

### Stage R2 — real operation replacement

Replace a real LM-head or selected projection during generation on unmodified small checkpoints, with held-out prompts and exact forward/tile accounting.

Offline full-contribution analysis is not E2.

### Stage R3 — model-wide propagation

A model-wide path needs either:

- direct final-token certification without reconstructing every hidden coordinate; or
- compositional operator certificates whose nonlinear propagation remains sound.

This is currently UNVERIFIED.

## Resource equations

```text
M_total = M_hot + M_kv + M_work + M_fallback <= 8 GiB
B_total/token = B_selector + B_normal + r_fallback * B_fallback
C_total/token = C_selector + C_normal + r_fallback * C_fallback
T_token >= max(B_total / effective_bandwidth,
               C_total / effective_throughput,
               serial_latency_floor)
```

PROJECTED same-bit traffic comparison:

```text
405B Q4 stream: 188.593 GiB
1.2x 4B Q4 allowance: 2.235 GiB/token
required evaluated fraction before overhead: 1.185%
```

Phase-D hardware terms remain NOT TESTED.

## Safety rule

No optimized path commits outside its declared certificate. Invalid metadata, numerical failure, or absent proof triggers exact fallback or abort.
