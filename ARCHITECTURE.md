# VORTEX Architecture

## Mission boundary

VORTEX is a runtime, not a retrained model. It must ingest an unmodified supported Hugging Face dense checkpoint and automatically construct any runtime metadata it needs.

## Target execution stack

```text
Hugging Face checkpoint
        |
        v
Checkpoint inspector and shard reader
        |
        +--> offline/runtime-format compiler
        |       - no training or model modification
        |       - checksummed, reproducible metadata
        |
        v
Causal token executor
        |
        +--> cheap proposal or partial evaluation
        +--> causal certificate
        +--> certified commit OR exact fallback
        |
        v
Memory scheduler
        - VRAM hot state <= 8 GiB
        - RAM/SSD cold state
        - asynchronous but causally correct transfer
        |
        v
Original-model-compatible tokens/logits contract
```

## Core architecture requirement

The primary runtime must avoid a substantial fraction of original 405B weight reads and arithmetic on unseen prompts. A component is core only when it participates in that operation-skipping path.

## Mandatory core interfaces

### 1. Proposal interface

Produces a candidate operation result, hidden-state interval, logit winner, or execution capsule without assuming future generated tokens.

Must report:

- bytes read;
- operations performed;
- state used;
- construction cost;
- causal inputs.

### 2. Certificate interface

Determines whether omitted work can change the declared output contract.

Possible contracts:

- deterministic exact;
- deterministic top-1;
- bounded-logit error;
- probabilistic top-1 with declared `delta`.

The contract must be explicit. Probabilistic certification is not deterministic exactness.

### 3. Fallback interface

When certification fails, execute all omitted original work needed to reproduce the reference contract.

Requirements:

- no silent approximation;
- no uncharged fallback stream;
- fallback counts and bytes recorded;
- failure of the optimized path cannot produce a worse output than its declared contract.

### 4. Memory virtualization interface

Separates:

- GPU-resident hot metadata;
- KV cache;
- work buffers;
- repair/fallback tiles;
- host RAM cache;
- SSD runtime format.

All transfers must be labeled logical and physical where measurable.

### 5. Evidence interface

Each run emits machine-readable records containing:

```text
phase
evidence_level
MEASURED
DERIVED
PROJECTED
UNVERIFIED
forward_calls
layer_calls
weight_bytes_read
logical_transfer_bytes
physical_transfer_bytes_if_measured
fallback_count
certificate_accept_count
wrong_accept_count
peak_RSS
peak_VRAM_if_measured
latency_distribution
quality/token/logit agreement
```

## Current component map

### Core candidates

- EXP-047 Causal Probabilistic Tile Certificate: active E0.

### Auxiliary accepted components

- safetensors discovery and slice access;
- mmap compact40/aligned64 exact pointer VM;
- atomic/checksummed runtime-format builder;
- exact bounded decision-index compiler;
- exact finite-horizon future-suffix DAG.

These may support future compilers, caches, or metadata storage, but they are not the operation-skipping principle.

### Rejected core families

See `FAILED_APPROACHES.md`.

## EXP-047 planned insertion point

Initial primitive:

```text
linear operator y = W x
W partitioned into T input-dimension tiles
random permutation pi chosen causally
partial sums observed sequentially
confidence sequence bounds omitted contribution
certificate accepts declared decision or triggers exact fallback
```

Phase B operates on synthetic linear decisions. Phase C must replace a real operation in an unmodified checkpoint. Hook-only analysis cannot promote beyond E1.

## Flagship resource equations

For every proposed full path define:

```text
M_total = M_hot + M_kv + M_work + M_fallback <= 8 GiB
B_total/token = B_selector + B_normal + fallback_rate * B_fallback
C_total/token = C_selector + C_normal + fallback_rate * C_fallback
T_token >= max(B_total / measured_effective_bandwidth,
               C_total / measured_effective_throughput,
               serial_latency_floor)
```

Current target hardware terms are UNVERIFIED because Phase D is NOT TESTED.

## Safety rule

No optimized path may commit a result outside its declared certificate. If the certificate cannot be evaluated soundly, the runtime must fall back or abort rather than silently approximate.
