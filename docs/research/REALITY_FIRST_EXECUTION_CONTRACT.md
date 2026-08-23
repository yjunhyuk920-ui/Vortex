# VORTEX Reality-First Execution Contract

This contract is normative for every new core research round after EXP-101A. A mechanism that succeeds only after granting impossible future information or zero-cost runtime components is not a credible route to the fixed VORTEX mission.

## 1. Authoritative arm

Every new core Gate has exactly one authoritative arm:

```text
REAL_EXECUTOR_ONLY
```

It must be executable as written with finite-word data, causal inputs, declared resources, and a fail-closed exactness path. A missing component is `NOT_EXECUTABLE` or `NOT_TESTED`; it is never replaced by a free oracle.

## 2. Forbidden promotion grants

The authoritative arm may not receive:

```text
future target tokens or hidden states
perfect target-seeing selector
perfect accepted block
free N/A = 1
free transforms, additions, packing, or address generation
free metadata, side state, or workspace
free verification, native-order repair, rollback, or fallback
free decompression or unmeasured compression ratio
peak hardware throughput presented as measured sustained throughput
unbounded offline output/state tables
remote or hidden execution omitted from the machine ledger
```

Synthetic fixtures may simplify only unit-test inputs. They cannot establish scientific promotion or resource feasibility.

## 3. Required executable accounting

Charge, measure, or conservatively bound every required component:

- candidate generation and causal state maintenance;
- every target position evaluated and the resulting `N/A`;
- exact verification, rejected work, mismatch repair, rollback, and fallback;
- transforms, additions, multiplications, scales, packing, writes, and synchronization;
- checkpoint bytes, lossless representation, decoder throughput, SSD/host/device traffic;
- GPU VRAM, host RAM, KV/cache, metadata, workspace, fragmentation, and allocator state;
- p50/p95 wall time against the same-machine baseline when hardware is available.

Unmeasured compression receives ratio `1.0`. Unmeasured bandwidth and throughput remain `NOT TESTED`; they receive no ideal value.

## 4. Causality and exactness

A deployable candidate is completed without observing the target continuation it predicts. Exactness covers the declared token/output contract and every required successor-state byte. A block or reassociated kernel that changes the declared incremental ABI fails unless a real, charged repair path restores it.

## 5. Evidence boundary

A small real-checkpoint run establishes at most E2/E3 evidence. Target storage, 8-GiB residency, CUDA/SASS behavior, and same-machine 4B-Q4 p50/p95 remain `NOT TESTED` until directly measured.

## 6. End-of-round reality Gate

```text
CAUSAL_INPUTS=true
EXECUTABLE_PATH=true
FINITE_WORD_PATH=true
EXACT_OR_FAIL_CLOSED=true
ALL_REQUIRED_COSTS_CHARGED=true
N_OVER_A_MEASURED=true
FALLBACK_CHARGED=true
MEMORY_LEDGER_COMPLETE=true
NO_FORBIDDEN_GRANT=true
README_CURRENT=true
REMOTE_COMMIT_VERIFIED=true
```

Failure of any item prevents core promotion.
