# VORTEX Research Progress Ledger

Last updated: 2026-08-03 Asia/Seoul

This chronological compatibility ledger supplements the root research state files. Authoritative current state is in `RESEARCH_STATE.md`; permanent rejections are in `FAILED_APPROACHES.md`; decisions are in `DECISION_LOG.md`.

## Fixed target

Arbitrary public unmodified Hugging Face dense model, runtime replacement only, real 405B flagship, <=8 GiB GPU VRAM, original declared ability/output contract preserved, and 4B-class user-perceived performance on the same target machine.

Current Phase D status: **NOT TESTED**.

## Evidence governance adopted

PR #56 introduces:

- Phase A/B/C/D separation;
- E0–E7 evidence scale;
- MEASURED/DERIVED/PROJECTED/UNVERIFIED separation;
- root state, decision, failure, assumption, validation, hardware, and reproducibility documents;
- prohibition on representing GitHub CPU work as 405B/GPU/PCIe/CUDA evidence;
- direct unseen-prompt operation-skipping filter for core research.

## Prior negative execution families

Persisted in `FAILED_APPROACHES.md`:

- low-rank/factorization/dictionary/activation-subspace families;
- progressive precision as primary path;
- exact-neuron selection;
- deterministic signed residual refinements;
- prompt recurrent programs;
- oracle sparse repair;
- suffix memory/replay;
- raw exact-prefix graphs;
- future-aware DAG as standalone runtime.

Do not recreate them under new names without addressing the recorded failure.

## Accepted proof guardrails

### PR #42 — exact dense-operator information bound

```text
405B Q4 exact information: 188.98828125 GiB
minimum information beyond 8 GiB: 180.98828125 GiB
115/115 coordinate adversaries flipped exact output/top-1
```

Scope: exact-output and coordinate relevance; not per-token traffic.

### PR #44 — metadata-aware direct/operator top-1 bound

```text
independently callable Llama-shaped operator collection: 9.5977783203125 GiB
```

Scope: direct/operator collection, not full final-token Transformer until PR #46.

### PR #46 — end-to-end Llama final-token metadata bound

```text
micro functions: 256/256
minimum winner margin: 0.24951063086132308
projected complete decision metadata: 26.158586645498872 GiB
```

Scope: constructed family closes all-resident 8 GiB representation; sparse host lookup remained open. Earlier pre-correction whole-target impossibility language is invalid.

### PR #48 — explicit host cell-probe Gate

```text
worst-chain misses: at least 249/256
logical bytes/token: 4.86328125
nonrepresentative CPU median: 224.27377 ns/probe
```

Decision: probe count alone does not prove latency failure.

## Accepted auxiliary implementations

### PR #50 — mmap host decision VM

Checksummed/atomic compact40/aligned64 files, strict corruption rejection, exact replay, and LRU cache.

```text
compact dependent p50/p99: 1,473/1,806.5 ns
aligned dependent p50/p99: 1,502/1,833.45 ns
compact storage saving: about 37.4%
```

CPU CI timing is nonrepresentative.

### PR #52 — bounded TinyLlama decision-index compiler

```text
compiled paths: 8 + exact duplicate control
model calls: 64
exact VM replay: 9/9 paths, 72/72 tokens
raw distinct-prefix nodes: 64/64
held-out start coverage: 0%
```

Implementation accepted; raw prefix scaling rejected.

### PR #54 — exact future-suffix DAG

```text
64 raw records ->38 exact nodes
DAG VM 326 bytes vs raw VM 456 bytes
exact replay 72/72
causal held-out start coverage 0%
```

Body compression accepted as auxiliary; future-token routing forbidden.

## EXP-047 — Causal Probabilistic Tile Certificate

Branch: `research/governance-exp047-cptc`

PR: `#56`

Authoritative workflow: `30791851508`

Source head recorded by run: `d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b`

Evidence: Phase A/B, E1. Phase D NOT TESTED.

### Mechanism

For decision-relevant tile contributions `z_i`, sample tiles uniformly without replacement in a causal random order. Use fixed-step Serfling intervals with alpha spending:

```text
delta_n = delta * 6/(pi^2 n^2)
```

Commit only when the total interval excludes zero. Otherwise evaluate all remaining tiles exactly.

### MEASURED correctness

```text
unit/property tests: 10 passed
cases: 525
wrong accepts: 0
fallback/reference mismatches: 0
independent-bound mismatches: 0
adversarial exact fallback: 15/15
future generated tokens used: false
```

Decision: correctness primitive accepted at E1.

### MEASURED performance signal

```text
certified: 4/525
fallback: 521/525 = 99.238%
N=64/128/256 mean evaluated fraction: 100%
N=512 mean evaluated fraction: 98.519%
N=1024 mean evaluated fraction: 98.294%
positive cancellation control: 107/1024 = 10.449%
Python optimized/reference time: roughly 8.8–9.1x
```

Decision: global-range CPTC-v1 is not promoted as the core executor; architecture status REVISE.

### PROJECTED target gap

```text
405B Q4 stream: 188.593 GiB
4B Q4 stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required fraction before selector/fallback: 1.185%
positive-control fraction / target: 8.817x
```

These are parameter-count projections, not target measurements.

### Infrastructure corrections

Two prior runs failed before scientific measurement:

1. eager package import required optional `safetensors`;
2. runner lacked repository root on `PYTHONPATH`.

Lazy imports and explicit `PYTHONPATH` corrected them. They are infrastructure failures, not hypothesis evidence.

## Current frontier

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`.

Use available unmodified small checkpoints and held-out prompts to compare:

- current global range;
- non-deployable per-state oracle-tight range;
- deployable checkpoint-derived stratified bounds;
- independently justified variance-adaptive finite-population bounds.

If even oracle-tight real-state intervals need high tile fractions, reject range-only CPTC instead of tuning it.

## Current classification

```text
Governance/provenance: implemented
Auxiliary mmap/index/DAG: bounded functional evidence
EXP-047 correctness: E1 PASS
EXP-047 broad execution savings: FAIL/REVISE
Real operation replacement: NOT TESTED
70B/405B scaling: NOT TESTED
8 GiB target execution: NOT TESTED
E6/E7: not achieved
```
