# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological index of hypotheses, executable Gates, measurements, accepted proof constraints, and rejection reasons. Detailed experiment documents, PRs, workflows, raw JSON, and Git history remain permanent authoritative data.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is final physical runtime completion.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Negative experiments remain permanent project data.
- Accepted proofs are guardrails, not runtime completion.
- Separate exact output, top-1 functions, final Transformer decisions, metadata size, logical bytes, physical transactions, and wall clock.
- Charge build, storage, host state, lookup, transfer, validation, cache, and fallback.
- Metadata size is not per-token traffic.
- Serial probes are not latency.
- CPU mmap success is not a model decision compiler.
- Before reporting meaningful work, update this ledger and `docs/SESSION_HANDOFF.md`.

## Foundational envelope

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse: 246.889 tokens
```

This was a conditional E0/E1 target envelope only.

## Rejected representation families

### Static dictionaries, factorization, activation atlases, and entropy

- Progressive Q2/Q3 failed quality; Q4 speculation needed more than one thousand accepted tokens to amortize a full target stream.
- Recurrent layer dictionaries fit storage but still read about 188 GiB/committed token.
- Global and semantic Kronecker approximations passed some memory projections but produced near-unit error and zero causal prefix.
- Gauge dictionaries and functional skeletons passed primitives but failed full-model decisions.
- Static activation ranks left continuation perpendicular energy near one.
- Online expansion became exact/tokenwise.
- ZIPTREE measured 11.3330 bits/weight and required about 10,649 accepted tokens.

Decision: reject static low-rank, generic factorization, activation-subspace caching, and ordinary whole-model lossless streaming as primary mechanisms.

### Exact-neuron and signed-residual sequence

- Optimistic neuron subsets reached at most two exact tokens while MLP traffic rose from about 0.623 to 12.285 GiB/token.
- PR #29 measured 132 single-layer damage points; nonlinear allocation still produced no useful prefix.
- PR #31 global Signed Dual Cone: about 97.93% mean refinement, 610.64 GiB/token.
- PR #32 partitioned residual bounds: about 96.46%, 607.6 GiB/token.
- PR #33 block signed residual code: 92.39%, 585.8 GiB/token.
- PR #34 global dual-price allocation: 90.74%, maximum 573.34 GiB/token.

Decision: signed cancellation exists, but exact refinement remains dominant.

## Rejected dynamic and prompt-derived programs

### PR #36 — Semantic-State Program Routing

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best mean reuse 1.364 tokens; switch traffic 0.7261 GiB/token; activation/dual perpendicular p95 near 100%. Rejected.

### PR #37 — Prompt-Compiled Hankel Program

```text
head: 12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow: 30778715832
```

Autonomous exact prefixes 1, 1, and 2 versus required 247. Rejected.

### PR #38 — Perfect-Oracle Sparse Repair

```text
head: 13e3f60876199e4b06577ca51e9fd71f575cb134
workflow: 30779062125
```

The impossible oracle still repaired 174–229 of 256 tokens, projecting 128.45–169.06 GiB/token and 551.70–726.09 GFLOP/token. Rejected.

### PR #40 — Nonlocal Exact Decision Memory

```text
head: 91b3e3f062d33087005ae38bbf94b357012f0ccd
corrected workflow: 30780847944
corrected CI: 30780847954
```

The initial boundary-misaligned run is invalid. Corrected future-aware global suffix maxima were 75, 28, and 5 versus required 247. Rejected independent of ANN configuration.

## Accepted proof guardrails

### PR #42 — Exact Dense-Operator Information Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate: 30781557141
full CI: 30781557096
merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

```text
405B Q4 information: 188.98828125 GiB
resident allowance: 8 GiB
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
4B arithmetic ratio: 101.462310912x
```

115/115 skipped-coordinate adversaries changed exact output and unique top-1. Scope: exact-output theorem and coordinate relevance.

### PR #44 — Metadata-Aware Top-1 Function Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate: 30782192795
full CI: 30782192768
merge: aca6657578b0decb58adbf98bcd22555169a6847
```

Selector/payload family implements `2^K` distinct functions for `K=p(d-p)`. Exhaustive 2/16/64/512-function families were injective. Independently callable Llama-shaped operator collection lower bound: 9.5977783203125 GiB.

### PR #46 — End-to-End Llama Final-Decision Metadata Bound

```text
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate: 30784848053
full CI: 30784848049
merge: 038d3fa72dbfe91f4d9837d482b9f9c10719a00f
raw JSON: results/llama_final_decision_routing_bound.json
```

Actual Llama-style family:

```text
micro functions: 256 / 256
exact Q4 recovery: true
minimum winner margin: 0.24951063086132308
projected Q4 decision coefficients: 56,175,137,076
complete final-decision metadata: 26.158586645498872 GiB
```

Scope: all-resident 8 GiB complete representation contradicted for the family; sparse host lookup remained open. Pre-correction whole-target impossibility language is invalid.

### PR #48 — Host-Indexed Cell-Probe Gate

```text
head: 2705613f943f36adc041a6a4bedd7eba5c42f2ac
certificate: 30785452594
full CI: 30785452584
merge: 4ca9d2c2d4876d1266b2ad5527e2350585c7db7c
raw JSON: results/host_indexed_cell_probe_gate.json
```

Explicit pointer theorem:

```text
worst-chain misses >= T - floor(C/S)
```

Target projection:

```text
serial host misses: at least 249 / 256
logical host bytes/token: 4.86328125
explicit pointer table: 261.5858664549887 GiB
```

Packed nonrepresentative CPU pointer chase median: 224.27377 ns/probe. Decision: reject probe count alone as an impossibility argument; build the host VM.

## Accepted constructive implementation

### PR #50 — mmap-Backed Host-Indexed Decision VM

```text
head: 8f029dde63984a3cf24f9ec2e9629c9e060d9352
certificate workflow: 30785924201
full CI: 30785924118
merge: a4a0e9b693184c9d5ea248822393998357df40db
raw JSON: results/host_indexed_decision_vm_gate.json
```

Implemented:

```text
64-byte versioned header
compact40 5-byte records
aligned64 8-byte records
payload and header CRC32
atomic temporary build and replace
file and parent fsync
mmap reader
strict corruption rejection
exact pointer replay
bounded LRU record cache
sequential/random/dependent/reopen benchmarks
```

Functional Gate:

```text
records: 262,144
chains / steps: 1,024 / 256
atomic replace: pass
failed rebuild preserves valid destination: pass
bad magic/truncation/corruption/invalid pointer: rejected
second cached replay: 256 hits / 0 mmap reads
```

Nonrepresentative CI format comparison:

```text
compact40 file: 1,318,976 bytes
aligned64 file: 2,105,408 bytes
compact storage saving: about 37.4%

compact dependent p50 / p99: 1,473 ns / 1,806.5 ns
aligned dependent p50 / p99: 1,502 ns / 1,833.45 ns

compact warm cached replay: 535.14 ns/token
aligned warm cached replay: 468.17 ns/token
```

One compact first-replay sample was 22.56 µs/token and aligned was 2.07 µs/token. OS cache state was uncontrolled; this single outlier is not used for format or target-hardware conclusions.

Decision: compact40 is the default v1 format. Aligned64 remains a diagnostic option.

Target host/disk projection:

```text
compact40 records: 261.5858664549887 GiB
aligned64 records: 418.53738632798195 GiB
start table: 1.6349116638302803 GiB
compact40 total: 263.22077817842364 GiB
aligned64 total: 420.1722980514169 GiB
```

Scope:

```text
portable CPU mmap pointer VM: proven
exact replay, integrity, and cache accounting: proven
CI timing target representative: false
released-model decision-index compiler: absent
GPU/pinned-memory integration: absent
real 405B execution: absent
```

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed representation: functionally implemented on CPU mmap
lookup mechanism: no longer a purely theoretical blocker
exact decision-index compiler from arbitrary checkpoint: unsolved
GPU-facing lookup: unsolved
405B/8 GiB/quality/wall-clock target: unsolved
```

## Prohibited repeats and overclaims

Do not:

- relabel metadata as traffic;
- relabel probes as latency;
- project CI nanoseconds to target hardware or 263 GiB files;
- treat the pointer VM as a language-model decision compiler;
- hide index build cost, grammar coverage, or fallback;
- claim released-model quality preservation;
- claim physical success or impossibility without real evidence.

## Current frontier — Experiment 045 Decision-Index Compiler Gate

Lookup is now implemented. Construction is the primary unknown.

First bounded compiler target:

```text
model: unmodified TinyLlama checkpoint
training: none
input domain: declared finite prompt grammar
horizon: explicit and finite
node key: exact prefix identity or exact state fingerprint
record: exact greedy token + successor node
output: Experiment 044 compact40 VM
```

### Mandatory Experiment 045 work

1. Create `research/decision-index-compiler-gate`.
2. Add `docs/EXPERIMENT_045_DECISION_INDEX_COMPILER_GATE.md`.
3. Define a finite prompt grammar with an exact completeness denominator.
4. Compile greedy transitions using only original checkpoint calls.
5. Deduplicate only with exact identity or a sound equivalence proof.
6. Export the compiled graph to compact40.
7. Replay every compiled path without model execution and require exact tokens.
8. Evaluate held-out grammar compositions and count fallback calls.
9. Record model calls, unique states, duplicate reuse, build time, index bytes, coverage, fallback, and growth by horizon.
10. Fit growth models but do not extrapolate beyond measured horizons as proof.
11. Reject universality claims outside the declared grammar.
12. Decide whether state growth is sufficiently sublinear to justify adaptive compilation.
13. Commit tests, workflow, raw JSON, PR decision, this ledger, and handoff.
