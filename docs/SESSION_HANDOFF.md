# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or model-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the final runtime is solved.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, workflows, PR comments, and raw JSON

Before reporting meaningful work, update and commit the ledger and this handoff.

## Current repository state

Latest decisions:

```text
PR #36  semantic-state program routing                    rejected
PR #37  prompt-compiled Hankel decision program           rejected
PR #38  perfect-oracle sparse repair                      rejected
PR #40  prompt-only nonlocal exact decision memory        rejected
PR #42  exact dense-operator information bound            accepted/merged
PR #44  metadata-aware direct/operator top-1 bound        accepted/merged
PR #46  end-to-end Llama final-token metadata bound       accepted/merged
PR #48  host-indexed cell-probe Gate                      accepted/merged
PR #50  mmap-backed host-indexed decision VM              accepted/merged
```

Latest main merge:

```text
PR #50 merge: a4a0e9b693184c9d5ea248822393998357df40db
```

Authoritative Experiment 044 evidence:

```text
branch: research/host-indexed-decision-vm
head: 8f029dde63984a3cf24f9ec2e9629c9e060d9352
certificate workflow: 30785924201
Python 3.10/3.12 CI + validation: 30785924118
raw JSON: results/host_indexed_decision_vm_gate.json
```

## Accepted lower-bound path

### Experiment 042

An actual bias-free Llama-style family exposed signed Q4 coefficients through final next-token winners.

```text
micro functions: 256 / 256
minimum winner margin: 0.24951063086132308
projected complete final-decision metadata: 26.158586645498872 GiB
```

This closed an all-resident 8 GiB exact-decision representation for the constructed family, but not sparse host lookup.

### Experiment 043

An explicit pointer-table representation proved:

```text
serial host misses: at least 249 / 256 tokens
logical host bytes/token: 4.86328125
explicit pointer host storage: 261.5858664549887 GiB
nonrepresentative packed CPU median: 224.27377 ns/probe
```

Decision: one serial host probe/token does not itself prove target failure. Advance to a constructive host VM.

## Accepted Experiment 044 result

### Binary format and atomic builder

The VM implements:

```text
64-byte versioned header
compact40 5-byte records
aligned64 8-byte records
Q4 value + exact next-address semantics
payload CRC32 and header CRC32
unique temporary build file
file fsync
atomic destination replace
parent-directory fsync attempt
```

Tests verify that a failed rebuild preserves an older valid destination and removes the temporary file.

### Reader and integrity

The mmap reader rejects:

```text
bad magic
unsupported or inconsistent format
truncation or trailing bytes
bad header checksum
bad payload checksum
out-of-range starts
out-of-range pointers
```

Exact replay matches the source pointer table. A 256-record LRU cache changes a repeated 256-token chain from 256 misses/256 mmap reads to 256 hits/0 mmap reads.

### Nonrepresentative CI benchmark

Workload:

```text
records: 262,144
chains: 1,024
steps/chain: 256
address samples/format: 20,000
```

Files:

```text
compact40: 1,318,976 bytes
aligned64: 2,105,408 bytes
compact saving: about 37.4%
```

Dependent reads:

```text
compact40 p50 / p99: 1,473 ns / 1,806.5 ns
aligned64 p50 / p99: 1,502 ns / 1,833.45 ns
```

Warm cached replay:

```text
compact40: 535.140625 ns/token
aligned64: 468.16796875 ns/token
```

A single compact first-replay sample was 22.56 µs/token versus 2.07 µs/token aligned. OS cache state was not controlled, so this outlier is not a format or target-hardware conclusion.

Decision: compact40 is the default v1 format because dependent mmap latency was effectively equal while storage was substantially smaller. Keep aligned64 as an alignment diagnostic.

### Target host-storage projection

```text
records: 56,175,137,076
starts: 219,434,129
compact40 records: 261.5858664549887 GiB
aligned64 records: 418.53738632798195 GiB
start table: 1.6349116638302803 GiB
compact40 total: 263.22077817842364 GiB
aligned64 total: 420.1722980514169 GiB
```

Timing was not projected to those sizes.

## Critical scope boundary

```text
portable mmap exact pointer VM: proven
atomicity and corruption handling: proven
exact replay and cache accounting: proven
CI timing target representative: false
pinned-memory/GPU lookup bridge: absent
released-model decision-index compiler: absent
real 405B execution: absent
physical runtime target: unsolved
```

The VM proves the representation is executable. It does not prove that arbitrary model behavior can be compiled into a finite decision index without enumerating an intractable context space.

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed pointer representation: implemented on CPU mmap
lookup mechanism: functionally viable
real model decision-index construction: unsolved
GPU integration: unsolved
405B/8 GiB/quality/wall-clock target: unsolved
```

## Prohibited overclaims

Do not:

- project CI nanoseconds to target hardware or 263 GiB files;
- treat mmap VM success as a model compiler;
- claim released model quality preservation;
- treat one deterministic pointer chain as arbitrary language-model behavior;
- hide decision-index build time or coverage;
- claim final success without real checkpoint and hardware evidence.

## Current frontier — Experiment 045 Decision-Index Compiler Gate

The bottleneck has moved from lookup to construction.

The next Gate asks:

> Can an exact or certifiably safe host decision index be generated automatically from an unmodified Hugging Face checkpoint without enumerating the entire token-context state space?

### First candidate

Compile a bounded exact decision graph from a real TinyLlama checkpoint:

```text
node key: exact token-prefix identity or exact KV/state fingerprint
record: next token + successor node
build source: original checkpoint execution only
training: none
```

Use a declared finite prompt grammar and horizon to make completeness measurable. The compiler must report:

```text
possible input prefixes
visited unique states
exact records generated
build model calls
duplicate-state reuse
file bytes
build time
coverage
held-out grammar paths
fallback frequency
```

### Required Experiment 045 sequence

1. Create `research/decision-index-compiler-gate`.
2. Add `docs/EXPERIMENT_045_DECISION_INDEX_COMPILER_GATE.md`.
3. Define a finite prompt grammar and exact completeness denominator.
4. Compile exact greedy transitions from TinyLlama without training.
5. Deduplicate only when exact state/token behavior is proven identical.
6. Export records into the Experiment 044 compact40 VM.
7. Replay every compiled path without the model and require exact tokens.
8. Evaluate held-out grammar compositions and count fallback model calls.
9. Charge build calls, build time, index bytes, coverage, and fallback.
10. Reject any claim of universality outside the declared grammar/horizon.
11. Decide whether compilation growth is sublinear enough to justify a broader adaptive compiler.
12. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.

## Correct communication

> Experiment 044 produced a functioning mmap-backed exact pointer VM. On a nonrepresentative CI CPU, compact40 and aligned64 dependent reads were both about 1.5 µs median, while compact40 used 37.4% less storage. This proves lookup is implementable, not that arbitrary model decisions can be compiled. The next Gate compiles a bounded exact decision graph from a real unmodified TinyLlama checkpoint, measures complete grammar coverage and fallback, and exports it to the VM.
