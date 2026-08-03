# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with:

- one 8 GiB VRAM GPU;
- no user training, distillation, fine-tuning, LoRA, or model-specific adapter authoring;
- original-model decisions and quality preserved;
- p50 warm decode at or below 1.2x a native 4B Q4 baseline on the same machine;
- flagship validation on a real 405B-class model.

Current evidence remains below E4. Do not claim the physical runtime target is solved or impossible.

## Mandatory startup and persistence

Read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. this file
6. active experiment documents, workflows, PR comments, and raw JSON

Before a user-facing progress/completion answer after meaningful work, update and commit the ledger and this handoff.

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
PR #48  host-indexed exact-decision cell-probe Gate       accepted/merged
```

Latest main merge:

```text
PR #48 merge: 4ca9d2c2d4876d1266b2ad5527e2350585c7db7c
```

Authoritative Experiment 043 evidence:

```text
branch: research/host-indexed-cell-probe-gate
head: 2705613f943f36adc041a6a4bedd7eba5c42f2ac
certificate workflow: 30785452594
Python 3.10/3.12 CI + validation: 30785452584
raw JSON: results/host_indexed_cell_probe_gate.json
```

## Accepted Experiment 042 guardrail

Experiment 042 constructed an actual bias-free Llama-style family using embeddings, RMSNorm, causal GQA, residuals, SwiGLU, final RMSNorm, and an LM head. A legal prompt selected signed-Q4 `up_proj` codes and final next-token winners decoded them.

```text
micro functions: 256 / 256
minimum winner margin: 0.24951063086132308
projected independent Q4 coefficients: 56,175,137,076
complete final-decision metadata: 26.158586645498872 GiB
resident allowance: 8 GiB
```

Accepted scope:

```text
all-resident complete exact-decision representation: contradicted for the family
per-token external traffic: not proven
host-indexed sparse lookup: open
physical runtime target: unsolved
```

## Accepted Experiment 043 result

### Explicit adaptive representation

```text
record[address] = (q4_value, next_address)
next_token_t = q4_value
address_(t+1) = next_address
```

For `S` disjoint chains of length `T` and a resident cache of `C` complete records, some chain contains at most `floor(C/S)` cached records and forces:

```text
host misses >= T - floor(C/S)
```

Balanced cache placements attained the bound exactly. Ten sampled cache placements never violated it. Early, middle, and late indistinguishable-table adversaries changed only the addressed record while preserving the entire prior trace, then changed the current token and next address.

### Target projection

```text
Q4 cells: 56,175,137,076
chain length: 256
complete chains: 219,434,129
address bits: 36
record bits: 40
explicit pointer table: 261.5858664549887 GiB
pointer overhead over Q4-only metadata: 235.42727980948985 GiB
8 GiB raw-record cache capacity: 1,717,986,918
cached records per worst-chain bound: 7
minimum serial host misses: 249 / 256
host-miss fraction: 97.265625%
logical host bytes/token: 4.86328125
```

### Nonrepresentative prototype

The packed 64-bit host pointer chase executed one dependent probe per step:

```text
cells: 262,144
steps/repeat: 200,000
repeats: 5
median: 224.27377 ns/probe
minimum: 222.98837 ns/probe
maximum: 224.94671 ns/probe
```

This timing is CI-machine evidence only and must not be projected to a target CPU, pinned memory, PCIe, or GPU.

### Decisive interpretation

```text
near-one serial host miss/token: proven for explicit pointer records
one logical record probe/token is sufficient: demonstrated
logical data volume: only 4.86 bytes/token
arbitrary compressed host index lower bound: not proven
physical transaction granularity: not proven
target lookup latency: not measured
host-indexed escape: open
```

Therefore “one serial host probe per token proves impossibility” is rejected. Cell-probe count alone does not violate the 4B-class latency target.

## Current classification

```text
all-resident arbitrary exact-decision metadata in 8 GiB: contradicted for the constructed family
explicit host pointer representation: nearly one serial miss/token
bandwidth-only impossibility: not proven
latency impossibility: not proven
constructive host-indexed exact-decision runtime: not yet built
405B/8 GiB/4B-speed physical runtime: unsolved
```

## Prohibited repeats and overclaims

Do not:

- relabel metadata size as traffic;
- relabel serial probe count as latency;
- use the 224ns CI result as target hardware proof;
- claim arbitrary compressed indexes obey the raw-record cache theorem;
- claim a released checkpoint admits the constructed decision table;
- claim success without real 405B/8-GiB/quality/wall-clock evidence.

## Current frontier — Experiment 044 Host-Indexed Exact-Decision VM

The next work changes from impossibility search to constructive execution.

### Required prototype

Build a host-indexed decision virtual machine with:

```text
packed records
mmap-backed storage
explicit index format and versioning
resident hot cache
sequential and dependent-random traces
optional software prefetch for known addresses
checksums and exact replay validation
build time and output size
cold/warm lookup latency distributions
logical versus physical bytes
```

The first implementation must remain CPU-only and portable. It establishes the representation and measurement contract before GPU/pinned-memory integration.

### Experiment 044 Gates

1. Create `research/host-indexed-decision-vm`.
2. Add `docs/EXPERIMENT_044_HOST_INDEXED_DECISION_VM.md`.
3. Implement a packed fixed-width record format with exact Q4 token and next address.
4. Build files atomically and validate header, size, checksum, and deterministic replay.
5. Use `mmap` for cold and warm dependent pointer chasing.
6. Measure p50/p95/p99 lookup latency and total decode latency separately.
7. Compare sequential, random, dependent, cached, and prefetched access patterns.
8. Report page faults and OS-cache state only when measurable; do not invent them.
9. Keep CI timing explicitly nonrepresentative.
10. Project only storage and logical operations to the 405B target; do not project timing.
11. Decide whether the host-indexed VM is functionally viable enough to justify pinned-memory/GPU integration.
12. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.

## Correct communication

> Experiment 043 proved that an explicit pointer-table representation can force 249 serial host misses in 256 tokens under an 8 GiB raw-record cache, but only about 4.86 logical bytes/token are required. A nonrepresentative CPU prototype measured roughly 224ns per dependent probe. This does not prove target latency, but it rejects cell-probe count alone as an impossibility argument. The host-indexed path remains open, so Experiment 044 now builds the first fully charged mmap-backed exact-decision VM.
