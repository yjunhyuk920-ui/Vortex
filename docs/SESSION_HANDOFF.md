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
PR #52  bounded TinyLlama decision-index compiler         implementation accepted; raw prefix scaling rejected
```

Latest main merge:

```text
PR #52 merge: 657667095bd23271ba34e1b705aea587ba7e102e
```

Authoritative Experiment 045 evidence:

```text
branch: research/decision-index-compiler-gate
head: 5fb32b30ceda3e362da7b6ee9ed2dee0c93231e5
compiler workflow: 30786618783
Python 3.10/3.12 CI + validation: 30786618729
raw JSON: results/decision_index_compiler_gate.json
manifest: results/decision_index_compiler_manifest.json
VM artifact: decision_index_compiler_compact40.vtx
```

## Accepted representation and lookup path

### Experiment 042 — end-to-end final-token metadata bound

An actual bias-free Llama-style family exposed signed Q4 coefficients through final next-token winners.

```text
micro functions: 256 / 256
minimum winner margin: 0.24951063086132308
projected complete final-decision metadata: 26.158586645498872 GiB
```

This closed an all-resident 8 GiB exact-decision representation for the constructed family, not sparse host lookup.

### Experiment 043 — explicit host cell probes

```text
serial host misses: at least 249 / 256 tokens
logical host bytes/token: 4.86328125
explicit pointer host storage: 261.5858664549887 GiB
nonrepresentative packed CPU median: 224.27377 ns/probe
```

Decision: one serial host probe/token does not itself prove target failure.

### Experiment 044 — mmap exact-decision VM

Implemented compact40/aligned64 files, atomic build, checksums, strict mmap reading, exact pointer replay, LRU caching, and access benchmarks.

```text
compact40 dependent p50 / p99: 1,473 / 1,806.5 ns
aligned64 dependent p50 / p99: 1,502 / 1,833.45 ns
compact40 storage saving: about 37.4%
second 256-token replay: 256 cache hits / 0 mmap reads
```

Compact40 is the v1 default. CI timing is not target evidence.

## Accepted Experiment 045 implementation

### Finite grammar

```text
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
training or fine-tuning: none
symbols: A, B, C
counts: 8, 12
templates: 2
full grammar combinations: 12
compiled A/B combinations: 8
held-out C combinations: 4
duplicate exact prompt control: 1
horizons: 2, 4, 8
```

The sound v1 state key was:

```text
exact chat prompt token IDs || boundary marker || exact generated prefix IDs
```

No hidden-state similarity, semantic routing, or approximate merge was used.

### Build accounting

```text
compiled model forward calls: 64
held-out ground-truth forward calls: 32
duplicate-control forward calls: 0
trace collection: 25.302875082 seconds on CI CPU
graph build: 1.073571 ms
compact40 export: 2.263952 ms
VM file: 456 bytes
manifest: 1,491 bytes
```

The compiled continuations contained nine distinct TinyLlama token IDs, fitting the sixteen-entry compact40 codebook.

### Exact replay

```text
graph nodes: 64
VM starts: 9
compiled and duplicate paths: 9
exact paths without model: 9 / 9
exact tokens: 72 / 72
mmap record reads: 72
```

The duplicate control shared the same start address as its source and reused every 2/4/8-node path at the corresponding horizon.

## Rejected Experiment 045 scaling mechanism

Excluding the intentional duplicate:

```text
horizon 2: 16 path records, 16 unique nodes, 0% reuse
horizon 4: 32 path records, 32 unique nodes, 0% reuse
horizon 8: 64 path records, 64 unique nodes, 0% reuse
```

Held-out C combinations:

```text
state denominator: 32
compiled hits: 0
fallback tokens: 32
coverage: 0%
first miss: position 0 on all four prompts
```

Decision:

> Accept the bounded real-checkpoint compiler and VM integration. Reject full-token-prefix identity as a broad execution mechanism. It is exact memoization with linear state growth and no held-out compositional coverage.

The principal build cost was original-model transition generation, not graph construction or VM lookup.

## Critical scope boundary

```text
bounded grammar completeness: proven
compiled replay without model: proven
exact duplicate reuse: proven
nontrivial reuse among distinct prompts: absent
held-out grammar generalization: absent
arbitrary prompt coverage: not proven
405B index construction: not performed
GPU integration and target wall clock: not proven
```

Do not report bounded grammar coverage as universal coverage or duplicate prompt reuse as semantic generalization.

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed VM: implemented and exact for explicit records
bounded real-model index compiler: implemented
raw exact-prefix state space: linear on measured grammar/horizon
held-out start routing: 0% coverage
sound behavior quotient and general start router: unsolved
405B/8 GiB/quality/wall-clock target: unsolved
```

## Prohibited repeats and overclaims

Do not:

- scale raw prefix enumeration and call it a universal compiler;
- merge approximate hidden states without a sound exact-decision certificate;
- treat the 456-byte bounded VM as evidence for arbitrary prompts;
- omit the 64 original-model calls needed for compiled transitions;
- project TinyLlama CI build time to 405B;
- claim model quality or hardware success outside the grammar.

## Current frontier — Experiment 046 Exact Future-Behavior DAG Quotient

Experiment 045 showed that exact state identity does not deduplicate distinct prompt paths. The next Gate separates graph-body compressibility from the start-router barrier.

### Candidate

Given exact compiled continuations, build the minimal deterministic acyclic decision graph by interning nodes backward on:

```text
node_signature = (exact next token ID, exact successor node address)
```

Two compiled states merge only when their complete remaining exact token suffixes are identical. This is a sound finite-horizon behavioral quotient and requires no approximate state assumption.

### Required Experiment 046 work

1. Create `research/exact-future-behavior-dag` from updated `main`.
2. Add `docs/EXPERIMENT_046_EXACT_FUTURE_BEHAVIOR_DAG.md`.
3. Consume authoritative Experiment 045 traces without new model calls for the first Gate.
4. Build minimal suffix DAGs at horizons 2, 4, and 8.
5. Compare raw prefix nodes, quotient nodes, start entries, VM bytes, and exact replay.
6. Keep each prompt's start address explicit; do not hide start-router storage.
7. Evaluate held-out prompts separately: a compressed graph body does not supply a sound start address.
8. Report how much compression comes from identical future strings, exact duplicate prompts, repeated single tokens, and terminal suffixes.
9. Export the quotient to compact40 and replay all compiled paths exactly.
10. Decide whether graph-body compression is meaningful enough to pursue a certified start router.
11. If body compression is high but held-out routing remains zero, identify the start-router/generalization barrier as the primary blocker.
12. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.

## Correct communication

> Experiment 045 successfully compiled eight TinyLlama grammar paths into compact40 and replayed all 72 checked tokens without the model. However, distinct prompts produced 64 path records and 64 unique exact-prefix nodes, while all four held-out compositions missed at position zero. The bounded compiler is real, but raw prefix memoization does not scale. Experiment 046 now tests the strongest sound finite-horizon graph quotient—merging only states with identical complete future decision suffixes—and isolates the unresolved start-router barrier.
