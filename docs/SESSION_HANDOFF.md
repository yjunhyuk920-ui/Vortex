# Session handoff

Last updated: 2026-08-03 (Asia/Seoul)

## Fixed objective

Build a universal runtime for arbitrary unmodified Hugging Face dense transformers with one 8 GiB GPU, no user training/fine-tuning/adapters, original-model decisions and quality, and p50 warm decode within 1.2x of a native 4B Q4 baseline. Final validation requires a real 405B-class checkpoint.

Current evidence remains below E4. Do not claim the final runtime is solved.

## Mandatory startup and persistence

Read `AGENTS.md`, `docs/PROOF_FIRST_CONTRACT.md`, `docs/WORK_SESSION_PROTOCOL.md`, `docs/RESEARCH_PROGRESS_LEDGER.md`, this file, and active experiment evidence. Before reporting meaningful work, update and commit the ledger and this handoff.

## Current repository state

Latest decisions:

```text
PR #36  semantic-state program routing                    rejected
PR #37  prompt-compiled Hankel program                    rejected
PR #38  perfect-oracle sparse repair                      rejected
PR #40  prompt-only nonlocal exact memory                 rejected
PR #42  exact dense-operator information bound            accepted/merged
PR #44  metadata-aware top-1 bound                        accepted/merged
PR #46  end-to-end Llama final-token metadata bound       accepted/merged
PR #48  host-indexed cell-probe Gate                      accepted/merged
PR #50  mmap host-indexed decision VM                     accepted/merged
PR #52  bounded TinyLlama compiler                        implementation accepted; raw prefix scaling rejected
PR #54  exact future-behavior DAG quotient                accepted/merged; start-router barrier isolated
```

Latest merge:

```text
PR #54 merge: 3b415c753554c1d48b3856142109737bc8611d04
```

Authoritative Experiment 046 evidence:

```text
branch: research/exact-future-behavior-dag
head: 199c84df2cb8b70086de27b63369294846a689aa
certificate workflow: 30787236873
full CI: 30787236839
source model calls reused: 64
new model calls: 0
raw JSON: results/exact_future_behavior_dag_gate.json
manifest: results/exact_future_behavior_dag_manifest.json
```

## Established runtime path

### End-to-end metadata and host lookup

Experiment 046's predecessors established:

```text
constructed Llama final-decision metadata: 26.158586645498872 GiB
all-resident 8 GiB path: contradicted for that family
explicit pointer logical bytes/token: 4.86328125
serial host misses: at least 249 / 256
compact40 mmap dependent p50/p99 on CI: 1,473 / 1,806.5 ns
CI timing target representative: false
```

Experiment 044 implemented compact40/aligned64, atomic build, CRCs, strict mmap reading, exact replay, and LRU caching. Compact40 is the v1 default.

### Bounded real-model compiler

Experiment 045 compiled eight unmodified TinyLlama grammar prompts plus one duplicate control:

```text
compiled model calls: 64
held-out ground-truth calls: 32
trace collection: 25.302875082 s on CI CPU
raw prefix graph: 64 nodes
VM: 456 bytes
exact paths/tokens: 9/9 and 72/72
held-out exact-prefix coverage: 0%
```

The compiler implementation is accepted. Raw complete-prefix identity is rejected as a scaling mechanism because eight distinct paths produced 64/64 unique nodes.

## Accepted Experiment 046 result

### Exact finite-horizon equivalence

Nodes are interned backward by:

```text
signature = (exact next token ID, exact successor node)
```

Two states merge only if their complete remaining token suffixes are identical. The quotient uses future tokens offline and is exact for compiled paths.

### Compression frontier for eight distinct prompts

| Horizon | Raw records | Quotient nodes | Reduction |
|---:|---:|---:|---:|
| 2 | 16 | 7 | 56.25% |
| 4 | 32 | 17 | 46.875% |
| 8 | 64 | 38 | 40.625% |

At horizon eight:

```text
complete-continuation classes: 5
cross-distinct-prompt suffix savings: 26 records
duplicate control raw records: 8
duplicate incremental nodes: 0
```

### VM replay

```text
DAG paths: 9
exact paths: 9 / 9
exact tokens: 72 / 72
mmap reads: 72
DAG VM: 326 bytes
raw-prefix VM: 456 bytes
VM reduction: 130 bytes = 28.5088%
```

### Held-out distinction

```text
future-aware held-out suffix states present: 8 / 32 = 25%
full held-out continuations present: 0 / 4
causal held-out start-router hits: 0 / 4
causal start coverage: 0%
```

The 25% figure is a future-token oracle and is not deployable. No held-out prompt had a sound start address.

Decision:

> Accept exact future-suffix DAG compression. The graph body is compactable, but the dominant unresolved blocker is a causal exact-certified prompt-to-start router.

## Critical scope boundary

```text
finite-horizon suffix equivalence: proven
compiled graph replay: proven
body compression: proven on eight TinyLlama paths
future-aware held-out body coverage: evaluation only
causal held-out routing: absent
arbitrary prompt coverage: not proven
405B compiler and target hardware: not tested
```

Do not convert future suffix existence into causal routing or project the TinyLlama compression ratio to 405B.

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed VM: implemented
bounded real-model compiler: implemented
raw prefix graph: linear
exact future-DAG body: 40.625% smaller at horizon 8
held-out causal start coverage: 0%
causal certified start router: primary blocker
405B/8 GiB/quality/wall-clock target: unsolved
```

## Prohibited overclaims

Do not scale raw prefix enumeration, call duplicate reuse semantic generalization, merge approximate states without a certificate, use future tokens in a deployable router, omit model build calls, or project TinyLlama/CI evidence to 405B hardware.

## Current frontier — Experiment 047 Causal Abstaining Start Router

The next Gate must choose a compiled DAG start from information available before future generation and must abstain when it cannot certify correctness.

Required separation:

```text
empirical router accuracy
accepted coverage
wrong-route count
sound-certificate coverage
abstention/fallback
cost of exact prefill features
```

### Mandatory work

1. Create `research/causal-abstaining-start-router`.
2. Add `docs/EXPERIMENT_047_CAUSAL_ABSTAINING_START_ROUTER.md`.
3. Reuse Experiment 045 prompts, continuation-class labels, and DAG starts.
4. Implement an exact prompt-ID router as the sound baseline; it must accept compiled prompts and abstain on held-out prompts.
5. Test causal feature routers using prompt token features and, separately, exact TinyLlama prefill final-hidden features.
6. Keep learned/nearest empirical predictions separate from certified accepts.
7. A certified accept is allowed only under an explicit proof rule; training purity or held-out accuracy alone is not a proof.
8. Report compiled and held-out accuracy, accepted coverage, wrong accepts, abstention, prefill model calls, feature bytes, and router latency.
9. Require zero wrong certified accepts. Any nonzero wrong accept rejects that certificate rule.
10. Do not use continuation tokens or future-aware DAG suffixes as router inputs.
11. Decide whether any sound rule exceeds exact-ID coverage on held-out prompts.
12. If sound held-out coverage remains zero, record the start-router barrier and stop treating the decision-index path as a universal runtime without fallback.
13. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.

## Correct communication

> Experiment 046 reduced the eight-path TinyLlama graph from 64 raw records to 38 exact future-DAG nodes and replayed all 72 tokens exactly. But no held-out prompt had a causal start address; the only 25% held-out body coverage used future tokens. The next Gate tests causal abstaining routers and requires zero wrong certified accepts. Graph compression succeeded; exact start selection is now the primary blocker.
