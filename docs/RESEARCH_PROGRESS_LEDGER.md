# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable index of hypotheses, executable Gates, measurements, accepted constraints, and rejection reasons. Experiment documents, PRs, workflows, raw JSON, and Git history remain authoritative.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is final physical runtime completion.

## Permanent rules

- Preserve negative evidence.
- Treat accepted proofs as guardrails, not runtime success.
- Separate metadata, logical bytes, physical transactions, latency, and wall clock.
- Charge model calls, build, coverage, storage, lookup, cache, and fallback.
- Bounded grammar coverage is not arbitrary-prompt coverage.
- Future-token oracle evidence is not causal routing.
- Before reporting meaningful work, update this ledger and `docs/SESSION_HANDOFF.md`.

## Rejected execution families

### Static compression and approximation

Progressive low precision, recurrent dictionaries, Kronecker approximations, gauge dictionaries, functional skeletons, activation-subspace caching, online expansion, and ZIPTREE all failed either original decisions or the traffic/amortization envelope. Exact-neuron subsets and signed residual refinements still required hundreds of GiB/token or nearly tokenwise exact work.

### PR #36 — Semantic-State Program Routing

```text
head/workflow: 499e5001c21d782adf79fba69ce6f2d445c0cb5e / 30778002226
best mean reuse: 1.364 tokens
switch traffic: 0.7261 GiB/token
```

Rejected.

### PR #37 — Prompt Hankel Program

```text
head/workflow: 12f859e4ec288f0d38b29d8b71e494bdc29f6586 / 30778715832
exact prefixes: 1, 1, 2
required: 247
```

Rejected.

### PR #38 — Perfect-Oracle Repair

```text
head/workflow: 13e3f60876199e4b06577ca51e9fd71f575cb134 / 30779062125
repairs: 174–229 / 256
traffic: 128.45–169.06 GiB/token
```

Rejected.

### PR #40 — Nonlocal Exact Memory

```text
head/corrected workflow: 91b3e3f062d33087005ae38bbf94b357012f0ccd / 30780847944
future-aware global maxima: 75, 28, 5
required: 247
```

The initial boundary-misaligned run is invalid. Corrected mechanism rejected.

## Accepted proof guardrails

### PR #42 — Dense-Operator Information Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate/full CI: 30781557141 / 30781557096
405B Q4 information: 188.98828125 GiB
minimum external information beyond 8 GiB: 180.98828125 GiB
```

115/115 skipped-coordinate adversaries changed exact output and top-1.

### PR #44 — Metadata-Aware Top-1 Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate/full CI: 30782192795 / 30782192768
independent Llama-shaped operator collection: 9.5977783203125 GiB
```

Direct/operator-collection scope only.

### PR #46 — End-to-End Llama Final-Decision Bound

```text
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate/full CI: 30784848053 / 30784848049
micro functions: 256 / 256
minimum margin: 0.24951063086132308
projected decision metadata: 26.158586645498872 GiB
```

All-resident 8 GiB representation contradicted for the constructed family. Sparse host access remained open. Pre-correction whole-target impossibility claims are invalid.

### PR #48 — Host Cell-Probe Gate

```text
head: 2705613f943f36adc041a6a4bedd7eba5c42f2ac
certificate/full CI: 30785452594 / 30785452584
serial misses: at least 249 / 256
logical bytes/token: 4.86328125
CI median: 224.27377 ns/probe
```

Decision: probe count alone does not prove latency failure.

## Accepted constructive implementations

### PR #50 — mmap Host Decision VM

```text
head: 8f029dde63984a3cf24f9ec2e9629c9e060d9352
certificate/full CI: 30785924201 / 30785924118
merge: a4a0e9b693184c9d5ea248822393998357df40db
```

Implemented compact40/aligned64, versioned header, CRCs, atomic build, mmap replay, corruption rejection, and LRU cache.

```text
compact40 dependent p50/p99: 1,473 / 1,806.5 ns
aligned64 dependent p50/p99: 1,502 / 1,833.45 ns
compact storage saving: about 37.4%
```

Compact40 is the default. CI timing is nonrepresentative.

### PR #52 — Bounded TinyLlama Compiler

```text
head: 5fb32b30ceda3e362da7b6ee9ed2dee0c93231e5
compiler/full CI: 30786618783 / 30786618729
merge: 657667095bd23271ba34e1b705aea587ba7e102e
```

```text
compiled prompts: 8
held-out prompts: 4
compiled calls: 64
trace collection: 25.302875082 s
codebook: 9 / 16
raw graph: 64 nodes
VM: 456 bytes
exact replay: 9/9 paths, 72/72 tokens
```

Implementation accepted.

## Rejected PR #52 raw-prefix scaling

| Horizon | Raw records | Unique exact-prefix nodes | Reuse |
|---:|---:|---:|---:|
| 2 | 16 | 16 | 0% |
| 4 | 32 | 32 | 0% |
| 8 | 64 | 64 | 0% |

Held-out exact-prefix coverage was 0/32 states with first miss at step zero. Raw complete-prefix memoization rejected as a general mechanism.

## Accepted PR #54 — Exact Future-Behavior DAG

```text
head: 199c84df2cb8b70086de27b63369294846a689aa
certificate/full CI: 30787236873 / 30787236839
merge: 3b415c753554c1d48b3856142109737bc8611d04
new model calls: 0
```

Exact quotient relation:

```text
signature = (exact next token, exact successor node)
```

Two states merge only when their complete remaining finite-horizon token suffixes are identical.

### Compression frontier for eight distinct prompts

| Horizon | Raw records | Quotient nodes | Reduction |
|---:|---:|---:|---:|
| 2 | 16 | 7 | 56.25% |
| 4 | 32 | 17 | 46.875% |
| 8 | 64 | 38 | 40.625% |

At horizon eight:

```text
complete-continuation classes: 5
cross-distinct-prompt suffix savings: 26
intentional duplicate incremental nodes: 0
DAG VM: 326 bytes
raw-prefix VM: 456 bytes
VM reduction: 28.5088%
exact replay: 9/9 paths, 72/72 tokens
```

Held-out evidence:

```text
future-aware suffix states present: 8/32 = 25%
full held-out continuations present: 0/4
causal start-router hits: 0/4
causal start coverage: 0%
```

Decision: accept exact suffix-DAG body compression. The 25% body figure uses future tokens and is not deployable. The causal exact start router is the dominant blocker.

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed VM: implemented
bounded real-model compiler: implemented
raw prefix graph: linear
exact future-DAG body: 40.625% smaller at horizon 8
held-out causal start routing: zero
certified start router/generalization: unsolved
GPU integration and real 405B target: unsolved
```

## Prohibited repeats and overclaims

Do not scale raw prefix enumeration, call duplicate reuse semantic generalization, use future tokens in deployable routing, merge approximate states without a certificate, omit model build calls, or project TinyLlama/CI results to 405B hardware.

## Current frontier — Experiment 047 Causal Abstaining Start Router

A router must choose a DAG start from causal prompt information and abstain unless correctness is certified.

### Mandatory Gate

1. Create `research/causal-abstaining-start-router`.
2. Reuse Experiment 045 prompt data, continuation classes, and Experiment 046 starts.
3. Implement exact prompt-ID routing as the sound baseline.
4. Test causal prompt-token feature routers and exact prefill-hidden empirical routers separately.
5. Report accuracy, accepted coverage, wrong accepts, abstention, fallback, feature bytes, prefill calls, and latency.
6. Keep empirical accuracy separate from sound certification.
7. A certificate may accept only under an explicit proof rule; bucket purity or held-out accuracy is not proof.
8. Require zero wrong certified accepts.
9. Do not use continuation tokens or future suffixes as inputs.
10. Determine whether any sound rule exceeds exact-ID held-out coverage.
11. If certified held-out coverage remains zero, record the start-router barrier and stop treating the decision-index path as a universal no-fallback runtime.
12. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.
