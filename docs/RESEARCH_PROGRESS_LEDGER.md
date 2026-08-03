# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable index of hypotheses, executable Gates, measurements, accepted constraints, and rejection reasons. Experiment documents, PRs, workflows, raw JSON, and Git history remain authoritative permanent evidence.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is final physical runtime completion.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Failed experiments remain permanent data.
- Accepted proofs are guardrails, not runtime success.
- Separate exact output, final decisions, metadata, logical bytes, physical transactions, and wall clock.
- Charge model calls, build, storage, lookup, transfer, cache, coverage, and fallback.
- Metadata size is not traffic; probe count is not latency.
- Bounded grammar coverage is not arbitrary-prompt coverage.
- Duplicate prompt reuse is not semantic generalization.
- Before reporting meaningful work, update this ledger and `docs/SESSION_HANDOFF.md`.

## Foundational envelope

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse: 246.889 tokens
```

This was only a conditional E0/E1 target envelope.

## Rejected model-execution representations

### Static dictionaries, factorization, activation atlases, and entropy

- Q2/Q3 progressive precision failed quality; Q4 speculation needed more than one thousand accepted tokens to amortize a full target stream.
- Recurrent layer dictionaries fit storage but still read about 188 GiB/committed token.
- Global and semantic Kronecker approximations produced near-unit error and zero useful causal prefix.
- Gauge dictionaries and functional skeletons passed primitives but failed full-model decisions.
- Static activation ranks left continuation perpendicular energy near one.
- Online expansion became exact/tokenwise.
- ZIPTREE measured 11.3330 bits/weight and required about 10,649 accepted tokens.

Decision: reject generic static compression and low-rank execution as primary mechanisms.

### Exact-neuron and signed-residual sequence

- Optimistic neuron subsets reached at most two exact tokens while traffic rose from about 0.623 to 12.285 GiB/token.
- PR #29 measured 132 single-layer damage points; nonlinear allocation still produced no useful prefix.
- PR #31 global Signed Dual Cone: about 97.93% refinement, 610.64 GiB/token.
- PR #32 partitioned residual bounds: about 96.46%, 607.6 GiB/token.
- PR #33 block signed residual code: 92.39%, 585.8 GiB/token.
- PR #34 global allocation: 90.74%, maximum 573.34 GiB/token.

Decision: exact refinement remains dominant.

### PR #36 — Semantic-State Program Routing

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best mean reuse 1.364 tokens; switch traffic 0.7261 GiB/token; perpendicular p95 near 100%. Rejected.

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
```

Initial boundary-misaligned evidence is invalid. Corrected future-aware global suffix maxima were 75, 28, and 5 versus required 247. Rejected independent of ANN configuration.

## Accepted proof guardrails

### PR #42 — Exact Dense-Operator Information Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate/full CI: 30781557141 / 30781557096
merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

```text
405B Q4 information: 188.98828125 GiB
resident allowance: 8 GiB
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
4B ratio: 101.462310912x
```

115/115 skipped-coordinate adversaries changed exact output and unique top-1.

### PR #44 — Metadata-Aware Top-1 Function Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate/full CI: 30782192795 / 30782192768
merge: aca6657578b0decb58adbf98bcd22555169a6847
```

Selector/payload families implemented `2^K` distinct functions for `K=p(d-p)`. Independently callable Llama-shaped operator collection lower bound: 9.5977783203125 GiB.

### PR #46 — End-to-End Llama Final-Decision Metadata Bound

```text
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate/full CI: 30784848053 / 30784848049
merge: 038d3fa72dbfe91f4d9837d482b9f9c10719a00f
```

```text
micro functions: 256 / 256
minimum winner margin: 0.24951063086132308
projected exact-decision metadata: 26.158586645498872 GiB
```

Scope: all-resident 8 GiB representation contradicted for the constructed family; sparse host lookup remained open. Pre-correction whole-target impossibility language is invalid.

### PR #48 — Host-Indexed Cell-Probe Gate

```text
head: 2705613f943f36adc041a6a4bedd7eba5c42f2ac
certificate/full CI: 30785452594 / 30785452584
merge: 4ca9d2c2d4876d1266b2ad5527e2350585c7db7c
```

```text
serial host misses: at least 249 / 256
logical host bytes/token: 4.86328125
explicit pointer table: 261.5858664549887 GiB
CI packed pointer median: 224.27377 ns/probe
```

Decision: reject probe count alone as an impossibility argument.

## Accepted constructive implementations

### PR #50 — mmap Host-Indexed Decision VM

```text
head: 8f029dde63984a3cf24f9ec2e9629c9e060d9352
certificate/full CI: 30785924201 / 30785924118
merge: a4a0e9b693184c9d5ea248822393998357df40db
```

Implemented compact40/aligned64 records, versioned header, CRCs, atomic build, mmap replay, strict corruption rejection, LRU caching, and benchmarks.

```text
compact40 dependent p50/p99: 1,473 / 1,806.5 ns
aligned64 dependent p50/p99: 1,502 / 1,833.45 ns
compact storage saving: about 37.4%
second replay: 256 hits / 0 mmap reads
```

Compact40 is the default. CI timing is nonrepresentative.

### PR #52 — Bounded TinyLlama Decision-Index Compiler

```text
head: 5fb32b30ceda3e362da7b6ee9ed2dee0c93231e5
compiler/full CI: 30786618783 / 30786618729
merge: 657667095bd23271ba34e1b705aea587ba7e102e
raw JSON: results/decision_index_compiler_gate.json
manifest: results/decision_index_compiler_manifest.json
```

Finite domain:

```text
unmodified TinyLlama
training: none
full grammar: 12 prompts
compiled A/B prompts: 8
held-out C prompts: 4
duplicate control: 1
horizons: 2, 4, 8
```

Build and replay:

```text
compiled model calls: 64
held-out ground-truth calls: 32
trace collection: 25.302875082 s on CI CPU
graph build: 1.073571 ms
VM export: 2.263952 ms
codebook: 9 / 16 tokens
VM: 456 bytes
manifest: 1,491 bytes
exact paths: 9 / 9
exact tokens: 72 / 72
```

Accepted implementation: original-checkpoint trace compiler and compact40 integration are real and exact for the declared domain.

## Rejected PR #52 scaling mechanism

Excluding the intentional duplicate:

| Horizon | Path records | Unique exact-prefix nodes | Reuse |
|---:|---:|---:|---:|
| 2 | 16 | 16 | 0% |
| 4 | 32 | 32 | 0% |
| 8 | 64 | 64 | 0% |

The duplicate control reused exactly its 2/4/8 nodes and shared the source start address.

Held-out C prompts:

```text
states: 32
hits: 0
fallbacks: 32
coverage: 0%
first miss: step 0 on every prompt
```

Decision: accept the bounded compiler implementation; reject complete token-prefix identity as a general execution mechanism. It is linear exact memoization with no measured compositional generalization.

## Current classification

```text
all-resident exact-decision metadata: contradicted for constructed family
host-indexed exact pointer VM: implemented
bounded real-model compiler: implemented
raw exact-prefix graph: linear growth on measured domain
held-out start routing: zero coverage
sound behavioral quotient: not yet tested
certified start router/generalization: unsolved
GPU integration and real 405B target: unsolved
```

## Prohibited repeats and overclaims

Do not:

- scale raw prefix enumeration and call it universal;
- treat exact duplicate reuse as semantic reuse;
- merge approximate states without a sound decision certificate;
- omit model calls and build coverage;
- project TinyLlama/CI measurements to 405B hardware;
- claim physical success or impossibility without real evidence.

## Current frontier — Experiment 046 Exact Future-Behavior DAG

The strongest sound finite-horizon quotient interns states backward by:

```text
signature = (exact next token, exact successor signature)
```

States merge only if their complete remaining exact decision suffixes are identical. This future-aware quotient is an optimistic compression upper bound for already compiled paths. It does not provide a deployable start router for unseen prompts.

### Mandatory work

1. Create `research/exact-future-behavior-dag`.
2. Add `docs/EXPERIMENT_046_EXACT_FUTURE_BEHAVIOR_DAG.md`.
3. Consume Experiment 045 traces without new model calls for the first Gate.
4. Build minimal suffix DAGs at horizons 2, 4, and 8.
5. Compare raw nodes, quotient nodes, start entries, VM bytes, and exact replay.
6. Attribute compression to identical suffixes, duplicate prompts, repeated tokens, and terminal merging.
7. Keep explicit prompt-to-start routing storage and held-out start coverage separate.
8. Export compact40 and replay all compiled paths exactly.
9. Decide whether graph-body compression is meaningful enough to justify a certified start router.
10. If body compression is high but held-out routing remains zero, record the start-router barrier as the primary blocker.
11. Commit tests, workflow, raw JSON, PR decision, ledger, and handoff.
