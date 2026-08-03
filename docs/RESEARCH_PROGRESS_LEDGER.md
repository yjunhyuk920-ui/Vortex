# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological index of hypotheses, executable Gates, measurements, accepted proof constraints, and rejection reasons. Detailed experiment documents, PRs, workflows, raw JSON, and Git history remain authoritative permanent data.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is physical runtime completion.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Negative experiments are permanent data.
- Accepted proofs are guardrails, not working runtimes.
- Keep exact output, top-1 functions, final Transformer decisions, metadata size, logical bytes, physical transactions, and wall clock separate.
- Charge build, storage, host state, lookup, transfer, validation, cache, and fallback.
- Metadata size is not per-token traffic.
- Serial probe count is not latency.
- Before reporting meaningful progress, update this ledger and `docs/SESSION_HANDOFF.md`.

## Foundational Gate 0 envelope

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse by compute: 246.889 tokens
```

Conditional E0/E1 only. Quality, attention, universality, physical bytes, CUDA, and wall clock were unproven.

## Rejected representation families

### Dictionaries, factorization, activation atlases, and entropy

- Q2/Q3 progressive precision failed quality; Q4 causal tree preserved exact path for 12 levels but needed about 1,057–1,232 accepted tokens to amortize the full stream.
- Recurrent layer dictionaries fit storage but still read about 188 GiB/committed token.
- Global/semantic Kronecker approximations passed some memory projections but produced near-unit weight error and zero useful causal prefix.
- Exact gauge dictionaries and functional skeletons preserved primitives but failed full-model decisions.
- Static activation ranks left continuation perpendicular energy near one.
- Online expansion became exact/tokenwise and exceeded the traffic envelope.
- ZIPTREE measured 11.3330 bits/weight and required about 10,649 accepted tokens.

Decision: reject static low-rank dictionaries, global factorization, activation-subspace caching, and ordinary whole-model lossless streaming as primary mechanisms.

### Exact-neuron and signed-residual families

- Exact-neuron optimistic subsets reached at most two exact tokens as MLP traffic rose from about 0.623 to 12.285 GiB/token.
- PR #29 measured 132 single-layer damage points; nonlinear allocation still produced no useful prefix.
- PR #31 global Signed Dual Cone: 8-bit mean refinement 97.93%, about 610.64 GiB/token.
- PR #32 partitioned residual bounds: about 96.46%, about 607.6 GiB/token.
- PR #33 block signed residual code: 92.39%, about 585.8 GiB/token.
- PR #34 global dual-price allocation: 90.74%, maximum about 573.34 GiB/token.

Decision: signed cancellation exists, but exact refinement remains dominant. Close these families.

## Rejected dynamic and prompt-derived execution programs

### PR #36 — Semantic-State Program Routing

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best reuse 1.364 tokens; switch traffic 0.7261 GiB/token; activation/dual perpendicular p95 near 100%. Rejected.

### PR #37 — Prompt-Compiled Hankel Decision Program

```text
head: 12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow: 30778715832
```

Autonomous exact prefixes were 1, 1, and 2 versus required 247. Rejected.

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

The initial boundary-misaligned run is invalid. Corrected future-aware global suffix maxima were 75, 28, and 5 versus required 247. Rejected independent of key rank, ANN, top-k, or distance.

## Accepted proof guardrails

### PR #42 — Exact Dense-Operator Information Lower Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate: 30781557141
full CI: 30781557096
merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

For arbitrary `N`-parameter `b`-bit exact operators, a lossless universal representation needs at least `N*b` bits.

```text
405B Q4 information: 188.98828125 GiB
8 GiB resident fraction: 4.2331%
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
4B arithmetic ratio: 101.462310912x
```

115/115 skipped-coordinate adversaries changed exact output and unique top-1. Scope: exact-output theorem and coordinate relevance, not a complete metadata-aware top-1 theorem.

### PR #44 — Metadata-Aware Exact Top-1 Function Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate: 30782192795
full CI: 30782192768
merge: aca6657578b0decb58adbf98bcd22555169a6847
```

Selector/payload classifier family:

```text
p = min(floor(m/2), floor(d/2))
K = p(d-p)
functions = 2^K
```

2, 16, 64, and 512 function families were exhaustively injective. Independently callable Llama-shaped operator collection lower bound: 9.5977783203125 GiB. Scope: direct/operator collection only.

### PR #46 — End-to-End Llama Final-Decision Metadata Bound

```text
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate: 30784848053
full CI: 30784848049
merge: 038d3fa72dbfe91f4d9837d482b9f9c10719a00f
raw JSON: results/llama_final_decision_routing_bound.json
```

Actual bias-free Llama-style family with embeddings, RMSNorm, causal GQA, residuals, SwiGLU, final RMSNorm, and LM head:

```text
micro functions: 256 / 256
exact signed-Q4 recovery: true
minimum winner margin: 0.24951063086132308
```

405B-shaped projection:

```text
loader / variable layers: 15 / 111
groups/layer: 31
neurons/group: 1,717
payload coordinates: 9,508
independent Q4 coefficients: 56,175,137,076
complete final-decision metadata: 26.158586645498872 GiB
```

Accepted scope:

```text
all-resident 8 GiB complete exact-decision representation: contradicted for the family
per-token external traffic: not proven
host-indexed sparse lookup: open
full physical target: unsolved
```

Pre-correction Experiment 042 comments that claimed the whole target was contradicted are invalid.

### PR #48 — Host-Indexed Exact-Decision Cell-Probe Gate

```text
head: 2705613f943f36adc041a6a4bedd7eba5c42f2ac
certificate: 30785452594
full CI: 30785452584
merge: 4ca9d2c2d4876d1266b2ad5527e2350585c7db7c
raw JSON: results/host_indexed_cell_probe_gate.json
```

Explicit pointer model:

```text
record[address] = (q4_value, next_address)
next_token_t = q4_value
address_(t+1) = next_address
```

For `S` disjoint chains of length `T` and `C` cached complete records:

```text
worst-chain host misses >= T - floor(C/S)
```

Balanced caches attained the bound; sampled caches never violated it. Three one-record adversaries changed the current token and next address after identical prefixes.

Target projection:

```text
Q4 cells: 56,175,137,076
chain length: 256
complete chains: 219,434,129
address / record bits: 36 / 40
explicit pointer table: 261.5858664549887 GiB
pointer overhead: 235.42727980948985 GiB
8 GiB raw-record cache capacity: 1,717,986,918
worst-chain cached records floor: 7
serial host misses: at least 249 / 256
logical host bytes/token: 4.86328125
```

Nonrepresentative packed CPU pointer chase:

```text
median: 224.27377 ns/probe
minimum: 222.98837 ns/probe
maximum: 224.94671 ns/probe
```

Accepted scope and architecture decision:

```text
explicit pointer serial dependency: proven
raw complete-record cache theorem: proven
one logical record probe/token is sufficient for the model: demonstrated
arbitrary compressed cache theorem: not proven
physical transfer size and target latency: not proven
host-indexed escape: open
```

Decision: reject “one serial host probe/token proves impossibility.” Advance to a constructive host-indexed VM.

## Current classification

```text
all-resident exact-decision metadata in 8 GiB: contradicted for the constructed family
explicit pointer host representation: nearly one serial miss/token
logical bytes/token lower bound: small
bandwidth impossibility: not proven
latency impossibility: not proven
host-indexed exact-decision VM: not yet built
real 405B/8 GiB/quality/wall-clock target: unsolved
```

## Prohibited repeats and overclaims

Do not:

- relabel metadata size as traffic;
- relabel serial probe count as latency;
- project CI timing to target hardware;
- apply the raw-record cache theorem to arbitrary compressed indexes;
- infer released-checkpoint complexity from worst-case constructed families;
- claim physical success or impossibility without target-relevant evidence.

## Current frontier — Experiment 044 Host-Indexed Exact-Decision VM

The next work is constructive.

Required first VM:

```text
portable CPU implementation
packed fixed-width records
atomic file builder
mmap-backed reader
versioned header
size and checksum validation
exact dependent replay
hot resident cache
sequential/random/dependent traces
cold/warm latency distributions
build time and file size
logical versus physical I/O separation
```

### Mandatory Experiment 044 work

1. Create `research/host-indexed-decision-vm` from updated `main`.
2. Add `docs/EXPERIMENT_044_HOST_INDEXED_DECISION_VM.md`.
3. Define and test the on-disk binary format.
4. Build files atomically and reject truncation, bad magic, wrong version, bad checksum, and out-of-range pointers.
5. Implement mmap lookup and exact pointer replay.
6. Add bounded LRU/hot cache and measure hit/miss behavior.
7. Benchmark sequential, shuffled random, dependent, warm, and reopened-file paths.
8. Record p50/p95/p99, total decode time, build time, file bytes, logical bytes, and records/second.
9. Mark CI timing nonrepresentative.
10. Project storage and logical operations only; do not project latency to 405B hardware.
11. Decide whether functional and CPU prototype evidence justifies pinned-memory/GPU integration.
12. Commit tests, workflow, raw JSON, PR decision, this ledger, and handoff.
