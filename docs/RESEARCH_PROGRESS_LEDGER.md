# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological index of hypotheses, executable Gates, measured evidence, accepted proof constraints, and rejection reasons. Detailed experiment documents, PR discussions, workflows, and raw JSON remain permanent authoritative data even when this ledger summarizes them.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is a completed physical runtime.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Failed hypotheses remain permanent project data.
- Accepted information proofs are guardrails, not runtime implementations.
- Separate exact output, top-1 functions, final Transformer decisions, total metadata, per-token traffic, and hardware wall clock.
- Charge checkpoint-specific construction, storage, host state, lookup, transfer, validation, and fallback.
- Metadata size must never be relabeled as per-token traffic.
- Before a user-facing progress/completion response after meaningful work, update this ledger and `docs/SESSION_HANDOFF.md`.

## Foundational Gate 0 envelope

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse by compute: 246.889 tokens
```

This was only a conditional E0/E1 envelope. Quality, attention, universality, CUDA scheduling, physical bytes, and wall clock were not proven.

## Rejected representation families

### Dictionaries, activation atlases, entropy, and exact-neuron subsets

- Exact gauge transformation error reached about `4.6e-7`, but 16/32 prototype dictionaries produced zero useful causal continuation.
- A 16-prototype functional skeleton reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Static activation ranks 4/8/16 left continuation perpendicular means about 0.956/0.947/0.934.
- Online activation expansion required one exact expansion per token and projected 2.9355 GiB/token of LM-head residual traffic.
- ZIPTREE measured 11.3330 bits/weight and required 10,649 straight accepted tokens.
- Exact-neuron optimistic subsets reached at most two exact tokens while traffic rose from about 0.623 to 12.285 GiB/token.
- PR #29 measured 132 single-layer damage points; nonlinear allocation still yielded zero useful prefix.

Decision: reject static dictionaries, activation-subspace caching, ordinary lossless streaming, and independent-neuron selection as primary mechanisms.

### Signed residual proof sequence — PR #31–#34

| PR | Mechanism | Decisive result | Decision |
|---:|---|---|---|
| 31 | global Signed Dual Cone | 8-bit mean refinement 97.93%; 610.64 GiB/token | reject |
| 32 | partitioned residual bounds | mean refinement about 96.46%; about 607.6 GiB/token | reject |
| 33 | block signed residual code | mean refinement 92.39%; 585.80 GiB/token | reject |
| 34 | global dual-price allocation | mean refinement 90.74%; max 573.34 GiB/token | reject |

Signed cancellation was real, but static bases did not transfer and exact refinement remained dominant.

## Rejected dynamic and multi-token programs

### PR #36 — Semantic-State Program Routing

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best point:

```text
mean program reuse: 1.364 tokens
switch traffic: 0.7261 GiB/token
activation perpendicular mean: 42.28%
dual perpendicular mean: 53.91%
p95: 99.50% / 99.84%
```

Decision: reject precompiled semantic-state program banks.

### PR #37 — Prompt-Compiled Hankel Decision Program

```text
head: 12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow: 30778715832
```

The small projected program failed real autonomous continuation:

```text
algorithm-runtime exact prefix: 1
distributed-database exact prefix: 1
korean-plm-governance exact prefix: 2
required: 247
```

Decision: reject prompt-only linear/quadratic/bilinear/full-lift recurrence.

### PR #38 — Perfect-Oracle Sparse Repair

```text
head: 13e3f60876199e4b06577ca51e9fd71f575cb134
workflow: 30779062125
```

| Prompt | Repairs / 256 | Mean interval | Repair traffic | Repair compute |
|---|---:|---:|---:|---:|
| algorithm-runtime | 226 | 1.133 | 166.84 GiB/token | 716.58 GFLOP/token |
| distributed-database | 229 | 1.118 | 169.06 GiB/token | 726.09 GFLOP/token |
| korean-plm-governance | 174 | 1.471 | 128.45 GiB/token | 551.70 GFLOP/token |

The oracle knew the exact target before deciding to repair. Decision: reject recurrence plus sparse exact repair and every weaker causal detector.

### PR #40 — Nonlocal Exact Decision Memory

```text
head: 91b3e3f062d33087005ae38bbf94b357012f0ccd
corrected workflow: 30780847944
corrected CI: 30780847954
```

The first continuation token was charged as the exact boundary anchor. The initial misaligned run is invalid. Corrected post-anchor results:

| Prompt | Nearest max | Top-64 max | Future-aware global max | Required |
|---|---:|---:|---:|---:|
| algorithm-runtime | 74 | 75 | 75 | 247 |
| distributed-database | 27 | 28 | 28 | 247 |
| korean-plm-governance | 4 | 5 | 5 | 247 |

The impossible global oracle searched every prompt suffix using future tokens. Decision: reject prompt-only exact suffix memory independent of ANN, key rank, distance, or router.

## Accepted proof guardrails

### PR #42 — Exact Dense-Operator Information Lower Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate workflow: 30781557141
full CI: 30781557096
main merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

For an arbitrary `N`-parameter `b`-bit checkpoint family, every representation supporting all exact dense operator outputs must distinguish `2^(N b)` checkpoints and needs at least `N b` bits in the worst case.

405B Q4:

```text
exact information: 188.98828125 GiB
resident allowance: 8 GiB
minimum external exact information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
ratio to 4B dense arithmetic: 101.462310912x
```

Skipped-coordinate adversary:

```text
matrix shapes: 2x4, 3x5, 4x7, 8x8
coordinates tested: 115
winner-flip adversaries: 115
```

Accepted scope:

```text
exact-output N*b information bound: proven
any omitted/unrepresented coordinate can flip top-1: proven
metadata-aware complete top-1 N*b theorem: not proven by PR #42
```

### PR #44 — Metadata-Aware Exact Top-1 Function Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate workflow: 30782192795
full CI: 30782192768
main merge: aca6657578b0decb58adbf98bcd22555169a6847
```

For an `m x d` classifier:

```text
p = min(floor(m/2), floor(d/2))
K = p(d-p)
```

The selector/payload family implements `2^K` distinct complete top-1 functions, so arbitrary exact checkpoint metadata for that family needs at least `K` bits.

Exhaustive certificate:

| Shape | Decision bits | Expected | Observed | Margin |
|---|---:|---:|---:|---:|
| 2x2 | 1 | 2 | 2 | 1.0 |
| 4x4 | 4 | 16 | 16 | 1.0 |
| 4x5 | 6 | 64 | 64 | 1.0 |
| 6x6 | 9 | 512 | 512 | 1.0 |

Independently callable Llama-shaped operator collection:

```text
one decoder layer: 77.9375 MiB
126 layers: 9.5899658203125 GiB
LM head: 8 MiB
total: 9.5977783203125 GiB
```

Accepted scope:

```text
direct classifier metadata theorem: proven
independently callable operator collection: proven
end-to-end final-token composition: not proven by PR #44
```

### PR #46 — End-to-End Llama Final-Decision Metadata Bound

```text
head: 7f1385b2585477d1557f50823047e41604803cb0
certificate workflow: 30784848053
full CI: 30784848049
main merge: 038d3fa72dbfe91f4d9837d482b9f9c10719a00f
raw evidence: results/llama_final_decision_routing_bound.json
```

The executable family uses standard bias-free Llama-style components:

```text
token embeddings
RMSNorm
causal grouped-query self-attention
residual connections
SwiGLU
final RMSNorm
linear LM head
```

A legal four-token prompt selects a signed Q4 `up_proj` coefficient. The final next-token winner decodes its exact code.

Micro-certificate:

```text
loader layers: 2
variable layers: 2
independent Q4 coefficients: 2
expected functions: 256
observed functions: 256
exact code recovery: true
minimum winner margin: 0.24951063086132308
```

Llama-405B-shaped projection, including the 1,024-dimensional GQA KV bottleneck:

```text
loader layers: 15
variable layers: 111
groups/layer: 31
neurons/group: 1,717
payload coordinates: 9,508
active intermediate/layer: 53,227
control coordinates: 14,666
vocabulary rows: 42,139
independent Q4 coefficients: 56,175,137,076
metadata bits: 224,700,548,304
metadata: 26.158586645498872 GiB
resident allowance: 8 GiB
```

Accepted theorem:

> A complete exact final-decision representation for the constructed arbitrary Q4 Llama-style family cannot fit entirely in an 8 GiB resident checkpoint-information allowance.

Critical scope:

```text
end-to-end final-token metadata-size bound: proven for the family
all-resident 8 GiB representation: contradicted for the family
per-query external traffic lower bound: not proven
host-indexed sparse random-access escape: open
fixed physical runtime target fully contradicted: false
released 405B maximum information complexity: not proven
real 405B execution and wall clock: not performed
```

The initial pre-correction PR comment that described the whole target as contradicted is invalid. Use the raw JSON at the authoritative head.

## Current classification

```text
arbitrary dense exact output using only 8 GiB total information: contradicted
arbitrary coordinate omission for universal exact top-1: contradicted
direct classifier exact top-1 metadata: lower-bounded
independent Llama-shaped operator collection: >8 GiB total metadata bound
constructed end-to-end Llama final-decision family: 26.1586 GiB total metadata bound
all-resident exact-decision representation in 8 GiB: contradicted for that family
host-indexed exact-decision representation: open
per-token probe, communication, and latency lower bound: open
405B/8 GiB/4B-speed physical runtime: unsolved
```

## Prohibited repeats and overclaims

Do not continue by only changing static rank, block size, state count, recurrence order, repair threshold, ANN parameters, speculative block length, or uncharged metadata. Do not:

- report total metadata as per-token traffic;
- cite invalid pre-correction Experiment 039 or 042 results;
- infer released-checkpoint complexity from a worst-case family;
- claim physical impossibility without a communication/latency theorem;
- claim runtime success without real 405B/8-GiB/quality/wall-clock evidence.

## Current frontier — Experiment 043 Host-Indexed Exact-Decision Cell-Probe Gate

Experiment 042 closes the all-resident path but leaves external sparse lookup open. The next Gate must address adaptive host access without assuming the result.

Construct an exact pointer-chasing family:

```text
value_t = table[address_t]
address_(t+1) = transition[address_t, value_t]
```

The next address is unknown until the current exact value returns. This can prove serial adaptivity and prevent parallel look-ahead for the constructed trace.

Required accounting:

```text
table entries and bits
address bits
probes/token
bytes/probe
serial dependency depth
resident cache size
host/PCIe transfers
lookup construction cost
fallback path
```

Strict scope:

```text
serial probes are an algorithmic result
latency is hardware evidence
one small probe/token may still meet the target
CI timing is not target-GPU timing
```

### Mandatory Experiment 043 work

1. Create `research/host-indexed-cell-probe-gate` from updated `main`.
2. Add `docs/EXPERIMENT_043_HOST_INDEXED_CELL_PROBE_GATE.md`.
3. Implement adaptive Q4 pointer-chasing decision tables.
4. Construct indistinguishable table pairs showing that skipping the addressed cell can change the token and all later addresses.
5. Verify exact serial dependency over multiple steps.
6. Test bounded resident caches and report hit/miss/probe counts.
7. Implement a host-memory lookup prototype; keep functional counts separate from nonrepresentative timing.
8. Derive equations before projecting to 405B.
9. Commit tests, workflow, raw JSON, PR decision, this ledger, and `docs/SESSION_HANDOFF.md`.
