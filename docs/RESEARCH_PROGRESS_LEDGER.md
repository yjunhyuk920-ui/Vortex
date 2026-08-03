# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable record of hypotheses, executable gates, measurements, accepted proof constraints, and rejection reasons. Every session must read it before creating another candidate.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is runtime completion.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Failed hypotheses remain permanent data.
- Accepted information proofs are guardrails, not working runtimes.
- Separate direct operators, operator collections, full Transformers, and hardware wall clock.
- Charge all checkpoint-specific metadata, construction, storage, transfer, lookup, and fallback.

## Foundational Gate 0

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse by compute: 246.889 tokens
```

This was a conditional E0/E1 envelope only. Quality, attention, universality, CUDA, physical traffic, and wall clock remain unproven.

## Rejected representation families

### Dictionaries, activation atlases, and entropy

- Exact gauge transform error reached about `4.6e-7`, but 16/32 prototype dictionaries produced zero useful causal continuation.
- A 16-prototype functional skeleton reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Static activation ranks 4/8/16 left continuation perpendicular means about 0.956/0.947/0.934.
- Online activation expansion required one exact expansion per token and projected 2.9355 GiB/token of LM-head residual traffic.
- ZIPTREE measured 11.3330 bits/weight and required 10,649 straight accepted tokens.

Decision: reject static dictionaries, activation-subspace caching, and whole-model lossless compression as the primary mechanism.

### Exact-neuron family — PR #29 and predecessors

Uniform optimistic oracle:

| Fraction | MLP traffic | Teacher top-32 | Exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

PR #29 measured 132 single-layer damage points; nonlinear allocation still produced zero useful prefix. Decision: close uniform, adjoint, and nonlinear independent-neuron selection.

### Signed residual proofs — PR #31–#34

| PR | Mechanism | Best decisive result | Decision |
|---:|---|---|---|
| 31 | global Signed Dual Cone | 8-bit mean refinement 97.93%, traffic 610.64 GiB/token | reject |
| 32 | partitioned residual norms | mean refinement about 96.46%, traffic about 607.6 GiB/token | reject |
| 33 | block signed residual code | mean refinement 92.39%, traffic 585.80 GiB/token | reject |
| 34 | global dual-price refinement | mean refinement 90.74%, max traffic 573.34 GiB/token | reject |

Signed cancellation was real, but static disjoint bases did not transfer and exact refinement remained dominant.

## Rejected dynamic and multi-token programs

### PR #36 — Semantic-State Program Routing

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best point `K=8/rank4`:

```text
mean reuse: 1.364 tokens
switch traffic: 0.7261 GiB/token
activation perpendicular mean: 42.28%
dual perpendicular mean: 53.91%
p95: 99.50% / 99.84%
```

Decision: reject precompiled semantic-state banks.

### PR #37 — Prompt-Compiled Hankel Decision Program

```text
head: 12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow: 30778715832
```

Rank32/control16/order2/full projected only 0.00674 GiB and 0.00822 GFLOP/token, but real autonomous exact prefixes were:

```text
algorithm-runtime: 1
distributed-database: 1
korean-plm-governance: 2
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

The oracle knew the exact target before repair. Decision: reject recurrence plus sparse exact repair.

### PR #40 — Nonlocal Exact Decision Memory

```text
head: 91b3e3f062d33087005ae38bbf94b357012f0ccd
corrected workflow: 30780847944
corrected CI: 30780847954
```

The first continuation token was charged as an exact boundary anchor. The initial misaligned run is invalid. Corrected post-anchor frontier:

| Prompt | Nearest max | Top-64 max | Future-aware global max | Required |
|---|---:|---:|---:|---:|
| algorithm-runtime | 74 | 75 | 75 | 247 |
| distributed-database | 27 | 28 | 28 | 247 |
| korean-plm-governance | 4 | 5 | 5 | 247 |

The global oracle searched every prompt suffix using future tokens. Decision: reject prompt-only exact suffix memory independent of retrieval implementation.

## Accepted proof guardrails

### PR #42 — Exact Dense-Operator Information Lower Bound

```text
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate workflow: 30781557141
full CI: 30781557096
main merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

#### Exact-output theorem

An arbitrary `N`-parameter `b`-bit checkpoint family has `2^(N b)` members. Any representation supporting every exact dense operator output must be injective and needs at least `N b` bits in the worst case.

405B Q4:

```text
exact information: 188.98828125 GiB
resident allowance: 8 GiB
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
ratio to 4B dense arithmetic: 101.462310912x
```

#### Skipped-coordinate top-1 adversary

```text
matrix shapes: 2x4, 3x5, 4x7, 8x8
coordinates tested: 115
passing indistinguishable-observation winner flips: 115
coverage: 100%
```

Accepted scope:

```text
exact-output N*b information bound: proven
any unrepresented coordinate can flip top-1: proven
metadata-aware complete top-1 N*b bound: not proven by PR #42
```

### PR #44 — Metadata-Aware Exact Top-1 Function Bound

```text
head: 95e202da8a31e564a80db509ad0b9b97bd71403d
certificate workflow: 30782192795
full CI: 30782192768
main merge: aca6657578b0decb58adbf98bcd22555169a6847
```

For an `m x d` dense classifier:

```text
p = min(floor(m/2), floor(d/2))
q = d - p
K = p q
```

The selector/payload construction produces `2^K` distinct complete top-1 decision functions. Arbitrary checkpoint-specific metadata for this family therefore needs at least `K` bits.

Exhaustive certificate:

| Shape | K | Expected functions | Observed | Margin |
|---|---:|---:|---:|---:|
| 2x2 | 1 | 2 | 2 | 1.0 |
| 4x4 | 4 | 16 | 16 | 1.0 |
| 4x5 | 6 | 64 | 64 | 1.0 |
| 6x6 | 9 | 512 | 512 | 1.0 |

Every bit table decoded exactly from the top-1 signature.

Llama-405B-shaped independently callable operator collection:

```text
one decoder layer: 653,787,136 bits = 77.9375 MiB
126-layer stack: 9.5899658203125 GiB
LM head: 8 MiB
total: 9.5977783203125 GiB
excess over 8 GiB: 1.5977783203125 GiB
```

Accepted scope:

```text
direct dense-classifier metadata bound: proven
independently callable operator-collection additive bound: proven
full end-to-end Transformer final-token bound: not proven
real 405B execution and wall clock: not performed
```

Do not report 9.5978 GiB as a full language-model theorem until layerwise bits are exposed through final token winners.

## Current classification

```text
arbitrary dense exact output with only 8 GiB information: contradicted
arbitrary coordinate omission for universal exact top-1: contradicted
metadata-aware direct classifier top-1 compression: lower-bounded
independently callable Llama-shaped operator collection: >8 GiB lower bound
full Transformer final-token metadata bound: open
405B/8 GiB/4B-speed runtime: unsolved
```

## Prohibited repeats

Do not continue by only changing rank, block size, state count, recurrence order, repair threshold, ANN settings, speculative block length, or uncharged lossless metadata. Do not conflate direct operators with final Transformer decisions.

## Current frontier — Experiment 042 End-to-End Llama Final-Decision Routing Bound

The next Gate must embed independent decision bits inside a legal Llama-like composition and expose them through final vocabulary top-1 decisions.

Required components:

```text
RMSNorm
causal self-attention
residual connections
SwiGLU MLP
final normalization
LM head
at least two layers with independent encoded bit tables
```

For each encoded bit, provide a legal token sequence whose final next-token winner reveals the bit with a positive margin. If `K_total` bits are independently exposed, the family contains `2^K_total` distinct end-to-end language-model decision functions and exact checkpoint metadata needs at least `K_total` bits.

### Promotion sequence

1. Build a deterministic small Llama-like micro-model.
2. Design selector/payload channels with known RMSNorm scale.
3. Route one layer's bit through attention or SwiGLU to final logits.
4. Prove two-layer additivity by exhaustive bit-table enumeration.
5. Verify unique final vocabulary winners and exact decoded signatures.
6. Only then derive symbolic scaling to 126 layers and target dimensions.
7. Keep lower-bound proof separate from runtime implementation and hardware measurement.

### Mandatory next step

1. Merge this documentation update after full CI.
2. Create `research/llama-final-decision-routing-bound`.
3. Add the Experiment 042 proof document and micro-model.
4. Add exhaustive tests, workflow, and raw JSON.
5. Update this ledger and `docs/SESSION_HANDOFF.md` before the next progress response.
