# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological record of architecture hypotheses, executable gates, measured evidence, and rejection reasons. Every session must read it before creating another candidate.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing in this file is target completion.

## Permanent rules

- `docs/WORK_SESSION_PROTOCOL.md` is mandatory.
- Failed hypotheses remain permanent project data.
- Separate measured evidence from 405B projections.
- Do not describe a lower-bound proof as a working runtime.
- Do not hide exact weight information in uncharged metadata.

## Foundational Gate 0 budget

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse by compute: 246.889 tokens
```

This remains only a conditional E0/E1 envelope. Quality, universality, attention, CUDA scheduling, physical bytes, and wall clock are unproven.

## Early representation rejections

### Dictionaries, activation atlases, and entropy

- Exact gauge transformation error reached about `4.6e-7`, but 16/32 prototype dictionaries produced teacher top-32 and causal prefix zero.
- A 16-prototype functional skeleton reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Static prompt activation ranks 4/8/16 left continuation perpendicular means about 0.956/0.947/0.934 and zero certificate rate.
- Online activation expansion required 32 exact expansions for 32 tokens and projected 2.9355 GiB/token of LM-head residual traffic.
- ZIPTREE measured 11.3330 bits/weight and required a 10,649-token straight accepted run.

Decision: reject static dictionaries, activation-subspace caching, and whole-model lossless compression as the primary execution mechanism.

### Exact-neuron heavy hitters — PR #29 and predecessors

Uniform optimistic oracle:

| Fraction | Projected MLP traffic | Teacher top-32 | Exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

PR #29 measured 132 single-layer damage points. Nonlinear allocation reached only 18.75% top-32 at 0.25% neurons and 50% at 0.50%, with zero useful prefix.

Decision: close uniform, adjoint, and measured nonlinear independent-neuron allocation. Single-layer damage is not additive under simultaneous replacement.

## Signed decision-certificate sequence

### PR #31 — Global-bound Signed Dual Cone — rejected

| Hot precision | Mean exact refinement | Maximum refinement | Maximum traffic |
|---:|---:|---:|---:|
| 4-bit | 100% | 100% | 614.25 GiB/token |
| 8-bit | 97.9333% | 99.4116% | 610.6393 GiB/token |

Intervals were sound. Global magnitude bounds were too loose.

### PR #32 — Partitioned Residual Signed Dual Cone — rejected

| Block | Metadata | Mean refinement | Maximum traffic |
|---:|---:|---:|---:|
| 128 | 2.4369 GiB | 96.4416% | 607.5478 GiB/token |
| 256 | 1.2372 GiB | 96.4628% | 607.6055 GiB/token |
| 512 | 0.6373 GiB | 96.4703% | 607.6285 GiB/token |

Block partitioning preserved almost none of the signed cancellation.

### PR #33 — Block Signed Residual Code — rejected

```text
r_b^T x_b
= (r_b^T U_b)(U_b^T x_b)
  + r_b,perp^T x_b,perp
```

| Block/rank | Metadata | Mean refinement | Maximum traffic |
|---:|---:|---:|---:|
| 512/1 | 4.8142 GiB | 94.4114% | 585.9646 GiB/token |
| 1024/1 | 2.4148 GiB | 94.4441% | 586.1607 GiB/token |
| 1024/2 | 3.6299 GiB | 92.3949% | 585.8031 GiB/token |

Signed cancellation was real, but disjoint activations and output duals remained mostly outside the static span.

### PR #34 — Global Margin Refinement — rejected

```text
metadata: 3.6299 GiB
equal-layer mean refinement: 92.3949%
global-width mean refinement: 90.7449%
dual-price mean refinement: 90.7432%
maximum refinement: 93.3392%
maximum traffic: 573.3446 GiB/token
```

Global allocation recovered about 1.65 percentage points only.

## Dynamic and multi-token sequence

### PR #36 — Causal Semantic-State Program Routing — rejected

```text
head: 499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow: 30778002226
```

Best coverage point `K=8/rank4`:

```text
active program: 0.9901 GiB
host bank: 7.9211 GiB
mean reuse: 1.364 tokens
switch traffic: 0.7261 GiB/token
activation perpendicular mean: 42.28%
dual perpendicular mean: 53.91%
p95: 99.50% / 99.84%
```

Increasing state count shortened reuse. Decision: reject static semantic program banks.

### PR #37 — Prompt-Compiled Hankel Decision Program — rejected

```text
head: 12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow: 30778715832
```

Rank32/control16/order2/full projected:

```text
program memory: 0.00673884 GiB
hot compute: 0.008217664 GFLOP/token
prompt projection build: 201.73 GFLOP
```

Real autonomous exact prefixes over 256 tokens:

```text
algorithm-runtime: 1
distributed-database: 1
korean-plm-governance: 2
required: 247
```

Higher-order models sometimes became non-finite. Decision: reject prompt-only linear/quadratic/bilinear/full-lift recurrence.

### PR #38 — Perfect-Oracle Sparse Hankel Repair — rejected

```text
head: 13e3f60876199e4b06577ca51e9fd71f575cb134
workflow: 30779062125
```

The oracle knew the exact target token before deciding to repair.

| Prompt | Repairs / 256 | Mean interval | Repair traffic | Repair compute |
|---|---:|---:|---:|---:|
| algorithm-runtime | 226 | 1.133 | 166.84 GiB/token | 716.58 GFLOP/token |
| distributed-database | 229 | 1.118 | 169.06 GiB/token | 726.09 GFLOP/token |
| korean-plm-governance | 174 | 1.471 | 128.45 GiB/token | 551.70 GFLOP/token |

Decision: reject recurrence plus sparse exact repair, including weaker causal detectors.

### PR #40 — Nonlocal Exact Decision Memory — rejected

```text
head: 91b3e3f062d33087005ae38bbf94b357012f0ccd
corrected workflow: 30780847944
corrected CI: 30780847954
```

One exact first continuation token was charged as a boundary anchor. The first workflow incorrectly counted it as replay and is invalid. Corrected post-anchor results:

| Prompt | Nearest max | Top-64 max | Future-aware global max | Required |
|---|---:|---:|---:|---:|
| algorithm-runtime | 74 | 75 | 75 | 247 |
| distributed-database | 27 | 28 | 28 | 247 |
| korean-plm-governance | 4 | 5 | 5 | 247 |

The global oracle ignored hidden retrieval and searched every prompt suffix. Failure therefore closes prompt-only exact suffix memory regardless of key rank or ANN implementation.

Metadata was not the failure: 65,536 entries at rank128/block256 required 100 MiB and 0.02097 GFLOP/query.

## Accepted proof guardrail

### PR #42 — Exact Dense-Operator Information Lower Bound — accepted and merged

```text
branch: research/exact-operator-lower-bound
head: 7733aa6b8ba1193ed64c20fddcfc643a3d43ed7c
certificate workflow: 30781557141
full CI: 30781557096
main merge: 663dd3d02095f19be269ef60a7c16959f6e16f2f
```

#### Exact-output information theorem

An arbitrary checkpoint with `N` independent `b`-bit parameter codes has `2^(N b)` possibilities. Any checkpoint-specific representation supporting every exact dense operator output must be injective and needs at least `N b` bits in the worst case.

405B Q4 certificate:

```text
N: 405,849,243,648
exact information: 1,623,396,974,592 bits
                 = 188.98828125 GiB
resident allowance: 8 GiB
resident fraction: 4.2330667%
minimum external information: 180.98828125 GiB
optimistic dense arithmetic: 811.698487296 GFLOP
4B arithmetic proxy: 8 GFLOP
ratio: 101.462310912x
```

Accepted conclusion:

> Arbitrary dense exact-output execution cannot be represented by only 8 GiB of checkpoint information. Remaining exact information must be read or equivalently represented, and its cost must be charged or amortized.

#### Skipped-coordinate exact top-1 adversary

For every coordinate in 2x4, 3x5, 4x7, and 8x8 matrices, the gate built two checkpoints that differed only at one uninspected/unrepresented coordinate, had identical observations, but produced different exact outputs and unique top-1 winners.

```text
coordinates tested: 115
passing adversaries: 115
coverage: 100%
```

Accepted conclusion:

> No arbitrary coordinate may be universally omitted from exact top-1 unless its effect is read or represented.

#### Scope boundary

The following remain unproven:

```text
metadata-aware top-1 representation requires N*b bits
real 405B GPU wall clock
real 405B execution
```

Do not report the exact-output cardinality theorem as a complete top-1-only theorem.

## Current interpretation

```text
prompt-derived semantic program reuse: about 1 token
prompt recurrence: at most 2 exact tokens
perfect-token repair: exact target on 68%–89% of tokens
prompt suffix global-oracle maxima: 75 / 28 / 5
exact-output Q4 information: 188.99 GiB
resident allowance: 8 GiB
```

The arbitrary dense exact-output version of the target is contradicted. The exact top-1-only target remains conditionally open only through a charged checkpoint-specific decision representation.

## Prohibited repeats

Do not continue by only changing:

- static rank, block size, or state count;
- norm precision or neuron ordering;
- recurrence rank/order/ridge/lift or detector threshold;
- ANN rank/index/distance/top-k;
- speculative block size while dense arithmetic remains per position;
- lossless metadata that omits information content or build/transfer cost;
- wording that silently substitutes exact-output proof for top-1 proof.

## Current frontier — Experiment 041 Metadata-Aware Top-1 Function Bound

The next Gate counts distinct exact top-1 decision functions rather than distinct weight tensors.

For an `m x d` dense classifier choose:

```text
p = min(floor(m/2), floor(d/2))
q = d - p
```

Use `p` selector coordinates and `p` row pairs. For each pair `r` and payload coordinate `j`, encode one bit `a[r,j]`. Query:

```text
x_(r,j) = selector_r + payload_j
```

A fixed selector margin suppresses all other rows; the winner within the selected pair reveals `a[r,j]`.

This creates:

```text
K = p q independent decision bits
2^K distinct exact top-1 functions
minimum exact metadata for this family >= K bits
```

For an even square `H x H` classifier:

```text
K = H^2 / 4 = N / 4 bits
```

This bound is metadata-aware for the constructed classifier family because distinct bit tables implement distinct top-1 functions.

### Required next work

1. Formalize the selector/payload construction and unique-winner margin.
2. Implement encode/decode queries and exhaustive small-family enumeration.
3. Verify injectivity across all `2^K` functions for tractable shapes.
4. Compute `K=p(d-p)` for attention/MLP matrix shapes.
5. Keep direct classifier, independently callable operator collection, and full transformer conclusions separate.
6. Do not sum layerwise bounds into a transformer theorem until a Llama-like routing construction exposes each layer's bits through final token decisions.
7. Add workflow and raw JSON evidence.

## Mandatory next step

1. Merge this documentation update after full CI.
2. Create `research/top1-function-information-bound` from the new `main`.
3. Add the Experiment 041 proof document, implementation, tests, workflow, and JSON certificate.
4. Update this ledger and `docs/SESSION_HANDOFF.md` before the next progress response.
