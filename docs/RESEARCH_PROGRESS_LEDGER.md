# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological record of architecture hypotheses, executable gates, measured evidence, and rejection reasons. Every new session must read this file before creating another candidate.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is target completion.

## Permanent workflow rule

`docs/WORK_SESSION_PROTOCOL.md` is mandatory. After meaningful repository work and before a user-facing progress/completion answer, update this ledger, `docs/SESSION_HANDOFF.md`, the experiment document, PR decision, and raw result JSON when available.

## Earlier decisive gates

### Cascade Capsule Gate 0

- E0/E1 symbolic budget and executable operation-replacement falsification harness.
- Projected memory about 3.881 GiB, traffic about 1.650 GiB/token, compute about 7.898 GFLOP/token.
- Critical cold-repair requirement: at least 246.889 tokens per full-stream-equivalent repair.
- Decision: conditional hypothesis only; rank sufficiency, quality, attention, universality, physical bytes, CUDA, and wall clock remain unproven.

### Dictionary, functional skeleton, and global proof families

- Exact gauge transform error about `4.6e-7`, but 16/32 prototype gauge dictionaries produced teacher top-32 and causal prefix zero.
- Functional skeleton at 16 prototypes reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Q4 LM head retained exact top-1 on 93.75% and exact target within top-32 on 100%, but unsigned norms, exact top-K rows, adaptive branch-and-bound, and global orthogonal sketches produced zero useful certificates.
- Decision: reject centroid/function dictionaries and magnitude-only global exclusion proofs.

### Activation atlas and ZIPTREE

- Static prompt atlas ranks 4/8/16: continuation perpendicular means about 0.956/0.947/0.934; certificate rate 0%.
- Online atlas: 32 exact residual expansions for 32 tokens, reuse 1 token/expansion, projected LM-head residual traffic 2.935546875 GiB/token.
- ZIPTREE: 8,388,588 exact TinyLlama FP16 values, 11.3330 bits/weight, 1.4118x compression, 10,649-token straight acceptance required.
- Decision: activation subspace caching and whole-model lossless compression do not close Gate 0.

### Exact-neuron heavy-hitter family

Uniform optimistic oracle:

| Fraction | Projected MLP traffic | Teacher top-32 | Exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

First-order adjoint allocation:

- 0.25% improved top-32 from 43.75% to 56.25%, but top-1 fell to 0% and prefix remained 0.
- 0.50% worsened top-32 and prefix.

Nonlinear measured allocation, PR #29:

- measured 22 layers × 6 counts = 132 single-layer damage points;
- 0.10%: uniform/nonlinear top-32 0%;
- 0.25%: uniform 43.75%, nonlinear 18.75%, traffic 1.6381 GiB/token;
- 0.50%: uniform 56.25%, nonlinear 50%, traffic 3.1608 GiB/token.

Decision: close uniform, adjoint, and nonlinear independent exact-neuron allocation. Single-layer damage is not additive under simultaneous replacement.

## Signed decision-certificate sequence

### PR #31 — Global-bound Signed Dual Cone — rejected

For one MLP input `x`, output dual `q`, and neuron `i`:

```text
a_i = SiLU(wg_i x) (wu_i x)
s_i = d_i^T q
c_i = a_i s_i
q^T y = sum_i c_i
```

The low-bit interval used global Cauchy residual-dot bounds, global `L_silu = 1+1/e`, and exact four-corner product hulls.

Measured TinyLlama warm decode:

| Hot precision | Mean exact refinement | Maximum refinement | Maximum projected traffic |
|---:|---:|---:|---:|
| 4-bit | 100% | 100% | 614.25 GiB/token |
| 8-bit | 97.9333% | 99.4116% | 610.6393 GiB/token |

All intervals were sound: zero unsafe certificates and zero containment failures.

Decision: signed scalar projection is insufficient with global magnitude bounds.

### PR #32 — Partitioned Residual Signed Dual Cone — rejected

Replaced global Cauchy with:

```text
|r_i^T x| <= sum_k ||r_i[B_k]||_2 ||x[B_k]||_2
```

and used interval-local SiLU slopes. Residual block norms were executed as upward-safe 8-bit codes with 16-bit row scales, including scale memory.

| Block | Metadata | Mean refinement | Maximum refinement | Maximum exact traffic |
|---:|---:|---:|---:|---:|
| 128 | 2.4369 GiB | 96.4416% | 98.9080% | 607.5478 GiB/token |
| 256 | 1.2372 GiB | 96.4628% | 98.9177% | 607.6055 GiB/token |
| 512 | 0.6373 GiB | 96.4703% | 98.9217% | 607.6285 GiB/token |

At block 128, gate/up radii remained about 99.54% and directional radii about 99.87% of global.

Decision: norm partitioning preserves almost none of the signed cancellation.

### PR #33 — Block Signed Residual Code — rejected

For each block and shared orthonormal basis `U_b`:

```text
r_b^T x_b
= (r_b^T U_b)(U_b^T x_b)
  + r_b,perp^T x_b,perp
```

Stored the first term as signed center and bounded only the orthogonal remainder. Build and evaluation prompts were disjoint.

| Block/rank | Metadata | Mean refinement | Maximum refinement | Maximum traffic |
|---:|---:|---:|---:|---:|
| 512/1 | 4.8142 GiB | 94.4114% | 95.3940% | 585.9646 GiB/token |
| 1024/1 | 2.4148 GiB | 94.4441% | 95.4255% | 586.1607 GiB/token |
| 1024/2 | 3.6299 GiB | 92.3949% | 95.3682% | 585.8031 GiB/token |

Rank2 reduced gate/up radius to about 69.2%, but evaluation activations retained about 69.4% perpendicular energy and duals about 81.9%.

Decision: signed cancellation is real, but static build-prompt codebooks do not transfer sufficiently.

### PR #34 — Global Margin Refinement — rejected

Removed equal per-layer error shares. For each interval:

```text
l_i = a_i - L_i
u_i = U_i - a_i
```

and required globally:

```text
sum unrefined l_i <= tau
sum unrefined u_i <= tau
```

A 41-point dual-price sweep ordered neurons by:

```text
lambda l_i + (1-lambda) u_i
```

using the strongest PR #33 representation.

Measured:

- metadata 3.6299 GiB;
- equal-layer mean refinement 92.3949%;
- global-width mean refinement 90.7449%;
- dual-price mean refinement 90.7432%;
- maximum dual-price refinement 93.3392%;
- maximum projected traffic 573.3446 GiB/token.

Decision: global allocation recovers only about 1.65 percentage points. Equal layer allocation was a secondary inefficiency, not the core failure.

## Current interpretation

The project has now falsified:

- static low-dimensional activation reuse;
- magnitude-only residual bounds;
- static signed residual codebooks built from disjoint prompts;
- independent exact-neuron subsets;
- equal-layer and global interval ordering as sufficient remedies;
- whole-model lossless compression plus speculative amortization as the primary path.

The consistent signal is that signed residual cancellation matters, but the relevant activation and output-dual state changes strongly across prompts and tokens. A viable next representation must be keyed to online semantic state or must amortize an exact decision program across multiple future tokens. Another static build-prompt basis, norm partition, or neuron-ordering variation is prohibited unless it introduces a new measurable reuse mechanism.

## Current frontier

No research PR remains promoted. PRs #29, #31, #32, #33, and #34 are closed with raw evidence committed on their branch heads.

The next architecture must begin with a new proof-first certificate for one of these two mechanisms:

1. **Semantic-state-keyed signed residual programs** — select or construct a small signed program from current hidden/dual state before exact weight reads, and prove its program-build cost amortizes across tokens.
2. **Multi-token decision program** — one exact target interaction produces a certified program that commits or bounds several future token decisions without tokenwise full residual images.

## Mandatory next step

1. Derive 405B memory/traffic/compute equations for both frontier mechanisms before implementation.
2. Reject any design whose program construction requires tokenwise full residual streaming.
3. Define a measurable reuse factor and minimum amortization threshold.
4. Implement the cheaper falsification first on disjoint multi-token TinyLlama traces.
5. Update this ledger and `docs/SESSION_HANDOFF.md` before the next user-facing progress answer.
