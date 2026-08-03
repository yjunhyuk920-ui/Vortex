# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological record of architecture hypotheses, executable gates, measured evidence, and rejection reasons. Every new session must read this file before creating another candidate.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model decisions and quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is target completion.

## Permanent workflow rule

`docs/WORK_SESSION_PROTOCOL.md` is mandatory. After meaningful repository work and before a user-facing progress/completion answer, update this ledger, `docs/SESSION_HANDOFF.md`, the experiment document, PR decision, and raw result JSON when available.

Failed hypotheses are permanent project data. Do not delete, soften, or relabel a negative result as success.

## Foundational Gate 0 budget

The Cascade Capsule symbolic envelope remains the comparison baseline:

```text
projected memory: about 3.881 GiB
projected traffic: about 1.650 GiB/token
projected compute: about 7.898 GFLOP/token
minimum full-stream repair reuse by compute: 246.889 tokens
```

This is only a conditional E0/E1 hypothesis. Rank sufficiency, original quality, attention, universality, physical bytes, CUDA scheduling, and wall clock remain unproven.

## Earlier decisive rejections

### Dictionary, activation-atlas, proof-bound, and entropy families

- Exact gauge transformation error reached about `4.6e-7`, but 16/32 prototype dictionaries produced teacher top-32 and causal prefix zero.
- A 16-prototype functional skeleton reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Q4 LM-head candidate discovery was strong, but unsigned residual norms, fixed/adaptive row proofs, and global orthogonal sketches produced zero useful certificates.
- Static prompt activation ranks 4/8/16 left continuation perpendicular means about 0.956/0.947/0.934 and zero certificate rate.
- Online activation expansion required 32 exact expansions for 32 tokens and projected 2.9355 GiB/token of LM-head residual traffic.
- ZIPTREE measured 11.3330 bits/weight and required a 10,649-token straight accepted run.

Decision: reject static dictionaries, magnitude-only global proofs, activation-subspace caching, and whole-model lossless compression as the primary execution mechanism.

### Exact-neuron heavy-hitter family

Uniform optimistic oracle:

| Fraction | Projected MLP traffic | Teacher top-32 | Exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

PR #29 measured 132 real single-layer damage points and solved a nonlinear allocation:

| Fraction | Uniform top-32 | Nonlinear top-32 | Nonlinear prefix | Traffic |
|---:|---:|---:|---:|---:|
| 0.10% | 0% | 0% | 0 | 0.6575 GiB/token |
| 0.25% | 43.75% | 18.75% | 0 | 1.6381 GiB/token |
| 0.50% | 56.25% | 50% | 0 | 3.1608 GiB/token |

Decision: close uniform, first-order adjoint, and measured nonlinear independent-neuron allocation. Single-layer damage is not additive under simultaneous replacement.

## Signed decision-certificate sequence

### PR #31 — Global-bound Signed Dual Cone — rejected

For one MLP input `x`, output dual `q`, and neuron `i`:

```text
a_i = SiLU(wg_i x) (wu_i x)
s_i = d_i^T q
c_i = a_i s_i
q^T y = sum_i c_i
```

Measured TinyLlama warm decode:

| Hot precision | Mean exact refinement | Maximum refinement | Maximum projected traffic |
|---:|---:|---:|---:|
| 4-bit | 100% | 100% | 614.25 GiB/token |
| 8-bit | 97.9333% | 99.4116% | 610.6393 GiB/token |

Every interval was sound. Global Cauchy and global SiLU bounds were too loose.

### PR #32 — Partitioned Residual Signed Dual Cone — rejected

Used blockwise Cauchy bounds and interval-local SiLU slopes with upward-safe 8-bit residual norms plus 16-bit row scales.

| Block | Metadata | Mean refinement | Maximum exact traffic |
|---:|---:|---:|---:|
| 128 | 2.4369 GiB | 96.4416% | 607.5478 GiB/token |
| 256 | 1.2372 GiB | 96.4628% | 607.6055 GiB/token |
| 512 | 0.6373 GiB | 96.4703% | 607.6285 GiB/token |

At block 128, gate/up radii remained about 99.54% and directional radii about 99.87% of global. Magnitude-only partitioning discarded cancellation.

### PR #33 — Block Signed Residual Code — rejected

For a block basis `U_b`:

```text
r_b^T x_b
= (r_b^T U_b)(U_b^T x_b)
  + r_b,perp^T x_b,perp
```

The signed center was stored exactly and only the orthogonal remainder was bounded.

| Block/rank | Metadata | Mean refinement | Maximum exact traffic |
|---:|---:|---:|---:|
| 512/1 | 4.8142 GiB | 94.4114% | 585.9646 GiB/token |
| 1024/1 | 2.4148 GiB | 94.4441% | 586.1607 GiB/token |
| 1024/2 | 3.6299 GiB | 92.3949% | 585.8031 GiB/token |

Rank2 reduced gate/up radius to about 69.2%, proving signed cancellation matters. However, disjoint activations retained about 69.4% perpendicular energy and output duals about 81.9%.

### PR #34 — Global Margin Refinement — rejected

Removed equal per-layer error shares and solved a global two-sided cover using 41 dual prices:

```text
score_i(lambda) = lambda (a_i - L_i) + (1-lambda) (U_i - a_i)
```

Measured:

```text
metadata: 3.6299 GiB
equal-layer mean refinement: 92.3949%
global-width mean refinement: 90.7449%
dual-price mean refinement: 90.7432%
maximum refinement: 93.3392%
maximum traffic: 573.3446 GiB/token
```

Global allocation recovered only about 1.65 percentage points. Equal layer allocation was secondary, not the core failure.

## Dynamic and multi-token program sequence

### PR #36 — Causal Semantic-State Program Routing — rejected

Evidence head:

```text
research/semantic-state-program-routing
499e5001c21d782adf79fba69ce6f2d445c0cb5e
workflow 30778002226
```

The router used only the previous completed token's final hidden state. Exact current MLP activations and top-two output duals were evaluation oracles.

| States/rank | Mean run | Switch traffic | Activation mean | Dual mean | Activation/dual p95 |
|---:|---:|---:|---:|---:|---:|
| 4/2 | 1.667 | 0.4095 GiB/token | 57.82% | 66.88% | high |
| 8/2 | 1.364 | 0.5005 GiB/token | 44.61% | 58.81% | high |
| 8/4 | 1.364 | 0.7261 GiB/token | 42.28% | 53.91% | 99.50% / 99.84% |
| 16/4 | 1.071 | 0.9241 GiB/token | 43.44% | 54.13% | high |
| 16/8 | 1.071 | 1.4984 GiB/token | no improvement | no improvement | high |

Increasing state count shortened reuse and increased transfer traffic. No configuration approached the <=10% mean and <=20% p95 coverage gate.

Decision: close precompiled semantic-state program banks at the tested scale.

### PR #37 — Prompt-Compiled Hankel Decision Program — rejected

Evidence head:

```text
research/prompt-hankel-decision-program
12f859e4ec288f0d38b29d8b71e494bdc29f6586
workflow 30778715832
```

The exact prompt prefill trajectory was compiled into:

```text
h_t = mean_h + U z_t
v_t = V^T E[token_t]
z_(t+1) = Theta^T phi(z_t, ..., z_(t-p+1), v_(t+1))
logits_t = (W_lm U) z_t + W_lm mean_h + b_lm
```

`phi` included linear, quadratic, bilinear, and full lifts. Only the exact first token from prefill anchored autonomous rollout. Future tokens and hidden states were evaluation-only.

For rank32/control16/order2/full:

```text
program memory: 0.00673884 GiB
hot compute: 0.008217664 GFLOP/token
prompt projection build: 201.73 GFLOP
minimum build reuse: about 21.03 tokens
```

Real 256-token results:

| Prompt | Prompt tokens | Best autonomous prefix | Best teacher top-1 | Best teacher top-32 |
|---|---:|---:|---:|---:|
| algorithm-runtime | 291 | 1 | 6.64% | 34.77% among linear points |
| distributed-database | 333 | 1 | 3.13% | 38.67% |
| korean-plm-governance | 953 | 2 | 29.69% | 92.97% |

Higher ranks and lifted orders reduced prompt training residual but did not preserve future decisions; several configurations became non-finite. Non-finite evidence is stored as JSON `null`, never clipped.

Decision: close prompt-only linear/quadratic/bilinear/full-lift Hankel programs at ranks 8–64 and orders 1–4.

### PR #38 — Perfect-Oracle Sparse Hankel Repair — rejected

Evidence head:

```text
research/oracle-sparse-hankel-repair
13e3f60876199e4b06577ca51e9fd71f575cb134
workflow 30779062125
```

The evaluator used an impossible oracle that knew the exact target token before deciding whether to accept the recurrence prediction or charge one full exact repair. This gives a strict lower bound on every deployable detector's repair rate.

One optimistic 405B repair was charged as:

```text
traffic: 188.9883 GiB
compute: 811.6985 GFLOP
required mean repair interval: >=247 tokens
```

| Prompt | Best config | Repairs | Accepted | Mean interval | Max interval | Repair traffic | Repair compute |
|---|---|---:|---:|---:|---:|---:|---:|
| algorithm-runtime | r8/q8/p1 linear | 226 | 11.72% | 1.133 | 3 | 166.84 GiB/token | 716.58 GFLOP/token |
| distributed-database | r16/q8/p2 linear | 229 | 10.55% | 1.118 | 3 | 169.06 GiB/token | 726.09 GFLOP/token |
| korean-plm-governance | r32/q16/p4 full | 174 | 32.03% | 1.471 | 3 | 128.45 GiB/token | 551.70 GFLOP/token |

The perfect oracle repaired 68%–89% of tokens. The required Gate allowed at most one repair in 256 tokens.

Decision: close prompt-Hankel recurrence plus sparse exact repair, including every weaker causal detector.

### PR #40 — Nonlocal Exact Decision Memory — rejected

Authoritative evidence:

```text
research/nonlocal-exact-decision-memory
91b3e3f062d33087005ae38bbf94b357012f0ccd
corrected workflow 30780847944
corrected full CI 30780847954
```

The prompt-only memory stored:

```text
key_i   = normalize(P^T (h_i - mean_prompt_hidden))
block_i = prompt_token_ids[i+1 : i+1+L]
```

No continuation token or hidden state entered memory construction. One exact first continuation token was charged as the block-boundary anchor. Replay began after that anchor:

```text
query_t  = continuation_hidden_states[t]
target_t = continuation_token_ids[t+1]
```

The first workflow attempt incorrectly counted the boundary anchor as replay. It is invalid and must not be cited as evidence. The corrected alignment is locked by a unit test; only workflow `30780847944` and head `91b3e3f...` are authoritative.

The experiment measured nearest hidden retrieval, top-4/top-16/top-64 future-token oracles, and an impossible global future-token oracle that ignored hidden retrieval and searched every prompt suffix.

Corrected 256-token post-anchor frontier:

| Prompt | Best rank | Nearest max | Top-64 max | Global max | Global first | Post-anchor EOS |
|---|---:|---:|---:|---:|---:|---:|
| algorithm-runtime | 32 | 74 | 75 | 75 | 0 | absent |
| distributed-database | 16 | 27 | 28 | 28 | 0 | absent |
| korean-plm-governance | 16 | 4 | 5 | 5 | 4 | absent |

Required exact replay horizon:

```text
>=247 tokens after the charged boundary anchor
```

The future-aware global oracle reached only 75, 28, and 5 tokens. This closes prompt-only exact suffix memory independent of key rank, ANN index, distance metric, or router.

The 405B metadata budget was small and therefore not the failure. At 65,536 entries, rank 128, block length 256, fp16 keys, 32-bit token IDs, and 25% index overhead:

```text
keys: 16 MiB
blocks: 64 MiB
index: 20 MiB
total: 100 MiB = 0.09765625 GiB
projection plus brute-force lookup: 0.02097152 GFLOP/query
```

Decision: close prompt-only nonlocal exact token-block memory. The required future exact content is not present in the prompt at the needed horizon.

## Current interpretation

The accumulated evidence now falsifies all tested prompt-derived or low-dimensional reusable objects:

```text
static semantic program reuse: about 1 token
prompt dynamic recurrence: at most 2 exact tokens
perfect-token recurrence repair: exact execution on 68%–89% of tokens
prompt suffix global-oracle maxima: 75 / 28 / 5
required strong reuse: 247 tokens
```

The obstruction is not merely VRAM. For the tested mechanisms, exact decision information changes too quickly or is absent from the prompt-derived program.

## Prohibited repeats

Do not create another candidate that only changes:

- static basis rank or block size;
- semantic state-cluster count;
- norm metadata precision;
- neuron ordering or local/global error allocation;
- Hankel rank, order, ridge, or polynomial/bilinear lift;
- recurrence detector thresholds;
- nearest-neighbor rank, ANN index, distance metric, or top-k width;
- ordinary speculative block size while dense target arithmetic remains per verified position.

Any new candidate must introduce a fundamentally different source of exact decision information and charge construction, storage, lookup, validation, and fallback.

## Current frontier — Experiment 040 Exact Dense-Operator Information/Traffic Lower Bound

Prompt-derived execution programs are exhausted. The next Gate must test the compatibility of the fixed universal target itself.

For a dense affine operator:

```text
y = W x
```

suppose an exact runtime neither reads `W[i,j]` nor retains an exact representation sufficient to recover its effect. For any `x[j] != 0`, construct `W'` identical to `W` on every observed or represented degree of freedom but differing at `W[i,j]`. The runtime's observations are identical for `W` and `W'`, while exact `y_i` differs. A downstream two-logit margin can make the exact top-1 token differ as well.

A universal exact runtime must therefore place each decision-relevant degree of freedom into one of:

```text
resident exact information
cold information read for the interaction
lossless metadata with equivalent information content
```

Compression can change representation but cannot erase arbitrary checkpoint information while guaranteeing exact behavior for every checkpoint and input.

Experiment 040 must turn this into an executable lower-bound certificate rather than an unsupported impossibility statement.

### Required implementation

1. Formalize the indistinguishable-checkpoint adversary for dense matrix-vector multiplication.
2. Extend it to exact top-1 decisions through a controlled downstream margin.
3. Implement a simulator that constructs `W/W'`, an input, an inspection mask, identical runtime observations, different exact outputs, and different exact winners.
4. Produce machine-readable lower-bound JSON.
5. Derive resident/read/lossless-metadata inequalities for the 405B target.
6. Separate worst-case arbitrary-checkpoint universality from empirical structure in specific released models.
7. State which fixed assumption must change if the theorem closes:
   - arbitrary checkpoint universality;
   - exact original decisions;
   - 8 GiB residency;
   - 4B-class warm-decode traffic/compute.

### Starting 405B quantities

```text
full Q4-equivalent target information: about 188.9883 GiB
full dense interaction compute: about 811.6985 GFLOP
resident VRAM: 8 GiB before KV/workspace
required full-interaction amortization: about 247 tokens
```

Do not hide skipped weights in uncharged metadata. Automatic first-run representations must be charged by exact information content, host storage, transfer, and construction cost.

## Mandatory next step

1. Merge this durable documentation update after full CI.
2. Create `research/exact-operator-lower-bound` from the new `main`.
3. Add `docs/EXPERIMENT_040_EXACT_OPERATOR_LOWER_BOUND.md`.
4. Implement the dense-output and top-1 adversarial checker with tests.
5. Add an isolated workflow and raw JSON evidence commit.
6. Decide whether the fixed target is mathematically compatible, compatible only under checkpoint structure, or contradicted in the arbitrary dense worst case.
7. Update this ledger and `docs/SESSION_HANDOFF.md` before the next user-facing progress answer.
