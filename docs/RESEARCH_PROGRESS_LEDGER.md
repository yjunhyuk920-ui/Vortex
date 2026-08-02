# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological record of architecture hypotheses, executable gates, measured evidence, and rejection reasons. Every new session must read this file before creating another candidate.

## Fixed target

Execute an arbitrary unmodified Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model quality, require no user training/distillation/fine-tuning/model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current evidence remains below E4. Nothing below is target completion.

## Permanent workflow rule

`docs/WORK_SESSION_PROTOCOL.md` is mandatory. After meaningful repository work and before a user-facing progress/completion answer, update this ledger, `docs/SESSION_HANDOFF.md`, the experiment document, PR decision, and raw result JSON when available.

## Research ledger

### Cascade Capsule Gate 0

- Evidence: E0/E1 symbolic budget plus executable operation-replacement falsification harness.
- Projected memory: about 3.881 GiB.
- Projected traffic: about 1.650 GiB/token.
- Projected compute: about 7.898 GFLOP/token.
- Critical cold-repair requirement: at least 246.889 tokens per full-stream-equivalent repair.
- Decision: conditional hypothesis only; ranks, quality, attention, universality, physical bytes, CUDA, and wall clock remain unproven.

### MLP dictionary, gauge, and functional-skeleton family

- Exact gauge transformation error was about `4.6e-7`.
- 16/32 prototype gauge dictionaries produced teacher top-32 and causal prefix of zero.
- Functional skeleton at 16 prototypes reached about 9.4% teacher top-32, output error about 0.972, and one exact causal step.
- Decision: reject centroid/dictionary/function-skeleton execution as the primary family.

### Decision-Proof LM-head family

- Q4 LM head preserved exact top-1 on 93.75% and kept the exact target within top-32 on 100% of tested positions.
- Unsigned residual norms, fixed exact top-K rows, adaptive row branch-and-bound, and global orthogonal residual sketches had zero unsafe certificates but zero useful certificates.
- Rank-64 global residual sketch removed only about 2.7% of residual energy.
- Decision: candidate discovery is not the limiting issue; excluding all unread rows with cheap global bounds is too loose.

### Static and online activation proof atlas

Static prompt-only ranks 4/8/16:

- continuation perpendicular means about 0.956/0.947/0.934;
- certificate rate 0%;
- unsafe certificates 0.

Online expansion:

- 32 continuation tokens required 32 exact residual-image expansions;
- reuse exactly 1 token per expansion;
- post-expansion certificate 100%;
- projected 405B LM-head residual traffic 2.935546875 GiB/token.

Decision: reject activation-subspace caching; it reproduces tokenwise residual streaming.

### VORTEX-ZIPTREE lossless entropy plus speculation

- 8,388,588 exact TinyLlama FP16 values passed byte-exact codec round trips.
- Measured rate: 11.3330 bits/weight; compression ratio 1.4118x.
- 405B 6 GiB resident threshold: 0.12699 bits/weight.
- At depth 12, projected serialized latency about 1.9275 seconds/token.
- Minimum straight accepted run at the measured rate: 10,649 tokens/pass.
- Decision: retain only as storage/I/O optimization; reject as Gate 0 solution.

### Uniform exact MLP heavy-hitter oracle

The oracle computed full exact gate/up activations and retained original neurons by contribution score. It was an optimistic quality upper bound, not a deployable selector.

| Requested fraction | Projected 405B MLP traffic | Teacher top-32 | Autonomous exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

At 2%, selected neurons covered only about 14.55% of the oracle score and mean MLP output error remained about 0.703.

Decision: reject uniform exact-neuron allocation.

### First-order adjoint layer allocation

A disjoint calibration prompt backpropagated exact top-one versus runner-up margins, and the same total original-neuron count was allocated nonuniformly across layers.

- 0.10%: uniform and adjoint top-32 0%, prefixes 0.
- 0.25%: top-32 improved 43.75% to 56.25%, but top-1 fell to 0% and prefix stayed 0; traffic about 1.638 GiB/token.
- 0.50%: top-32 worsened 56.25% to 43.75%; prefix fell 2 to 0; traffic about 3.172 GiB/token.

Decision: layer sensitivity is nonuniform, but first-order utility does not preserve nonlinear multi-layer behavior.

### Nonlinear layer-damage allocation — rejected

Branch: `research/nonlinear-heavy-hitter-allocation`

Closed PR: `#29 research: reject nonlinear exact-neuron allocation`

Raw-evidence head: `5b1b97c0b449171bda44f3489640ace94a61ee6d`

The experiment measured 22 layers × 6 counts = 132 actual single-layer final-logit damage points, solved a discrete allocation, and evaluated all 22 sparse MLPs simultaneously on a disjoint Korean prompt against an equal-cost uniform allocation.

| Fraction | Used neurons | Projected traffic | Uniform top-32 | Nonlinear top-32 | Uniform prefix | Nonlinear prefix |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10% | 132 | 0.6575 GiB/token | 0% | 0% | 0 | 0 |
| 0.25% | 330 | 1.6381 GiB/token | 43.75% | 18.75% | 0 | 0 |
| 0.50% | 637 | 3.1608 GiB/token | 56.25% | 50% | 2 | 0 |

No nonlinear point improved the equal-cost uniform baseline. The traffic-compatible point preserved no useful token behavior. Single-layer measured damages were not additive under simultaneous multi-layer replacement.

Decision: close the independent exact-neuron heavy-hitter allocation family. Do not recreate uniform, first-order, or single-layer-damage allocation under a new name.

### Signed Dual Cone for SwiGLU — active

Branch: `research/signed-dual-mlp-certificate`

Draft PR: `#31 research: signed dual SwiGLU decision certificate`

Current implementation head at experiment start: `20fcd00f8a6f277bab4a0997f4bcc2876bd5094e`

Motivation: preserve the scalar influence along a token-decision direction rather than reconstructing the complete MLP output vector.

For one fixed MLP input `x` and output dual `q`:

```text
g_i = w^g_i x
u_i = w^u_i x
a_i = SiLU(g_i) u_i
s_i = d_i^T q
c_i = a_i s_i
q^T y = sum_i c_i
```

Low-bit row/column approximations define:

```text
alpha_g_i = ||w^g_i - ŵ^g_i||_2 ||x||_2
alpha_u_i = ||w^u_i - ŵ^u_i||_2 ||x||_2
beta_i    = ||d_i - d̂_i||_2 ||q||_2
L_silu    = 1 + 1/e
```

A sound activation radius is:

```text
alpha_i = L_silu * alpha_g_i * (|û_i| + alpha_u_i)
          + |SiLU(ĝ_i)| * alpha_u_i
```

The implementation uses the exact four-corner product interval for:

```text
a_i in [â_i - alpha_i, â_i + alpha_i]
s_i in [ŝ_i - beta_i, ŝ_i + beta_i]
```

A useful symmetric interpretation is:

```text
|c_i - â_i ŝ_i|
<= |â_i| beta_i + |ŝ_i| alpha_i + alpha_i beta_i
```

Exact original neurons are refined in descending interval width. The 405B exact refinement traffic for selected fraction `f` is:

```text
B_exact(f)
= layers * ceil(intermediate * f)
  * 3 * hidden * source_bits / 8
```

The active real-model Gate uses TinyLlama cached warm decode on English, Korean, code, and mathematics prompts. The exact top-one versus runner-up margin supplies an optimistic fixed dual for every MLP output. It measures 4-bit and 8-bit intervals, sign certificates, equal margin-share certificates, unsafe accepts, and projected 405B exact bytes.

Promotion requires for every prompt:

```text
unsafe certificates = 0
interval containment failures = 0
all layer margin-share targets close
projected 405B exact refinement <= 1.6 GiB/token
```

This remains only a local fixed-dual Gate. Runtime dual construction, dual drift through later nonlinear layers, attention, LM head, hot-state memory, CUDA scheduling, and wall clock remain open even on a pass.

Workflow state when this ledger update began:

```text
CI run 30764835255: Python 3.10/3.12 and validation passed
signed-dual run 30764835283: 4-bit and 8-bit pretrained measurements running
```

## Current interpretation

The project has falsified static low-dimensional activation reuse, global unsigned exclusion proofs, whole-model lossless compression as the main mechanism, and independent exact-neuron vector reconstruction. The active question is narrower and more decision-aligned: whether signed scalar influence intervals become certifiable with a traffic-compatible number of exact original-neuron reads.

## Mandatory next step

1. Finish signed-dual workflow `30764835283`.
2. Inspect actual 4-bit and 8-bit result JSON and PR #31 report.
3. Reject the family if 8-bit still requires traffic-incompatible exact refinement.
4. If a precision point passes, implement multi-layer interval/dual transport without exact future logits or teacher gradients at runtime.
5. Update this ledger and `docs/SESSION_HANDOFF.md` before the next user-facing progress answer.
