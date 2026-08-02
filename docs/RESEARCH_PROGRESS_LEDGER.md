# VORTEX research progress ledger

Last updated: 2026-08-03 (Asia/Seoul)

This is the durable chronological record of architecture hypotheses, executable gates, measured evidence, and rejection reasons. New sessions must read this file before creating another candidate.

## Fixed target

Execute an unmodified arbitrary Hugging Face 405B-class dense model on one 8 GiB VRAM GPU, preserve original-model quality, require no user training or model-specific adapter work, and reach p50 warm decode within 1.2x of a native 4B Q4 baseline on the same machine.

Current project evidence remains below E4. No result below may be described as target completion.

## Research ledger

### Gate 0 Cascade Capsule

- Evidence: E0/E1 symbolic budget plus executable falsification harness.
- Projected memory: about 3.881 GiB.
- Projected traffic: about 1.650 GiB/token.
- Projected compute: about 7.898 GFLOP/token.
- Critical cold-repair requirement: at least 246.889 tokens per full-stream-equivalent repair.
- Status: conditional hypothesis only. Real disjoint-prompt amortization, rank sufficiency, quantization quality, bounded attention, CUDA wall clock, and architecture universality remain unproven.

### MLP dictionary, gauge, and functional-skeleton family

- Real TinyLlama all-layer MLP replacement was executed.
- Exact gauge transform error was about `4.6e-7`, but 16/32 prototype collapse produced teacher top-32 and causal prefix of zero.
- Functional skeleton reached only about 9.4% teacher top-32 at 16 prototypes, output error about 0.972, and one exact causal step.
- Decision: reject centroid/dictionary/function-skeleton compression as the primary execution family.

### Decision-Proof LM-head family

- Q4 LM head preserved exact top-1 on 93.75% and exact target within top-32 on 100% of the tested positions.
- Unsigned residual norm, fixed exact top-K rows, adaptive row branch-and-bound, and global orthogonal residual sketches all produced zero unsafe certificates but zero useful certificates.
- Rank-64 global residual sketch removed only about 2.7% of residual energy.
- Decision: candidate discovery is not the limiting issue; exclusion bounds for all unread vocabulary rows are too loose.

### Static and online activation proof atlas

Static prompt-only atlas:

- ranks 4/8/16;
- continuation perpendicular mean about 0.956/0.947/0.934;
- certificate rate 0% at every rank;
- unsafe certificates 0.

Online expansion:

- 32 continuation tokens required 32 exact residual-image expansions;
- reuse was exactly 1 token per expansion;
- post-expansion certificate rate 100%;
- projected 405B LM-head residual traffic 2.935546875 GiB/token.

Decision: reject static and online activation-subspace caching. It reproduces tokenwise residual streaming rather than amortizing it.

### VORTEX-ZIPTREE lossless entropy plus speculation

- Sampled 8,388,588 exact TinyLlama FP16 values with byte-exact codec round trips.
- Measured bit rate: 11.3330 bits/weight.
- Measured compression ratio: 1.4118x.
- 405B target threshold for 6 GiB resident target state: 0.12699 bits/weight.
- At candidate depth 12, projected serialized latency: about 1.9275 seconds/token.
- Minimum straight accepted run at measured rate: 10,649 tokens per target pass.
- Decision: retain codec as an I/O/storage optimization only; reject as Gate 0 solution.

### Uniform exact MLP heavy-hitter oracle

This oracle computed full exact gate/up activations and then retained the highest original-neuron contribution scores. It is an optimistic quality upper bound, not a fast selector.

| Requested fraction | Projected 405B MLP traffic | Teacher top-32 | Autonomous exact prefix |
|---:|---:|---:|---:|
| 0.10% | 0.623 GiB/token | 0% | 0 |
| 0.25% | 1.546 GiB/token | 43.75% | 0 |
| 0.50% | 3.080 GiB/token | 56.25% | 2 |
| 1.00% | 6.148 GiB/token | 50% | 0 |
| 2.00% | 12.285 GiB/token | 50% | 0 |

At 2%, the selected neurons covered only about 14.55% of the oracle contribution score and mean MLP output error remained about 0.703.

Decision: reject uniform tokenwise exact-neuron allocation.

### First-order adjoint layer allocation

A disjoint calibration prompt backpropagated exact top-one versus runner-up logit margins. The same total original-neuron count was assigned nonuniformly across layers and tested on a disjoint Korean prompt.

- 0.10%: uniform and adjoint top-32 both 0%, prefixes 0.
- 0.25%: top-32 improved 43.75% to 56.25%, but top-1 fell to 0% and prefix stayed 0; projected traffic 1.638 GiB/token.
- 0.50%: top-32 worsened 56.25% to 43.75%; prefix fell 2 to 0; projected traffic 3.172 GiB/token.

Decision: layer sensitivity is nonuniform, but first-order margin utility does not preserve nonlinear multilayer behavior. Reject the tested allocator.

### Nonlinear layer-damage allocation — active

Branch: `research/nonlinear-heavy-hitter-allocation`

Draft PR: `#29`

Hypothesis:

- measure actual final-token cross-entropy damage when one MLP layer at a time is replaced by exact original-neuron subsets at counts 1/4/8/16/32/64;
- solve a discrete byte-constrained allocation over the measured nonlinear curves;
- validate all 22 replaced MLPs simultaneously on a disjoint prompt against an equal-cost uniform allocation.

The initial workflow failed before measurement because it referenced `tests/test_adjoint_heavy_hitter.py`, a sibling-branch file absent from the branch. This is infrastructure failure, not candidate evidence.

Fix commit: `b7ad7aefb8ef8a64cc1979735e1c9ba487e944ac`

The workflow now explicitly validates only branch-owned test files and checks they exist before pytest.

Promotion thresholds remain:

- projected partial MLP traffic at or below 1.6 GiB/token;
- disjoint teacher top-32 at least 95%;
- autonomous exact prefix at least 4 tokens;
- no hidden full-activation or exact-target cost omitted from accounting.

## Current interpretation

The accumulated failures show that the tested activations and MLP functions are not reusable in a sufficiently low-dimensional static basis, and that tokenwise exact-neuron sparsity is far weaker than required under a uniform or first-order layer budget. The remaining open question is whether actual nonlinear layer damage is sufficiently concentrated that a discrete allocation can preserve behavior within the same byte budget.

## Mandatory next step

1. Complete PR #29 after the workflow-isolation fix.
2. Commit its raw result JSON.
3. Close or promote the exact-neuron family from measured disjoint results.
4. If rejected, move to an execution representation whose reusable object is a certified multi-layer decision influence cone rather than an activation basis, whole residual image, or independent-neuron subset.
