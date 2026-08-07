# EXP-080A -- Hyperblock Exact Rectangular-Multiplication Gate

## Status

Complete. The authoritative E1 result rejects standard constructive Strassen
Hyperblocks as a core. Source commit
`7f1c66125a844f5366e2b88f0aaa07b089dc0378`; evidence commit
`98e90899a0adf3614d8dbe17691167ce426bae3b`.

## Question

If a dense 405B decoder is granted an exact block of `K` future activations,
can exact cross-token rectangular matrix multiplication reduce both target
weight traffic and arithmetic to the final 4B-equivalent p50 fraction?

This experiment tests the new arithmetic mechanism only. It does not reopen
the causal proposal mechanisms rejected by EXP-048, EXP-049, EXP-050, or
EXP-076. Exact future activations are a non-deployable favorable oracle.

## Why this is not another Jacobi or draft experiment

The closed block experiments amortized a target weight stream across proposed
positions but still charged one dense matrix-vector product per position.
EXP-080A changes the query shape from `K` independent vectors to one matrix:

```text
W[m,n] @ X[n,K] -> Y[m,K]
```

It then asks whether an exact bilinear algorithm shares scalar arithmetic
across the `K` columns. No new token proposal, fixed-point iteration, draft
selection, tree search, or acceptance measurement is performed.

## Registered workload

The shape population is the frozen Llama-405B inventory in
`results/exp_071/raw/tensor_rows.jsonl`, SHA-256
`5009a4ed7234bd24eb1488b0f96a9c847158c2a614a6c90519a16cf4212ba82a`.
All registered projection families and the LM head are included. The embedding
matrix is excluded from dense arithmetic because decode performs a row lookup,
not a complete matrix multiply.

Registered block lengths:

```text
32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
```

Every rectangular product is partitioned into power-of-two square tiles no
larger than `min(m,n,K)`. Padding, all square products, and cross-inner-block
output accumulations are charged.

## Exact constructive arm

The constructive arm is ordinary recursive Strassen multiplication:

```text
M(s) = 7 M(s/2)
A(s) = 7 A(s/2) + 18 (s/2)^2
```

At a classical leaf of width `b`:

```text
M(b) = b^3
A(b) = b^2 (b - 1)
```

For every tile width, the prototype grants a free oracle choice of the
power-of-two leaf width that minimizes total scalar multiplies plus additions.
This omits dispatch, layout conversion, cache misses, integer widening,
rounding reconstruction, and kernel overhead and is therefore favorable.

Small signed-integer matrices at widths 1, 2, 4, and 8 must be multiplied by
both an independent classical reference and the recursive Strassen evaluator.
Any exact mismatch invalidates the result.

## Non-constructive asymptotic control

A separate diagnostic grants a unit-constant arithmetic count
`s^2.371552` for each square tile and charges only cross-tile accumulation.
This is not an implementation and cannot promote the candidate. It answers
whether even an unrealistically favorable modern matrix-multiplication exponent
leaves a mathematical route.

## Registered resource equations

The final fractions are:

```text
p50 = 1.2 * 4 / 405 = 0.011851851851851851
p95 = 1.5 * 4 / 405 = 0.014814814814814815
```

For one perfect target sweep over `K` exact future positions:

```text
target traffic fraction = 1 / K
target arithmetic fraction = fast_block_ops / classical_block_ops
```

For `R` target sweeps, both target fractions are multiplied by `R`. A streamed
4B draft adds `4/405` per proposed token before target work. The authoritative
Gate grants `R=1`, a zero-cost perfect future oracle, and no draft cost.

The favorable workspace estimate charges one BF16 input block, one BF16 output
block, one Q4 weight tile, and three FP32 square scratch tiles. KV cache, full
Transformer nonlinear state, allocator fragmentation, and physical kernels are
reported `UNVERIFIED`; therefore a workspace pass is necessary but insufficient.

## E0 scorecard

1. **Target-scale upside.** Weight traffic falls as `1/K`; subcubic block
   arithmetic is the only proposed route for reducing dense MACs across tokens.
2. **Mechanism novelty.** Prior block experiments used dense per-position MACs.
   This Gate tests exact cross-query bilinear arithmetic instead.
3. **Evidence basis.** Exact Strassen identities are constructive; the frozen
   shape population fixes the target matrices. Practical benefit is unassumed.
4. **Scaling reason.** The theoretical arithmetic ratio improves with the
   square tile width and therefore with sufficiently wide blocks and matrices.
5. **Universality.** The tiler is derived mechanically for arbitrary dense
   matrices and does not modify or train the checkpoint.
6. **Correctness closure.** The exact integer control must equal the classical
   product; future target state remains oracle-only and never deployable.
7. **Resource closure.** Scalar multiplies, additions, padding, tile
   accumulation, draft fraction, sweeps, traffic, and favorable workspace are
   explicit. Physical timing remains unverified.
8. **Cheap falsifiability.** Count exact constructive operations before any
   model execution, packed kernel, checkpoint download, or target-server work.

## Gate

The constructive Strassen arm must contain at least one registered `K` where,
under the zero-cost perfect-future oracle:

```text
traffic fraction <= p50 target fraction
arithmetic fraction <= p50 target fraction
favorable workspace <= 8 GiB
exact integer control mismatches = 0
```

Passing authorizes only an exact packed/rectangular constructor Gate. It does
not reopen causal block generation. Passing only the unit-constant exponent
control is classified as theoretical headroom without promotion.

Failure decision:

```text
REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY
```

Control failure decision:

```text
INVALID_HYPERBLOCK_ARITHMETIC_CONTROL_FAILURE
```

## Strongest falsification

Even if arithmetic passes, an arbitrary causal triangular target can reveal at
most one previously unknown token per black-box target round. EXP-049 already
observed that barrier, EXP-048 measured 58 target passes for 32 exact tokens,
and a streamed 4B draft needs at least 507 perfectly accepted target positions
before overhead. EXP-080A cannot convert an arithmetic pass into a causal
runtime result.

## Stop rule and claim boundary

On constructive failure, do not build CUDA, Strassen, bit-sliced, speculative,
tree, or target-server backends around this path. A continuation requires an
exact algorithm with an independently justified operation count below this
constructive ceiling and a materially new causal source that addresses the
committed triangular/draft failures.

Evidence is capped at Phase A/B E1 synthetic/reference. No checkpoint is
downloaded or executed. CUDA, physical latency, PCIe, SSD, 8 GiB peak, exact
BF16/Q4 reduction-order compatibility, causal future production, 122B/405B
execution, and E2-E7 remain **NOT TESTED**.

## Authoritative result

All 80 signed-integer reference comparisons matched exactly across widths
`1,2,4,8`, every registered classical leaf size, and eight deterministic trials.
The frozen workload contained eight dense-operation families representing
`403,747,897,344` per-token coefficients.

The constructive arm produced no joint p50 pass at any registered block length.
The most favorable row was `K=16,384`:

```text
traffic fraction                         0.0061035%
constructive arithmetic fraction        36.1115797%
constructive speedup                     2.769195x
p50 arithmetic allowance                 1.1851852%
remaining miss factor                   30.469145x
favorable workspace                      7.5390625 GiB
```

Thus traffic and the deliberately incomplete workspace equation passed, while
constructive arithmetic failed by more than thirty times even with exact future
activations, a zero-cost proposal, one target sweep, free leaf selection, and no
kernel/layout/numerical overhead.

The unit-constant `omega=2.371552` diagnostic first crossed the joint Gate at
`K=512`; after charging a streamed 4B draft it first crossed at `K=8,192`.
Those rows are non-constructive asymptotic controls. Existing causal evidence
reaches at most six target-only fixed-point positions, three external-draft
positions, and requires 507 perfect positions for a 4B draft, so the diagnostic
does not create a deployable path.

Decision:

```text
REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY
```

This closes standard recursive Strassen as the arithmetic engine for the
registered Hyperblock interface. It does not prove every exact low-constant
rectangular or packed algorithm impossible. Reopening requires a constructive
fully charged arithmetic reduction at least `30.469145x` beyond this arm and a
materially new causal block source; neither exists in the current evidence.

Authority: `results/exp_080a/summary.json`; deterministic core SHA-256
`7578c4c9f463da8135f3c320df9d7fb920ffc172d31fdd2f60b30af9778280ce`.
