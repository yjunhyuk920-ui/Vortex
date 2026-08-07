# EXP-082A -- Exact Differential Spanning-Tree MatVec Gate

Status: E0 candidate admitted for preregistration. No result exists yet.

## Question

Do real Q4 Transformer projection matrices admit an exact row- or column-
difference spanning tree whose complete coefficient work can approach the
`1.185185185%` 405B/4B-equivalent p50 fraction?

## New mechanism

For `W[m,n]`, add an all-zero row and construct a spanning tree over the rows
with edge weight equal to Hamming distance. If row `i` has parent `p`, store the
exact sparse difference `Delta_i = W_i - W_p` and evaluate:

```text
y_root = 0
y_i = y_p + Delta_i dot x
```

The exact coefficient work is the total Hamming weight of the tree. The dual
column construction stores column differences and computes subtree sums of the
current `x`, then accumulates one scaled delta vector per edge. The candidate
chooses the cheaper exact orientation for every matrix.

This is materially broader than EXP-064. That experiment attached every row to
at most 32 frequent prototype rows. EXP-082A permits every row or column to be
an intermediate prototype and permits arbitrarily long exact difference
chains. The mechanism has a new asymptotic basis: Anand, van den Brand, and
McCarty prove exact online MatVec bounds from low-weight Hamming MSTs and
VC/Pollard structure:

<https://arxiv.org/html/2502.21240v3>

## Correctness contract

Every stored edge is an exact integer difference of pinned Q4 values. Tree
traversal must reconstruct every source row/column and agree with direct
integer `W @ x` before any candidate output can commit. Malformed trees,
indices, deltas, scales, nonfinite activations, checksum failures, or missing
metadata must run the unchanged dense Q4 path or abort. No future token,
approximation, learned router, retraining, or target-specific hand-authored
adapter is allowed.

Q4-to-original-model output preservation is not established by this Gate and
must remain `NOT TESTED`.

## Cheapest decisive Gate

Computing a complete exact MST is unnecessary if a certified lower bound
already exceeds the final budget. Partition every candidate row into exact
blocks of `B=32` coefficients and assign a collision-free pattern ID within
each block. Define:

```text
d_B(u,v) = number of blocks whose exact patterns differ
```

Since every differing block contains at least one differing coefficient,
`d_B(u,v) <= Hamming(u,v)`. For every vertex `v`, including the added zero
vertex, let `delta(v)` be its nearest-neighbor `d_B` distance. Any spanning tree
therefore obeys:

```text
MST_Hamming >= ceil(sum_v delta(v) / 2)
```

Compute this bound for both row and column orientations and grant the smaller
one to the candidate. A hash is not permitted to assert inequality; exact block
patterns must receive the IDs. This is a favorable lower bound because it
charges only one coefficient operation per differing block and grants all
within-block differences, metadata decoding, tree traversal, and memory
effects for free.

Stage 1 stops and rejects if the weighted minimum lower-bound operation
fraction alone exceeds `1.185185185%`. Only an inconclusive Stage 1 authorizes
an exact/independently checked MST construction on the registered tensors.

## Registered small-checkpoint population

Use the already present, pinned, unmodified payload:

```text
model       Qwen/Qwen3.5-0.8B
revision    2fc06364715b967f1860aea9cf38778875588b17
weight      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
layers      3, 11, 23
families    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
quantizer   unchanged symmetric per-output-row Q4 reference
```

The layers are data-independent and were already registered as valid full-
attention/depth locations in EXP-081A. No prompt or forward execution is
required because the tree must serve every current causal activation.

## E0 resource route

Let `h` be the chosen tree's nonzero difference fraction. A direct sparse edge
encoding must at minimum charge coefficient work, parent/row pointers,
coordinate indices, signed deltas, per-row scales, current activation reads,
output writes, original cold Q4 fallback, and build amortization.

The optimistic structural ceiling is real: if each new row differs from an
earlier row in only `O(1)` coordinates, coefficient work is `O(m+n)` rather
than `O(mn)`, strengthening with width. With 16-bit coordinates, one signed
delta byte, and two-byte activations, even this five-byte-per-change favorable
encoding can fit the final traffic fraction when `h` is below approximately
`0.1185%`. Constant-dimension theory supplies an asymptotic reason to test the
population; the measured tree still controls promotion.

The strongest counterexample is a random Q4 matrix: two rows differ in about
`15/16` coordinates, so even its best spanning tree remains near dense. EXP-064
is prior evidence that this counterexample may describe Transformer weights,
but its restricted prototype set did not measure the full-tree optimum.

## Gates

Stage 1 scientific rejection:

- any reconstruction/control mismatch;
- any block-distance lower bound that is not independently dominated by exact
  Hamming distance in controls;
- weighted minimum row/column MST operation lower bound above
  `1.185185185%`;
- missing registered tensor family or checkpoint/hash mismatch.

Stage 2 promotion, only if Stage 1 is inconclusive:

- exact tree reconstruction and direct integer MatVec equality in every
  control;
- weighted p50 and every-family exact coefficient fraction no greater than
  `1.185185185%`;
- fully charged projected operation and logical traffic fractions no greater
  than `1.185185185%`;
- projected sidecar no greater than 8 GiB;
- compile amortization no greater than 512 committed tokens.

A pass authorizes only a minimal operation-replacement reference. It does not
authorize CUDA, a packed kernel, a larger download, or the Ubuntu host.

## Stop rule

If Stage 1 rejects, do not build an MST, sparse-delta runtime, CUDA kernel, or
larger-checkpoint sweep. Do not rescue the family by changing block size,
layers, projections, quantizer, tree metric, or selecting isolated matrices.
Reopening would require a new exact shared-computation structure not captured
by row/column Hamming difference trees.

## Evidence ceiling

Phase A/B exact structural controls plus Phase C small-real-checkpoint weight
observation, ceiling E1. Physical operation replacement, BF16/Q4 output
equivalence, 8 GiB allocation, target-server bandwidth/latency, 122B/405B, and
E2-E7 remain `NOT TESTED`.

