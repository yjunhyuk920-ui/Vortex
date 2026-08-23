# EXP-101A — Structured Direct-Sum Composition Gate

## Question

Can the first published exact tensor source outside the frozen AlphaTensor catalog,

\[
\langle6,6,6\rangle
\le 137\langle1\rangle\oplus8\langle1,1,2\rangle,
\]

be recursively composed with the exact integral small-coefficient catalog retained by EXP-100A and reduce the registered Llama-3.1-405B projection multiplication count to at most 10%?

## Three premise flips

1. **Nonuniform direct-sum recursion:** one parent produces two differently shaped child populations rather than one uniform rank population.
2. **Cyclic axis routing:** the doubled child may be routed through token, input-channel, or output-channel dimensions at every state.
3. **Cross-source composition:** every child may switch between the new structured relation, any retained AlphaTensor integral scheme orientation, and classical multiplication.

The third is selected because it strictly contains the first two and the EXP-100A catalog comparator.

## Frozen primary source

- paper: Kauers, Moosbauer, Wood, *Exploiting the Structure in Tensor Decompositions for Matrix Multiplication*, arXiv:2602.11041v3;
- repository: `mkauers/matrix-multiplication`;
- commit: `12c26b29a5458e173813911fb4f2c2865fba841e`;
- file: `structured/666.exp`;
- Git blob: `9e940a8308d6faf5784967ec160d6429e5d1f563`;
- nonempty product rows: 153.

The reported triple-cyclic construction is independently recomputed as

\[
137^3+3(137^2)(8)(2)+3(137)(8^2)(4)+8^3(7)=3{,}581{,}065.
\]

## Exact recurrence

For positive dimensions `(m,k,n)` and remaining depth `d`, let `C_d` be the minimum scalar multiplication count. Classical multiplication is always available:

\[
C_d(m,k,n)\le mkn.
\]

For every exact uniform catalog scheme `(a,b,c,r)`, include

\[
r\,C_{d-1}(\lceil m/a\rceil,\lceil k/b\rceil,\lceil n/c\rceil).
\]

For each cyclic doubled axis `q`, include

\[
137 C_{d-1}(u,v,w)+8 C_{d-1}(D_q(u,v,w)),
\]

where `(u,v,w)=(\lceil m/6\rceil,\lceil k/6\rceil,\lceil n/6\rceil)` and `D_q` doubles one selected child dimension. Dynamic programming chooses independently at every child state.

## Favorable grants

This is deliberately stronger than a deployable executor. It charges **only scalar multiplications**. The following are free:

- all input, checkpoint, and output transforms;
- all additions, scales, copies, packing, and address generation;
- all cold bytes and transformed checkpoint expansion;
- all workspace and metadata;
- all native-order BF16 repair;
- perfect future blocks with `N/A=1`;
- exact finite-word representation.

Therefore rejection is decisive for this frozen tensor source as a 10x arithmetic core. Promotion only authorizes a fully charged transform-circuit Gate.

## Population and Gate

All non-embedding registered 405B projection families are weighted by their actual counts. Frozen block lengths are `32..16384` powers of two. Maximum recursion depth is 12, with classical fallback at every node.

Promotion requires

\[
\min_K\frac{\sum_f c_f C_{12}(K,k_f,n_f)}{\sum_f c_f Kk_fn_f}\le0.10
\]

with zero integrity failures.

## Stop rule

On rejection, do not tune depth, block list, cyclic order, catalog subset, or thresholds. Reopening requires an exact tensor outside both pinned sources, a transform circuit shared across several projections/layers, or a nonlinear whole-layer instruction whose dominant operation is not this bilinear recursion.

## Claim boundary

This Gate is an E0 analytical oracle. It does not execute TARGET-W, construct a causal future block, preserve a complete successor state, allocate physical 8-GiB VRAM, implement a CPU/GPU kernel, or measure same-machine 4B-Q4 latency.
