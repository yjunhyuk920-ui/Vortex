# EXP-100A — Explicit Rectangular Tensor-Scheme / Weight-Transform Dual-Roofline Gate

## 1. Question

EXP-099A established a favorable necessary condition for exact fast matrix
multiplication on the pinned public DEV-W checkpoint: after rounding an
accurately enclosed real dot product to BF16, only a very small fraction of
projection rows required the official native accumulation order. That result
does **not** provide a fast algorithm. EXP-100A asks the next cheapest question:

> Does a pinned public catalog of explicit, exact, small-coefficient rectangular
> matrix-multiplication tensors contain a mixed recursive program that, after all
> factor transforms, writes, checkpoint traffic, favorable workspace, and
> EXP-099A native-row repairs are charged, reduces the registered 405B dense
> projection arithmetic by at least 10×?

The experiment is a bounded structural/resource Gate. It deliberately grants a
perfect future activation block and a perfect repair selector. It is not an
executor and cannot promote itself to 405B feasibility.

## 2. Three premise reversals considered before implementation

This round starts from three materially different execution principles rather
than another parameter sweep around standard square Strassen.

### A. Cyclic rectangular tensor recursion

The hidden premise in EXP-080A was that one square `2×2×2` recursion should be
applied to every projection shape. EXP-100A instead uses exact rectangular
tensors and every cyclic/reflected assignment of their three factor spaces to

```text
left activation × checkpoint weight -> output
```

A mixed sequence

```text
<a1,b1,c1>:R1, <a2,b2,c2>:R2, ..., <ad,bd,cd>:Rd
```

has aggregate split products

```text
A = product_i ai
B = product_i bi
C = product_i ci
R = product_i Ri
```

and, after padding, leaf multiplication count

```text
R * ceil(M/A) * ceil(K/B) * ceil(N/C).
```

This permits one recursion level to favor the token-block axis, another the
inner dimension, and another the output-row dimension.

### B. Checkpoint-static outer weight transforms

For a recursion cut depth `g`, the first `g` weight-factor transforms are
compiled once. Their online additions and scales disappear, but the persistent
checkpoint representation expands by approximately

```text
beta_g = padding_KN * R_g / (B_g*C_g)
         * transformed_word_bytes / BF16_word_bytes.
```

Offline construction is granted free, but every compiled byte, compressed cold
byte, and required original-BF16 repair-row side stream is charged. This flips
the assumption that an offline-transformed checkpoint is automatically smaller
or free to stream.

### C. Original-stream tile-resident transformation

At cut depth zero, the unchanged compressed BF16 checkpoint is streamed once
and all weight linear forms are generated while the source macro-tile is
resident. This avoids transformed-checkpoint expansion but charges every online
weight-transform addition, non-unit coefficient scale, form write, and favorable
one-pass scratch lower bound.

B and C are opposite endpoints of one exact information-flow tradeoff. The Gate
enumerates every cut depth for every retained mixed sequence.

## 3. Pinned explicit algorithm source

The external source is the official AlphaTensor standard-arithmetic catalog:

```text
repository: google-deepmind/alphatensor
commit: 1949163da3bef7e3eb268a3ac015fd1c2dbfc767
path: algorithms/factorizations_r.npz
Git blob SHA-1: 5ff45960f86da6237f105f78aaa82d29cb18c30e
```

For each catalog key whose leading dimensions are `(a,b,c)`, the workflow loads
three factors `U,V,W` and verifies the complete integer identity

```text
T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
```

against the symmetrized matrix-multiplication tensor

```text
T[i*b+j, j*c+k, k*a+i] = 1.
```

Only finite integral factorizations with rank `< a*b*c` and maximum absolute
coefficient at most two enter the search. Every eligible factorization also runs
deterministic integer matrix-product controls. A catalog hash mismatch,
reconstruction mismatch, or control mismatch invalidates the experiment.

## 4. Frozen 405B projection population

The registered tensor inventory is read from the committed EXP-071 artifact and
its Git blob identity is checked. The weight populations are:

```text
q_proj      16,384 × 16,384, count 126
k_proj       1,024 × 16,384, count 126
v_proj       1,024 × 16,384, count 126
o_proj      16,384 × 16,384, count 126
gate_proj   53,248 × 16,384, count 126
up_proj     53,248 × 16,384, count 126
down_proj   16,384 × 53,248, count 126
lm_head    128,256 × 16,384, count 1
```

The tied/standalone embedding lookup is excluded because it is not the dense
matrix multiplication under test. `lm_head` is included. EXP-099A did not
measure `lm_head`; the Gate therefore assigns it the maximum measured
projection-role p95 repair fraction, which is deliberately unfavorable relative
to using the global p95.

## 5. Explicit arithmetic ledger

For a target multiplication

```text
X[M,K] * W^T[K,N]
```

and a sequence of depth `d`, the runner charges:

1. all leaf scalar multiplications;
2. all leaf dot-product accumulation additions;
3. all left-factor additions and non-unit coefficient scales;
4. all online weight-factor additions and scales below the static cut;
5. all output-factor reconstruction additions and scales;
6. one logical write/move for each generated form and reconstructed block;
7. full native-order row-dot work for the role-p95 repair fraction.

The exact recurrence is evaluated level by level. At level `l`, parent product
count and remaining block dimensions determine how many times every factor
column or output row is applied. No asymptotic matrix-multiplication exponent is
credited and no non-constructive rank receives runtime credit.

The direct arithmetic ratio is

```text
r_direct = (leaf operations + explicit transforms + moves + repairs)
           / classical BF16 projection operations.
```

A second `free_transform_rank_oracle` grants every factor transform, scale,
packing operation, and transform workspace free. It retains only leaf
multiplication/addition work plus native repairs. It can show algebraic rank
headroom but cannot promote an executor.

## 6. Cold-byte ledger

For each family, raw BF16 bytes are

```text
S_raw = 2 * rows * columns * count.
```

The primary transformed stream is

```text
S_primary = S_raw * beta_g / c_lossless,
```

where `c_lossless=1.5` is a favorable frozen grant.

If `g>0`, native-order repairs require the original BF16 row, which is not
contained in the transformed representation. The Gate therefore adds

```text
S_repair = S_raw * repair_fraction / c_lossless.
```

At `g=0`, original rows are already in the primary stream and are not read a
second time. The per-token I/O fraction of a block of length `A` is

```text
r_io = sum_family(S_primary + S_repair)
       / (A * sum_family(S_raw)).
```

The joint dynamic program chooses one family plan while enforcing the frozen
p50 fraction

```text
r_io <= 1.2 * 4 / 405 = 0.011851851851851851.
```

The same 1.5× ratio is optimistically granted to transformed words and repair
rows. Actual transformed-form entropy is `NOT TESTED`.

## 7. Favorable workspace screen

The direct plan reports a favorable tiled lower-bound workspace containing:

- one activation block;
- one output block;
- one exact transformed activation leaf;
- the smaller of source-macro-tile retention or transformed-weight accumulators;
- the smaller of product-leaf retention or output-reconstruction blocks;
- compact factor metadata.

Only plans at or below 8 GiB survive the first screen. This is **not** a complete
8-GiB runtime ledger: KV cache, allocator reserve, code, decompressor state,
fragmentation, concurrent pipeline buffers, and the causal block generator are
not present and remain `NOT TESTED`.

## 8. Bounded search contract

The frozen search uses:

```text
block lengths: 32 ... 16,384
maximum mixed-recursion depth: 12
maximum padding multiplier per axis: 2
small coefficients: abs(value) <= 2
Pareto orientations retained: <=128
beam width: 192
per-split survivors: 2
static transformed word widths: 2 and 4 bytes
```

The beam retains both rank-favorable and transform-favorable candidates. This is
not exhaustive over all exact bilinear algorithms, all AlphaTensor algorithms,
or all mixed recursive sequences. A negative result closes only the frozen
catalog/language/search scope.

## 9. Candidate and successor-state grant

EXP-100A assumes a perfect future block:

```text
candidate nodes N = committed tokens A = block length.
```

No causal generator, branch tree, or successor-state construction is included.
The grant is intentionally maximally favorable so that a failure cannot be
blamed on speculative acceptance. A structural pass still requires a separate
causal block and exact successor-state Gate.

## 10. Frozen decisions

### Direct promotion

At least one complete-population joint plan must satisfy all of:

```text
explicit arithmetic ratio <= 10%
p50 cold-byte fraction <= 1.185185185...%
favorable FMM workspace <= 8 GiB
zero integrity failures
```

Decision:

```text
PROMOTE_CATALOGUED_RECTANGULAR_FMM_TO_FINITE_WORD_KERNEL_AND_CAUSAL_BLOCK_GATE
```

This is only a first 10× structural promotion. The final 4B-equivalent
arithmetic target remains approximately 1.185% p50 and 1.481% p95.

### Rank headroom only

If no explicit plan passes but the free-transform oracle passes the same 10×,
I/O, and workspace screens:

```text
RETAIN_RECTANGULAR_RANK_HEADROOM_REQUIRE_EXPLICIT_TRANSFORM_CIRCUIT
```

This says useful tensor rank exists but the known explicit transforms destroy
the saving.

### Rejection

If neither arm passes:

```text
REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE
```

### Invalid

Any catalog identity, exact tensor, deterministic execution, registered-input,
or search-population control failure yields:

```text
INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE
```

## 11. Stop rule

On rejection, do not rescue this scope by sweeping beam width, depth, block
length, coefficient cap, orientation order, I/O bins, transformed word size,
lossless ratio, padding allowance, repair percentile, or selected projection
roles.

Reopening requires at least one materially new source:

- a newly explicit tensor scheme with a demonstrated rank/transform constant
  outside the pinned catalog;
- a checkpoint-derived factor-transform circuit that eliminates the measured
  transform overhead rather than hiding it;
- a causal exact block source that changes the matrix aspect ratio and is fully
  charged;
- a fused finite-word kernel with measured operation/traffic behavior that
  invalidates this ledger.

## 12. Claim boundary

This experiment establishes only exact integer factorization validity and a
bounded projected resource ledger over registered 405B shapes. It does not load
TARGET-W, execute a Transformer layer, produce exact successor state, establish
finite-word transformed-weight equality, implement CUDA/SASS, demonstrate a
causal long block, physically allocate 8 GiB, or measure same-machine 4B and
405B latency. Those remain `NOT TESTED`.
