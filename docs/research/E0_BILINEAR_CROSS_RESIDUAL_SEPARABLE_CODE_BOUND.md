# E0 Bilinear Cross-Residual Separable-Code Bound

## Status and evidence boundary

- Date: 2026-08-09 Asia/Seoul
- Status: COMPLETE SCOPED E0 LOWER BOUND; NO SURVIVING CANDIDATE
- Evidence: exact algebra, finite sphere-covering argument, deterministic
  calculator, and eight independent reference/property controls
- Model/checkpoint/hardware action: none

This note does **not** prove that every exact bilinear data structure is
impossible. It closes one precisely declared extension of the Causal Residual
Atlas: matrix-local row and column linear covers whose only uncached term is a
raw-coordinate cross residual.

## Question

Can the missing term

```text
r^T W u
```

be made target-cheap by replacing the small Atlas bases with arbitrary exact
linear covering codes on the left and right, while retaining raw-checkpoint
repair only for the two coordinate-sparse residuals?

This is the strongest direct separable generalization of the post-Atlas
identity. It permits arbitrary linear codes, optimal nearest-codeword choices,
lossless compression of their image pair, and all eight GiB of hot state. It
grants lookup, decoding, image access, and all non-cross arithmetic for free.

## Declared model

For each binary matrix `W_i in F_2^(m_i x n_i)`, choose independent linear
subspaces

```text
A_i <= F_2^m_i,  dim(A_i) = a_i
B_i <= F_2^n_i,  dim(B_i) = b_i.
```

Every query direction is represented as

```text
r = p + e,  p in A_i
u = q + f,  q in B_i.
```

The exact identity is

```text
r^T W_i u
  = p^T W_i q
  + e^T W_i q
  + p^T W_i f
  + e^T W_i f.                                           (1)
```

The structure stores enough checkpoint-derived information to make the first
three terms free. Only the last term probes raw matrix coordinates, so its
worst-case coefficient work is at least `wt(e) * wt(f)`.

The full scoped contract is deliberately stronger than a real Transformer
trace: each matrix receives an independently selectable pair from the
Cartesian product of its direction spaces. This makes a model-wide sum valid.
Whether a particular causal Transformer can reach those independent worst
cases is **NOT TESTED**. Cross-matrix shared advice, nonlinear encodings,
word-packed probe algorithms, and causally restricted query populations are
outside this lower bound.

## Why the binary restriction is favorable and valid in scope

An executor claiming this all-query primitive for arbitrary Q4 checkpoints
must also handle the binary subclass. We therefore reduce every coefficient
and direction to `F_2` and still grant the complete `8 GiB` hot allowance to
that one bit plane. This is more favorable than charging four Q4 bit planes,
BF16 direction values, KV state, runtime state, or numerical verification.

The comparison is a logical **coefficient-probe** comparison. It is not a
physical theorem about SIMD words, cache lines, PCIe transactions, or SSD
requests.

## Lemma 1 -- lossless image-pair rank

The linear map

```text
W_i -> (A_i^T W_i, W_i B_i)
```

has rank

```text
s_i = a_i*n_i + m_i*b_i - a_i*b_i.                       (2)
```

The two images have ranks `a_i*n_i` and `m_i*b_i`; their shared
`A_i^T W_i B_i` image has rank `a_i*b_i`. A change of bases reduces the map to
overlapping row and column coordinates, making (2) exact. Consequently even
an arbitrary lossless encoding of the pair needs at least `s_i` bits over the
arbitrary binary checkpoint population.

The tests independently construct the small coordinate maps and reproduce
their GF(2) ranks; they do not call the formula under test.

Write the row/column rates as `x_i=a_i/m_i` and `y_i=b_i/n_i`. The matrix-local
side fraction is therefore

```text
sigma_i = s_i/(m_i*n_i) = x_i + y_i - x_i*y_i.            (3)
```

## Lemma 2 -- finite covering-radius witness

A binary code with at most `2^k` codewords and covering radius `R` in length
`d` must satisfy the elementary sphere-covering condition

```text
2^k * sum(j=0..R, C(d,j)) >= 2^d.                         (4)
```

For `R/d <= 1/2`, the binomial volume is at most
`2^(d*H_2(R/d))`. The rational witness used here is

```text
code rate threshold                 3/10
residual-radius witness             3/16
H_2(3/16)                           0.6962122601251458
3/10 + H_2(3/16)                    0.9962122601251457 < 1.
```

Thus any code of rate at most `3/10` that covers the complete binary direction
space has worst-case relative radius strictly greater than `3/16`. This is a
finite volume contradiction with an explicit numerical margin, not an
asymptotic `Omega` statement.

## Theorem -- registered 405B coefficient frontier

Use the frozen non-embedding coefficient population

```text
D                                      403,747,897,344
complete hot grant                     68,719,476,736 bits
hot bits per binary coefficient        16,384 / 96,261
                                       17.0203924746%
p50 target alpha                       8 / 675
                                       1.18518518519%.
```

From (3), `x_i <= sigma_i` and `y_i <= sigma_i`. Applying Markov's inequality
with coefficient counts as weights proves that at least

```text
1 - (16,384/96,261)/(3/10)
  = 124,943 / 288,783
  = 43.2653584179%                                      (5)
```

of the coefficient population belongs to matrices with `sigma_i <= 0.3`.
Both code rates are then at most `0.3`, so Lemma 2 supplies left and right
residual fractions above `3/16`. Under the Cartesian all-query contract, choose
their two worst-case directions together. Even after granting every other
operation for free, the model-wide raw cross work is at least

```text
(124,943/288,783) * (3/16)^2
  = 124,943 / 8,214,272
  = 1.52104775688%
  = 6,141,198,336 raw coefficient probes.                (6)
```

The complete p50 allowance is only

```text
(8/675) * D = 4,785,160,264.817778 coefficient uses.     (7)
```

Equation (6) is `1.28338404487x` (7). It has already spent the whole `8 GiB`
on binary code images and charged none of the following:

- center lookup or nearest-codeword decoding;
- the three cached terms in (1);
- reading the stored images;
- selectors, indices, metadata, verification, or fallback;
- BF16/Q4 numerical semantics;
- KV cache, runtime state, or physical transfers.

Therefore no positive cost can rescue this scoped model.

Solving only this witness for the hot-state point where it stops rejecting
gives `9.34710279225 GiB`. That is already above the complete `8 GiB` target,
but it is **not** a sufficient construction at 9.35 GiB; it is only where this
particular conservative proof ceases to decide.

## Literature boundary

[Ramamoorthy and Rashtchian, ITCS 2020](https://doi.org/10.4230/LIPIcs.ITCS.2020.35)
prove an exact equivalence between systematic linear inner-product data
structures and rigid query sets, and derive a finite-constant rigidity lower
bound for binary rank-one queries. They also state that it remains unknown
whether a linear-dimensional subspace places all rank-one queries at
sublinear Hamming distance. Their general cell-probe result is much weaker
than a dense-fraction lower bound at the registered dimensions.

[Chakraborty, Kamma, and Larsen, STOC 2018](https://doi.org/10.1145/3188745.3188830)
give systematic succinct tradeoffs for Boolean matrix-vector and `F_2`
vector-matrix-vector queries. As already recorded by EXP-071, their redundancy
range and the absence of a jointly preprocessed Transformer direct-sum theorem
do not certify the VORTEX objective impossible.

The proof in this note does not claim to improve those general theorems. It
obtains a finite model-wide number only by declaring matrix-local separable
images and an independent Cartesian query tuple. Removing either declaration
returns to the open general data-structure frontier.

## Decision

```text
REJECT_MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE_AS_CORE
DO_NOT_REOPEN_ATLAS_WITH_LARGER_OR_COVERING_BASES
KEEP_NONSEPARABLE_GLOBAL_OR_CAUSALLY_RESTRICTED_SOURCE_OPEN
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_MODEL_HARDWARE_WORK
```

This closes arbitrary linear covering-code upgrades to the primal/dual Atlas
under the declared all-query coefficient-probe model. It does not reject an
exact nonlinear joint code, cross-matrix shared advice, a word-packed data
structure with a valid physical equation, or a causal restriction proven on
real checkpoint traces.

## Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_bilinear_cross_residual_frontier.py `
  --output-dir results\e0_bilinear_cross_residual_frontier

.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_bilinear_cross_residual_frontier.py
```

Expected focused result: `8 passed`. Authority:

```text
results/e0_bilinear_cross_residual_frontier/summary.json
results/e0_bilinear_cross_residual_frontier/checksums.sha256
```

