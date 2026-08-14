# E0 Biorthogonal Decomposition Cancellation Gate

Status: **STRUCTURALLY VALID / THREE MORE FIXED-LINEAR SHAPES REJECTED**

This is an E0 theorem and exact finite calculation.  It performs no model or
hardware run.  Structurally valid conditions were established.  Large-model
performance remains unverified.

## 1. Question

The previous determinantal Gate used only this implication:

```text
rank-one radius t  =>  rank-at-most-r radius r*t.
```

It treated the `r` selected representatives as if all their supports could be
disjoint.  The missing question is whether one fixed dictionary can keep them
disjoint across every minimal decomposition of every rank-`r` matrix.

## 2. The biorthogonal decomposition graph

Fix a binary rank-`r` matrix `M`.  After changing row and column bases, take
`M=I_r`.  A rank-one matrix `q=x*y^T` can occur in a minimal decomposition of
`M` exactly when

```text
y^T x = 1.
```

Two such terms can occur together exactly when their cross pairings vanish:

```text
y^T x' = 0,   y'^T x = 0.
```

They are then a partial biorthogonal frame and extend to a full biorthogonal
basis.  The resulting graph is the point-hyperplane anti-flag graph of
`PG(r-1,2)`.  It has

```text
N_r = (2^r-1) 2^(r-1)
d_r = (2^(r-1)-1) 2^(r-2)
lambda_min = -2^(3(r-2)/2).
```

The spectrum follows by writing the adjacency operator through the binary
point-hyperplane incidence matrix.  On functions of only the point or only
the hyperplane, the nonconstant singular value is `2^((r-2)/2)`, multiplied
by `2^(r-2)`.  On the row/column-zero complement the remaining eigenvalues
are `2^(r-3)` and `-2^(r-2)`.  Thus the displayed value is least for `r>=3`.
The executable Gate uses even `r`, so all values are exact integers.

## 3. Distinct representatives force activity

Choose one representative `f(q)` of weight at most `t` for every nonzero
rank-one query.  Representatives of distinct queries are distinct because
the atom map sends them to distinct matrices.

For the `N_r` vertices associated with one `M`, their total support activity
`L` is therefore at least the total weight obtained by filling the nonzero
Hamming shells in increasing order:

```text
L >= 1*C(S,1) + 2*C(S,2) + ...
```

with the last shell truncated after `N_r` words have been selected.

For atom coordinate `c`, let `A_c` be the graph vertices whose representative
uses that coordinate, and put `n_c=|A_c|`.  Hoffman's least-eigenvalue bound
and Cauchy--Schwarz give

```text
sum_c e(A_c)
  >= (d_r+s_r)/N_r * sum_c n_c^2 - s_r*L
  >= (d_r+s_r)L^2/(S*N_r) - s_r*L,

s_r = 2^(3(r-2)/2).
```

A uniformly random pair of positions in a uniformly random minimal
decomposition is a uniformly random graph edge.  Hence the expected total
number `I` of shared atom coordinates among all term pairs is at least

```text
C(r,2) * sum_c e(A_c)/(N_r*d_r).
```

Some decomposition attains the ceiling of this exact expectation lower
bound.

## 4. Shared coordinates cancel

If coordinate `c` occurs in `m_c` representatives, it contributes

```text
C(m_c,2)
```

pair intersections but loses

```text
m_c - (m_c mod 2)
```

incidences when the representatives are XORed.  A finite exact knapsack
inverts this relation conservatively.  Thus it converts the forced pair
intersection count into a forced cancellation `Delta` and replaces `r*t` by

```text
max((r-1)t, r*t-Delta)
```

for the complete rank-at-most-`r` set.

## 5. Decisive finite witnesses

For `30 x 40`, `S=1,404`, `t=14`, and `r=22`:

```text
minimum average representative weight = 4.981567223238774...
expected pair intersections            = 3.5230839509486778...
guaranteed integer pair intersections  = 4
forced XOR cancellation                = 4
forced rank-at-most-22 radius          = 304
Ball(1404,304) / RankLeq22(30,40)      = 0.05068730227558649...
```

The prior first survivor is therefore impossible.  The same exact Gate gives

```text
31 x 39, S=1,414, t=14: ratio 0.5723204167945867...
30 x 44, S=1,544, t=15: ratio 0.18141146098814276...
```

Combining the Fourier, ruling-preimage, ordinary rank-amplification,
mixed-weight, and new cancellation Gates rejects the first **846** rectangles
in the registered side-`1..128` exact subset-slack order.  The new first
unclosed case is:

```text
31 x 43, S=1,559, t=15, rank 24
guaranteed pair intersections = 5
forced cancellation           = 4
forced radius                 = 356
best ball/determinantal ratio = 3.409163670153519...
```

This is not a construction.

## 6. Claim boundary and next design action

The theorem covers a fixed binary linear atom map and any nonlinear algorithm
that merely locates a short XOR representative in that map.  It does not
cover adaptive word-valued probes, nonlinear output decoding, native
BF16/FP32 order, a shared 32-query physical union, or a complete Vortex
runtime.

The fixed-linear search should not return to `30 x 40`.  A continuation must
either exploit the `31 x 43` margin with a concrete atom/decomposer equation,
or change mechanism to a genuinely adaptive finite-word decoder.

## 7. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_biorthogonal_cancellation_gate.py `
  --output-dir results\e0_biorthogonal_cancellation_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_biorthogonal_cancellation_gate -v
```
