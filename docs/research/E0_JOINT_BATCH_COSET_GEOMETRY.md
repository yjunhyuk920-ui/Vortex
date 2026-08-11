# E0 Joint-Batch Coset Geometry

## Verdict

```text
PROMOTE SEGRE WEIGHT HIERARCHY AS EXACT BATCH GEOMETRY
REJECT LITERAL FACTOR-ENVELOPE CATALOG
REJECT COUNTING-ONLY PROMOTION
KEEP IMPLICIT JOINT FACTOR-ENVELOPE GENERATOR OPEN
NO SURVIVING CANDIDATE
```

This is a theorem and cheapest-possible structure Gate. No checkpoint, model,
backend, download, server, or hardware action occurred.

## Elementary explanation

One rank-one question is made from two arrows:

```text
left arrow r  x  right arrow u.
```

For 32 questions, all left arrows together occupy at most 32 dimensions and
all right arrows together occupy at most 32 dimensions. Therefore every
question is inside one common box with at most

```text
32 x 32 = 1,024
```

basis slots. This is real shared structure; treating the 32 questions as
unrelated loses it.

The box cannot simply be cached in advance. At a registered 16,384-wide
matrix there are more than `2^1,046,528` independently selectable pairs of
32-dimensional left and right spaces. A literal cache would need 1,024 exact
summaries for each pair.

The remaining useful question is therefore precise:

> Can one near-source-size shared encoding generate the requested 1,024-slot
> restriction on demand, without reading the dense source?

No such generator is constructed here.

## 1. Exact Segre/product-code identity

Over `F_2`, every nonzero rank-one `m x n` mask is a projective point

```text
r tensor u,  r != 0, u != 0.
```

The complete point set is the Segre product of the projective systems of two
binary simplex codes `S_m` and `S_n`. If `V` is a `t`-dimensional binary
matrix subspace, projective-system duality gives

```text
max_dim(V)=t |V intersect RankOne(m,n)|
  = (2^m-1)(2^n-1) - d_(mn-t)(S_m tensor S_n).             (1)
```

Binary simplex generalized weights are

```text
d_i(S_m) = 2^m - 2^(m-i).
```

Schaathun proves the exact product-code weight formula when both factors
satisfy the chain condition; simplex codes do. Substituting their weight
hierarchies into that theorem makes (1) finite and exact.

Primary source: Hans Georg Schaathun, *The Weight Hierarchy of Product
Codes*, IEEE Transactions on Information Theory 46(7), 2000,
<https://doi.org/10.1109/18.887876>.

## 2. Independent exhaustive confirmation

The throwaway prototype at commit `0206a08` on branch
`prototype/rank-one-dictionary-small-search` enumerates every binary subspace
via its unique reduced-row-echelon basis. It checks 418,199 subspaces for the
`2 x 4` case alone. The exact maxima are:

| shape | maximum rank-one points as subspace dimension grows |
|---|---|
| `2 x 2` | `0,1,3,5,9` |
| `2 x 3` | `0,1,3,7,9,13,21` |
| `2 x 4` | `0,1,3,7,15,17,21,29,45` |

Every value agrees with (1). The maximizers have the compression/Ferrers
shape seen in the prototype: fill one row or column, then extend the next
strip. For a square of side `n`, the first two strips have the closed form

```text
M_n(t) = 2^t - 1,                         0 <= t <= n
M_n(t) = 2^n + 2^(t-n+1) - 3,             n <= t <= 2n.  (2)
```

Thus compression spaces are not merely a plausible example; in this regime
they attain the largest possible rank-one intersection.

## 3. Exact joint factor envelope

For a batch `q_i=r_i u_i^T`, define

```text
R = span(r_1,...,r_K),
U = span(u_1,...,u_K).
```

Then

```text
span(q_1,...,q_K) subseteq R tensor U,
dim(R tensor U) <= min(m,K) min(n,K).       (3)
```

At `K=32`, equation (3) gives 1,024 dimensions. It does not say the 1,024
checkpoint summaries are already stored. It identifies exactly what a new
on-demand restriction generator would have to return.

The number of 32-subspaces of `F_2^n` is the Gaussian binomial
`[n choose 32]_2`, which is greater than `2^(32(n-32))`. Choosing left and
right spaces independently at `n=16,384` therefore gives strictly more than

```text
2^(2*32*(16,384-32)) = 2^1,046,528         (4)
```

factor-envelope names. Equation (4) rejects a literal envelope catalog, not
an implicit shared code.

## 4. Why the sharper geometry still does not prove the target impossible

Grant one registered square every atom from the complete model-wide binary
source plus all 8 GiB advice:

```text
S = 405,849,243,648 + 68,719,476,736
  = 474,568,720,384 atoms.
```

A support of `t < 2n` atoms has a span containing fewer than `2^(n+1)`
rank-one masks by (2). The number of supports of size at most `t` is at most

```text
(eS/t)^t.
```

Meanwhile the ordered 32-query universe has more than
`2^(32*(2n-2))` batches. Substitution at `n=16,384` proves only

```text
t >= 20,218 bit atoms.                       (5)
```

With impossibly favorable packing, (5) is just 316 64-bit words or 2,528
bytes. It is a valid rank-one-specific covering bound, but it is many orders
of magnitude too weak to decide the physical VORTEX Gate. Promoting it into a
general impossibility statement would be incorrect.

## 5. Consequence

Three routes are now separated cleanly.

1. Independent minimum coset leaders are already rejected by their near-full
   tiny batch unions.
2. Literal catalogs of `R tensor U` restrictions are rejected by (4).
3. Support-counting lower bounds are rejected as a decision method by (5).

The one surviving mathematical interface is not a candidate implementation:

```text
checkpoint compiler -> near-source-size shared code
(R,U) query          -> implicit addresses for W restricted to R tensor U
restriction          -> exact paired answers with native accumulation order.
```

It must also charge generator metadata, address construction, physical
probes, native value width, decoding, KV/state, verification, and fallback.
Until that object and a favorable complete 405B equation exist, the status
remains `NO_SURVIVING_CANDIDATE`.

## 6. Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_joint_batch_coset_geometry.py `
  --output-dir results\e0_joint_batch_coset_geometry

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_joint_batch_coset_geometry -v
```

Authority:

```text
results/e0_joint_batch_coset_geometry/summary.json
results/e0_joint_batch_coset_geometry/checksums.sha256
```
