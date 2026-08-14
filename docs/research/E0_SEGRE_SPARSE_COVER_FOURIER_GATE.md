# E0 Segre Sparse-Cover Fourier Gate

## Verdict

```text
REJECT THE TIGHTEST NEAR-CAPACITY RECTANGULAR SPARSE COVERS
REJECT RAW SUBSET COUNTING AS A CONSTRUCTION
REQUIRE LOW-WEIGHT KERNEL DEFECTS IN EVERY REMAINING LINEAR-DICTIONARY ROUTE
KEEP THE FIRST SLACK CASE (13 x 89) OPEN BUT UNCONSTRUCTED
KEEP ADAPTIVE FINITE-WORD DECODERS OUTSIDE THIS THEOREM
NO SURVIVING CANDIDATE
```

No checkpoint, model, download, backend, server, or hardware action occurred.
The result is an exact binary theorem for listed parameters, not a native
runtime and not a universal cell-probe lower bound.

## Elementary explanation

The previous count said that 619 boxes have just enough six-box combinations
to give every answer in a `23 x 23` puzzle. But “enough combinations” ignored
their signs.

If all answers really fit, then every rank-one test light must see almost all
619 boxes agree: its signed total must have magnitude at least 527. Squaring
and averaging those totals gives at least `527^2 = 277,729`.

The same average can be computed by pairs of boxes. Equal boxes correlate by
one. Unequal boxes correlate with a random rank-one test by less than one
half. The boxes must span all 529 puzzle directions, so even after placing all
90 extras on one duplicate there are at most 8,809 equal ordered pairs. The
largest possible squared average is only

```text
195,984.95537375606...,
```

contradicting 277,729. Thus the apparently perfect six-read code does not
exist. This is a proof about the strongest tight case, not a random-search
failure.

## 1. Cover and Fourier equations

Let `V=F_2^(a x b)` and let arbitrary atoms `g_1,...,g_S` define

```text
phi(x) = XOR(j:x_j=1, g_j).
```

Assume every rank-at-most-one matrix has a representative of weight at most
`t`:

```text
Segre(a,b) subset phi(Ball(S,t)).                         (1)
```

The query count and ball size are

```text
Q = 1 + (2^a-1)(2^b-1),
B = sum(j=0..t,C(S,j)).                                  (2)
```

Choose one distinct representative for every query. For a dual matrix `M`,
the complete ball character sum differs from the selected-query sum by at
most the `B-Q` leftover subsets. If

```text
c_M[j] = <M,g_j>,       k_M = weight(c_M),
```

then the exact necessary condition is

```text
| sum(j=0..t,K_j(k_M;S)) - Rhat_rank(M) | <= B-Q,         (3)
```

where `K_j` is the binary Krawtchouk polynomial. The rank-one-set transform is

```text
Rhat_r = 2^(a+b-r) - 2^a - 2^b + 2.                     (4)
```

Equations (3) and (4) are integer identities; no asymptotic approximation or
sample is used.

## 2. Exact `23 x 23` contradiction

The favorable registered parameters are

```text
a=b                                      23
D=ab                                    529
S=floor((D_global+8 GiB bits)/D_global*D) 619
t=floor((8/675)D)                          6
Q                              70,368,727,400,450
B                              77,004,073,569,046
```

For every nonzero rank-one dual `M`, evaluating (3) at every integer
`0 <= k <= 619` leaves only

```text
k_M in [30,46] union [574,590].                           (5)
```

Therefore, for

```text
Y_M = sum_j (-1)^<M,g_j> = S-2k_M,
```

equation (5) forces

```text
|Y_M| >= 527,
E_M Y_M^2 >= 527^2 = 277,729.                            (6)
```

On the other hand,

```text
E_M Y_M^2 = sum_(j,k) beta_rank(g_j+g_k).                (7)
```

For a nonzero matrix difference, the maximum correlation occurs at rank one:

```text
beta_1 = 35,184,355,311,617 / 70,368,727,400,449
       < 1/2.                                            (8)
```

Because the rank-one masks include every matrix unit, (1) forces the atoms to
span all `D=529` dimensions. At least 529 distinct nonzero atoms are necessary.
Concentrating all 90 extra copies on one atom maximizes equal ordered pairs:

```text
E_equal <= (S-D+1)^2 + D-1 = 8,809.                     (9)
```

Combining (7)--(9) gives

```text
E_M Y_M^2
 <= 8,809 + (619^2-8,809) beta_1
  = 195,984.95537375606... < 277,729,                  (10)
```

contradicting (6). This covers arbitrary atom matrices, duplicates, and every
possible decoder that returns a subset of at most six of these fixed linear
cells. It does not assume random atoms or a systematic source.

## 3. Rectangular result and the first unclosed case

The same exact Gate rejects the best raw-capacity candidates below:

| shape | `D` | `S` | `t` | `B/Q` | verdict |
|---|---:|---:|---:|---:|---|
| `14 x 56` | 784 | 917 | 9 | 1.0392060 | reject |
| `21 x 25` | 525 | 614 | 6 | 1.0421958 | reject |
| `22 x 24` | 528 | 617 | 6 | 1.0732002 | reject |
| `23 x 23` | 529 | 619 | 6 | 1.0942940 | reject |
| `15 x 47` | 705 | 824 | 8 | 1.1156201 | reject |
| `12 x 122` | 1,464 | 1,713 | 17 | 1.1341541 | reject |
| `13 x 81` | 1,053 | 1,232 | 12 | 1.2335939 | reject |
| `17 x 37` | 629 | 736 | 7 | 1.2642945 | reject |
| `14 x 64` | 896 | 1,048 | 10 | 1.4093795 | reject |

Within the deterministic side range `1..128`, sorting capacity-feasible
rectangles by exact rational `B/Q` rejects the first nine and makes `13 x 89`
the tenth and first case not closed by this second moment:

```text
D=1,157, S=1,353, t=13, B/Q=1.5370839...
```

Its pointwise Fourier interval reaches the center (`min |S-2k|=1`), so (6)
provides no contradiction. This does not construct the code. It identifies
the next mathematical target without pretending the current proof covers it.

## 4. Every remaining cover needs short kernel defects

There is a separate structural necessity. Fix one ruling of the Segre set,
which is a linear subspace of dimension `max(a,b)`. Choose one weight-`t`
representative `f(x)` for every point. For any `x,y`,

```text
f(x)+f(y)+f(x+y) in ker(phi)
```

has weight at most `3t`. If the kernel minimum distance were greater than
`3t`, every defect would vanish and `f` would be linear. A binary linear space
whose every vector has weight at most `t` has support strictly below `2t`, so
its dimension is below `2t`.

Hence whenever the ruling dimension is at least `2t`, any cover requires

```text
d_min(ker(phi)) <= 3t.                                  (11)
```

For `13 x 89, t=13`, a survivor needs a kernel relation of weight at most 39.
Random high-girth atoms therefore cannot be promoted. The remaining code must
use deliberately structured short relations and still decode every rank-one
query.

## 5. Claim boundary and next Gate

This theorem rejects the tight low-slack sparse-cover cases, not all block
shapes. It does not cover:

- the unconstructed `13 x 89` or higher-slack linear dictionaries;
- data-dependent adaptive addresses or arbitrary word decoders;
- native Q4/BF16/FP32 accumulation and rounding;
- a shared physical atom union for 32 causal queries;
- state, KV, metadata, scheduling, verification, and repair.

The cheapest next linear-code task is therefore not another random search. It
is to combine the mandatory weight-at-most-39 kernel relations with (3) for
the `13 x 89` case, or construct its atoms and a sub-dense decomposer. Failure
to do either keeps `NO_SURVIVING_CANDIDATE`.

## 6. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_segre_sparse_cover_fourier_gate.py `
  --output-dir results\e0_segre_sparse_cover_fourier_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_segre_sparse_cover_fourier_gate -v
```
