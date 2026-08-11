# E0 Determinantal Rank-Amplification Gate

## Verdict

```text
PROMOTE RANK AMPLIFICATION AS A MANDATORY SPARSE-COVER GATE
PROMOTE MIXED-WEIGHT REPRESENTATIVE AVERAGING AS A MANDATORY GATE
REJECT THE FIRST 770 CAPACITY-FEASIBLE RECTANGLES EXACTLY
PROMOTE 30 x 40, S=1,404, t=14 AS THE NEXT LINEAR CONSTRUCTION TARGET
KEEP ADAPTIVE FINITE-WORD DECODERS OUTSIDE THIS THEOREM
NO SURVIVING CANDIDATE
```

No checkpoint, model, download, backend, server, or hardware action occurred.
This is an exact binary theorem for fixed linear atoms. It is not an atom
construction, native numerical lift, or general adaptive cell-probe lower
bound.

## Elementary explanation

Suppose every one-line answer can be made with at most eight blocks. Any
two-line answer is the sum of two one-line answers, so it must be makeable with
at most sixteen blocks. The same rule continues:

```text
rank at most r  =>  at most r*t blocks.
```

The earlier `17 x 43` candidate has enough eight-block combinations for all
rank-one answers. But sixteen blocks give only

```text
3,450,282,117,207,738,592,915,369,522,984,017
```

patterns, while there are

```text
221,532,928,720,799,928,211,216,887,624,892,418
```

rank-zero, rank-one, or rank-two matrices. It misses by a factor of about 64.
Therefore that dictionary cannot exist, regardless of how cleverly its atoms
or decoder are chosen.

## 1. Exact rank amplification

Let arbitrary atoms `g_1,...,g_S` define the surjective binary map

```text
phi(x) = XOR(j:x_j=1,g_j).
```

Assume every rank-one matrix has a representative of Hamming weight at most
`t`. Every rank-`r` matrix over a field is a sum of `r` rank-one matrices.
XORing their representatives gives weight at most `rt`, hence

```text
RankLeq(r) subset phi(Ball(S,rt)).                        (1)
```

The exact number of `a x b` binary matrices of rank `r` is

```text
N_r(a,b) = product(i=0..r-1)
  ((2^a-2^i)(2^b-2^i))/(2^r-2^i).                       (2)
```

Equations (1) and (2) give the mandatory capacity inequalities

```text
sum(j=0..rt,C(S,j)) >= sum(i=0..r,N_i(a,b))              (3)
```

for every `1 <= r <= min(a,b)`. This is not the old rank-one subset count;
it tests every determinantal rank forced by closure under addition.

For `17 x 43`, `S=855`, `t=8`, (3) first fails at `r=2`:

```text
Ball(855,16) / RankLeq2(17,43)
  = 0.01557457908009767...
```

The earlier `18 x 36`, `S=758`, `t=7` case also fails at rank two with ratio
`0.003959624495469352...`.

## 2. Mixed-weight representative averaging

Some more balanced shapes survive (3). A second exact necessity avoids the
mistake of granting each representative weight a different favorable
subspace.

Choose one distinct subset `X_q`, `1 <= |X_q| <= t`, for every nonzero
rank-one query `q`. Sample `s` of the `S` atom positions uniformly. A
weight-`k` representative is contained with probability

```text
C(s,k)/C(S,k).
```

For `s<S`, this probability decreases with `k`. Therefore every
weight-at-most-`t` representative is contained with probability at least

```text
C(s,t)/C(S,t).                                           (4)
```

All contained answers lie in the span of the sampled atoms, whose dimension
is at most `s`. The exact product-simplex weight hierarchy gives the largest
number `M(a,b,s)` of nonzero rank-one matrices in such a subspace. For
`a<=b`, write `s=qb+r`, `0<=r<b`; then

```text
M(a,b,s) = (2^q-1)(2^b-1)+2^q(2^r-1).                  (5)
```

Combining (4) and (5) yields, for every `t<=s<=ab`,

```text
((2^a-1)(2^b-1)) C(s,t)/C(S,t) <= M(a,b,s).              (6)
```

This one inequality handles all representative weights simultaneously. It
does not assume that every representative actually has weight `t`; shorter
ones are sampled more often and consume more of the same bound.

For `32 x 39`, `S=1,460`, `t=14`, the strongest witness is `s=775`:

```text
canonical representative upper  2,228,227,201,975,458,712,467
required nonzero queries         2,361,183,240,880,771,825,665
ratio                            0.9436909272421747...
```

Thus adding weight-13 or smaller representatives cannot repair the gap.

## 3. Exact combined scan

All rectangles with sides `1..128` were sorted by the exact original
rank-one subset-slack ratio. Among the 6,543 shapes that pass the raw
rank-one count:

- 766 of the first 770 fail a determinantal rank-amplification inequality;
- the other four (`32 x 39`, `33 x 38`, `34 x 37`, `35 x 36`) fail (6);
- the 771st and first unclosed shape is `30 x 40`.

The rank at which (3) first fails varies from two through 21, showing why a
rank-two-only check would still be incomplete.

The new frontier is

```text
a=30, b=40, D=1,200, S=1,404, t=14, kernel dimension=204.
```

Its closest rank-amplification condition occurs at rank 22 and still has
ratio

```text
Ball(1404,308) / RankLeq22(30,40)
  = 8.417418510567584...
```

Its strongest mixed-weight witness is `s=795`, with upper/required ratio
`1.5198090797208368...`. The prior ruling-preimage and Fourier Gates also do
not reject it. These are passed necessities, not an atom family.

## 4. Why this advances the construction search

The previous frontier asked for one ordinary covering code induced by a
ruling. Rank amplification exposes a much stronger global closure rule: any
candidate must cover every determinantal rank at the linearly amplified
radius. This removes 770 favorable capacity shapes without random atom search
or a checkpoint experiment.

The `30 x 40` case is deliberately not promoted. It still lacks:

- 1,404 explicit atoms spanning 1,200 binary matrix dimensions;
- a decoder producing at most 14 atoms for every rank-one matrix;
- a small physical union for 32 causal queries;
- native Q4/BF16/FP32 reference-exact lifting;
- state, KV, metadata, scheduling, and complete latency accounting.

The next fixed-linear task is to exploit the rank-22 closure, not to run a
random dictionary search. A useful next obstruction must couple the many
rank-one decompositions of the same higher-rank matrix or bound
determinantal-variety intersections inside sampled atom spans. Otherwise the
mechanism class should change to a genuinely adaptive word decoder.

## 5. Claim boundary

The proof permits arbitrary binary matrix atoms, duplicates, and arbitrary
subset-location logic. It assumes that the returned answer is the XOR of a
fixed atom subset of size at most `t`. It does not cover adaptive word-valued
probes, nonlinear output decoders, native floating-point execution, or a
complete Vortex runtime.

`30 x 40` is an unconstructed mathematical frontier. `NO SURVIVING CANDIDATE`
remains correct.

## 6. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_determinantal_rank_amplification_gate.py `
  --output-dir results\e0_determinantal_rank_amplification_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_determinantal_rank_amplification_gate -v
```
