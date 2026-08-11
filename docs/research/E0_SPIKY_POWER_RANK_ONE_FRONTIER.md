# E0 Spiky / Entrywise-Power Rank-One Frontier

Status: both direct constructors rejected by their cheapest finite Gate;
the general nonlinear rank-one probe gap remains open.

## Scope

This audit does not use the old `2.5%` premise.  It asks whether two exact,
globally nonlocal checkpoint representations can fit the unchanged
405B/8 GiB/20 ms contract before any model or hardware work:

1. `OMEGA-SPIKECUT`: a sum of spiky matrices, where each component is a
   collection of disjoint rectangles and is rank one inside every rectangle;
2. `OMEGA-POWERFOLD`: an entrywise integer power of a low-rank root.

Both candidates receive optimal preprocessing, exact real arithmetic, free
selection, free block reductions, free metadata/storage/traffic, and free
verification/fallback.  Native BF16/FP32 order is not claimed.

## OMEGA-SPIKECUT

For one component

```text
S = B .* (a b^T),
```

let the nonzero diagonal blocks of `B` be `I_k x J_k`.  Its scalar query is

```text
r^T S u = sum_k <r[I_k], a[I_k]> <b[J_k], u[J_k]>.
```

This is broader than ordinary low rank: one component may contain many
independent diagonal rank-one blocks.  It is nevertheless a static direct
evaluator.  Let `L` be the total number of active row- and column-factor
coordinates over every component.  Even with every block combine free, an
arbitrary query needs `L` factor/query interactions.

### Finite description Gate

For an `N x N` sign matrix and a fixed `L`:

1. for one component with `a` active rows and `b` active columns, choosing
   supports and matching their set partitions into blocks gives
   `C(N,a) C(N,b) sum_k S(a,k) S(b,k) k! <= (2 e N)^(a+b)` possibilities;
2. composing any number of nonempty components with total incidence `L`
   then bounds all block-mask sequences by `2 (4 e N)^L`;
3. after masks are fixed, every output cell is a degree-two polynomial in at
   most `L` real variables;
4. Warren's theorem bounds their sign patterns by `(8 e N^2/L)^L`.

Therefore no more than

```text
2 (32 e^2 N^3/L)^L
```

of the `2^(N^2)` sign matrices have a direct representation of factor
incidence at most `L`.

The authoritative Gate embeds all 883 registered matrices into disjoint
global row/column coordinate universes.  It permits one component to couple
blocks across matrices and ignores invalid cross-matrix cells.  With `P`
registered sign coefficients, maximum global coordinate dimension `G`, and
total factor incidence `L`, the same proof gives

```text
2 (32 e^2 G P/L)^L
```

possible sign checkpoints.  One factor interaction receives an ideal FMA, so
the complete `9.6 GFLOP/token` grant permits `L=4,800,000,000`.

Registered model-wide substitution:

```text
P non-embedding coefficients              403,747,897,344
matrix instances                                         883
global row / column coordinates      19,997,952 / 19,111,936
complete factor-incidence allowance            4,800,000,000
log2(all sign checkpoints)                   403,747,897,344
log2(representable upper)                     184,958,474,578.07
hard-checkpoint exponent                      218,789,422,765.93 bits
```

For comparison, the single-square diagnostic is:

```text
N                                      16,384
complete work fraction                 1.1827051732%
allowed factor incidences L            3,174,800
log2(all sign matrices)                268,435,456
log2(representable upper)              generated in summary.json
hard-instance exponent                 178,629,392.76 bits
```

Thus there are complete bounded-word `+1/-1` checkpoints for which the direct
spiky evaluator cannot fit even the full compute allowance.  This is an
existence/counting rejection for an arbitrary-checkpoint guarantee, with
global budget reallocation and cross-matrix masked components already
granted, before block arithmetic, storage, traffic, or native-order repair.

As a separate diagnostic, the 2026 paper proves that a random Boolean
`N x N` matrix has spiky rank at least `N/(12 log_2 N)` with probability at
least `1-2^(-N^2/2)`.  If every factor were dense, this alone gives

```text
2 spr(W)/N >= 1/(6 log_2 N) = 1/84 = 1.1904761905%,
```

which is `1.0065705x` the complete allowance.  The resource-sensitive
description Gate above is authoritative because it also grants sparse factor
supports.

Decision:

```text
REJECT_OMEGA_SPIKECUT_DIRECT_EVALUATOR_BY_MODEL_WIDE_DESCRIPTION_GATE
```

Primary source: Hambardzumyan, Myasnikov, Riazanov, Shirley, and Shraibman,
"Spiky Rank and Its Applications to Rigidity and Circuits," ECCC TR26-030,
2026: <https://eccc.weizmann.ac.il/report/2026/030/>.

## OMEGA-POWERFOLD

Suppose for a positive integer `p` that

```text
Z = A B^T, rank(Z)=r, and W = Z^(entrywise p).
```

The multinomial theorem gives an exact separable expansion with

```text
K = binom(r+p-1, p)
```

rank-one terms.  Consequently `r_query^T W u_query` is just an ordinary
static rank-`K` evaluation.  At `N=16,384`, even granting dense factor terms
and every other cost free, the complete allowance permits at most `K=96`:

```text
p    maximum root rank    expanded terms
2             13                91
3              7                84
4              5                70
5              4                56
```

The next rank exceeds 96 in every row.  This does not open a new execution
class: after exact expansion it is the already rejected static low-rank
normal form.  The cited EPMF model also factors nonnegative magnitudes; an
arbitrary signed Transformer matrix still needs its sign information, and
neither that paper nor this audit preserves native accumulation order.

Decision:

```text
REJECT_OMEGA_POWERFOLD_AS_RELABELED_STATIC_LOW_RANK_EXECUTION
```

Primary source: Gillis, Saha, Sicilia, and Vandaele, "On the Complexity of
Low-Rank Matrix Signing and Entrywise Power Matrix Factorization," arXiv
2607.04875, 2026: <https://arxiv.org/abs/2607.04875>.

## Claim boundary and next direction

The finite description argument rejects the named direct sum-of-spiky
evaluator.  It does not cover a different algorithm that preprocesses the
factors with a further nonlinear adaptive data structure.  The power result
rejects the exact multinomial evaluator, not every entrywise nonlinear
representation.

No model forward, checkpoint mutation, experiment number, backend, kernel,
download, Ubuntu command, or hardware action was used.  The active ticket
therefore remains claimed:

```text
NO_SURVIVING_CANDIDATE
GENERAL NONLINEAR RANK-ONE PROBE GAP OPEN
TARGET NOT ACHIEVED
```

## Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_spiky_power_rank_one_frontier.py `
  --output-dir results\e0_spiky_power_rank_one_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_spiky_power_rank_one_frontier -v
```
