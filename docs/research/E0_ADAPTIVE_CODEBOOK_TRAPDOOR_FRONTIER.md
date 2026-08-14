# E0 Adaptive Codebook / Trapdoor Frontier

Status: two new exact-query proposals rejected by their first necessary Gate;
the general compressed adaptive bilinear oracle remains open.

## Scope

This audit does not use the old `2.5%` premise. It derives its favorable
coefficient-work screen from the registered p50 target `8/675` and asks two
different questions:

1. can a nonlinear router cache exact answers for representative rank-one
   queries and repair only the coefficient positions on which the requested
   query differs; and
2. can a fast trapdoored random matrix mask an arbitrary checkpoint so that
   the checkpoint product becomes fast by subtraction?

No model forward, checkpoint mutation, experiment number, backend, kernel,
download, Ubuntu command, or hardware action is used.

## OMEGA-NEARESTPAIR

Work first over `F_2`, a favorable exact subclass. For a requested query mask

```text
Q = r tensor u
```

and a cached representative mask `C`, a literal table stores the one-bit
answer `<W,C>`. Exact repair is

```text
<W,Q> = <W,C> + <W,Q + C>  over F_2.                 (1)
```

The direct implementation reads precisely the checkpoint cells on which
`Q` and `C` differ. Routing, representative storage, table lookup, address
generation, and every non-read operation are granted free. Representatives
may be arbitrary joint masks; they need not be rank one or arise from
separate left/right codebooks.

### Finite packing Gate

Let `R` be the maximum repaired cells. Restrict the legal query family to

```text
Q_r = r tensor 1,  r in F_2^N.
```

For two such masks,

```text
dist(Q_r, Q_s) = N * dist(r,s).                       (2)
```

Set

```text
d = floor(2R/N).
```

A greedy binary packing contains at least

```text
2^N / sum(j=0..d, C(N,j))                             (3)
```

vectors with pairwise Hamming distance greater than `d`. Their corresponding
matrix masks have pairwise distance greater than `2R`. By the triangle
inequality, no radius-`R` ball around any cached representative can cover two
packing members. Therefore a direct literal table needs at least the number
of entries in (3). This argument does not assume linear codebooks, separate
routing, or representative masks from the legal query family.

Registered substitution gives

```text
N                                             16,384
target coefficient fraction                    8/675
repair budget R                            3,181,457
packing removal radius d                         388
log2 minimum literal table bits       13,741.25484686
log2 complete 8 GiB bit capacity                  36
storage exponent deficit              13,705.25484686
minimum literal address width                 13,742 bits
```

Even one bit per cached answer therefore needs more than `2^13741` bits and
cannot be addressed by the declared 32- or 64-bit word machine. This is
decisive before codebook bytes, numerical values, metadata, traffic, or native
rounding.

Decision:

```text
REJECT_OMEGA_NEARESTPAIR_LITERAL_TABLE_BY_FINITE_COVERING_STORAGE_GATE
```

This rejects the literal cached-answer table. If codewords instead expose
reusable linear images such as `Wq` or `p^T W`, replacing a nonlinear
codebook by its linear span cannot worsen its covering radius and returns to
the matrix-local separable-image model closed by F-055. A materially different
succinct nonlinear decoder for the cached answers is the still-open general
data-structure problem; equation (3) does not reject it.

## OMEGA-TRAPSHIFT

Trapdoored matrices are a real new information source for matrices sampled
from their special distribution: a short associated circuit evaluates their
matrix-vector product in near-linear time. The proposed arbitrary-checkpoint
lift chooses such a mask `R` and uses

```text
W x = (W + R) x - R x.                                (4)
```

Equation (4) is exact over an associative field, and the trapdoor makes `R x`
fast. It does not make `(W+R)x` fast. For arbitrary fixed `W`, the shifted
matrix is still an arbitrary dense matrix. Computing that term directly
leaves exactly one full dense product; assuming a fast average-case solver
merely assumes the missing source.

The cited work uses trapdoored matrices to replace random matrices inside
algorithms and in worst-to-average reductions that already assume an
average-case algorithm. It does not compile an arbitrary supplied matrix into
the sampled trapdoored family. Its finite-field and exact-real identities also
do not preserve native BF16/FP32 accumulation order.

Decision:

```text
REJECT_OMEGA_TRAPSHIFT_AS_SOURCE_FREE_MASKING_IDENTITY
```

Primary source: Vaikuntanathan and Zamir, "Improving Algorithmic Efficiency
using Cryptography: Trapdoored Matrices and Applications," SODA 2026 / arXiv
2502.13065: <https://arxiv.org/abs/2502.13065>.

## Claim boundary

The two closures do **not** prove arbitrary nonlinear exact data structures
impossible. In particular, they do not cover a nonliteral compressed decoder
that answers the representative scalars without materializing their table,
nor do they prove that the all-query packing subfamily is reachable in a real
causal Transformer trace.

```text
OMEGA-NEARESTPAIR LITERAL TABLE: REJECTED
OMEGA-TRAPSHIFT: REJECTED
GENERAL COMPRESSED ADAPTIVE BILINEAR ORACLE: OPEN
NO SURVIVING CANDIDATE
TARGET NOT ACHIEVED
```

## Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_adaptive_codebook_trapdoor_frontier.py `
  --output-dir results\e0_adaptive_codebook_trapdoor_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_adaptive_codebook_trapdoor_frontier -v
```
