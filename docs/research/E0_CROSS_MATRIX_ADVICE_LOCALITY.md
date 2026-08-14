# E0 Cross-Matrix Advice Locality

## Status and evidence boundary

- Date: 2026-08-09 Asia/Seoul
- Status: COMPLETE SCOPED LOCALIZATION THEOREM; TARGET-SIZED DIRECT SUM NOT
  PROVED; NO SURVIVING CANDIDATE
- Evidence: exact finite linear algebra, a finite sphere-covering witness,
  deterministic calculator, and nine independent/exhaustive GF(2) controls
- Model/checkpoint/hardware action: none

This note answers whether one globally mixed linear advice state provides free
information to many matrix-local Bilinear Cross Residual queries. It does not:
all advice content outside the requested matrix must be canceled either inside
the advice combination or by charged raw coefficients. The strongest finite
consequence proved here is nevertheless about 747 times too weak to reject the
registered target. General nonlinear and data-dependent data structures remain
open.

## Question

Can the complete `8 GiB` checkpoint-derived state be mixed across all 883
registered non-embedding matrices, then reused to answer each exact

```text
r_i^T W_i u_i
```

without paying matrix-local state or an equivalent cancellation cost?

The prior separable-code proof could not answer this question because it
allocated independent row/column image state to every matrix. Here the advice
may be an arbitrary global linear map of the complete binary checkpoint.

## Declared systematic linear coefficient-use model

Let

```text
V = V_1 direct-sum ... direct-sum V_K = F_2^D
D = 403,747,897,344
K = 883.
```

The unchanged binary checkpoint subclass is `w in V`. Preprocessing chooses a
global subspace `U <= V`, `dim(U)=s`, and stores the advice bits

```text
z_j = <g_j, w>,       span(g_1,...,g_s) = U,
s <= 68,719,476,736 bits = 8 GiB.
```

A query for matrix block `i` has coefficient vector

```text
q_i = vec(r_i u_i^T) in V_i.
```

The query algorithm may select any `a_i in U` as a function of the query and
may read the raw residual coordinates

```text
e_i = q_i + a_i.                                      (1)
```

The answer is exact because

```text
<q_i,w> = <a_i,w> + <e_i,w>.                           (2)
```

All advice reads, combination arithmetic, address discovery, preprocessing,
verification, misses, and fallback are granted for free. The only charged
quantity is `wt(e_i)`, counted as coefficient uses and summed across the
independently selectable registered query tuple. Probe locations may depend on
the query but not on checkpoint contents observed during the query. This is the
systematic linear model, not the general adaptive cell-probe model.

## Theorem 1 -- exact block localization

Let `pi_-i` project onto every block except `i`. Since `q_i` is zero outside
`V_i`, (1) gives

```text
pi_-i(e_i) = pi_-i(a_i).                               (3)
```

Thus every nonzero outside-block advice coefficient is also a raw cancellation
coefficient. If no outside coordinate is probed, then

```text
a_i in U_i := U intersect V_i.                         (4)
```

The spaces `U_i` have disjoint support and their direct sum is contained in
`U`, hence

```text
sum_i dim(U_i) <= dim(U) <= s.                         (5)
```

Equation (5), not the sum of projection dimensions, is the valid direct-sum
account. A small counterexample makes the distinction unavoidable:

```text
U = {(x,x) : x in F_2^n} <= F_2^n direct-sum F_2^n.
```

Both block projections have dimension `n`, while both shortened spaces
`U intersect V_i` have dimension zero. Using `(x,x)` for the local query
`(x,0)` requires canceling the second copy of `x` from raw coordinates.

## Theorem 2 -- a fixed outside support buys at most one dimension per probe

For a fixed outside coordinate set `T`, define the local image that can be
formed without touching any other outside coordinate:

```text
L_i(T) = pi_i(U intersect (V_i direct-sum E_T)).
```

Projection from `U intersect (V_i direct-sum E_T)` into `E_T` has kernel
`U intersect V_i`. Rank-nullity therefore gives

```text
dim L_i(T) <= dim(U intersect V_i) + |T|.               (6)
```

One fixed outside raw probe can expose at most one additional local linear
dimension. Nine tests check (2) and (3) by complete small-checkpoint replay and
check (6) exhaustively over every two-row presentation in a six-bit ambient
space.

Equation (6) is not a full direct-sum lower bound when `T` changes with every
query. The union of many small fixed-support images need not itself be a small
subspace. For example, different weight-one outside supports can expose many
different local basis vectors. Collapsing this union without charging its
query-dependent support is precisely the invalid step this audit prevents.

## Theorem 3 -- global rank-one span-to-cover consequence

Let

```text
t_i = max over rank-one q_i in V_i of distance_H(q_i,U)
P   = sum_i t_i.
```

Every binary `m_i x n_i` block is a sum of at most
`k_i=min(m_i,n_i)` rank-one matrices. Adding the corresponding advice vectors
and residuals proves that every ambient vector is within

```text
sum_i k_i*t_i <= k_max*P
```

of `U`, where the frozen population has

```text
k_max = 16,384.                                        (7)
```

Equivalently, align the rank decompositions across blocks: an arbitrary
block-diagonal tuple is a sum of at most `k_max` tuples containing one
rank-one block query each. Therefore the Hamming covering radius `R(U)` obeys

```text
R(U) <= k_max*P.                                       (8)
```

This argument permits globally mixed advice and query-dependent residual
supports. It does not divide the state by matrix count.

## Finite registered witness

If an `s`-dimensional binary subspace covers `F_2^D` within radius `R`, the
elementary sphere-covering condition is

```text
2^s * sum(j=0..R, C(D,j)) >= 2^D.                      (9)
```

For `R/D <= 1/2`, the ball volume is at most `2^(D H_2(R/D))`. Use the
conservative rational witness

```text
hot advice rate                    16,384 / 96,261
radius fraction                    13 / 50
H_2(13/50)                         0.8267463724926178
hot rate + entropy                 0.9969502972388806
strict margin below one            0.003049702761119377.
```

Thus (9) fails for every radius at most `(13/50)D`, and

```text
R(U) >= floor((13/50)D)+1
     = 104,974,453,310.                                (10)
```

Combining (8) and (10) gives

```text
P >= ceil(104,974,453,310 / 16,384)
  = 6,407,133 coefficient uses.                        (11)
```

This is a genuine finite global-advice lower bound, but it is nowhere near the
mission threshold:

```text
lower-bound fraction of D            0.00158691427055%
complete p50 allowance               1.18518518519%
bound / allowance                    0.001338958916
allowance / bound                    746.848904934x.
```

The theorem therefore does not reject a globally mixed linear code. It only
proves that cross-matrix projection rank cannot be counted as free local state.

## Complete favorable resource equation

The registered lower-bound accounting is deliberately stronger than any real
executor:

```text
representation              z = G w, rank(G) <= 68,719,476,736
hot state                   complete 8 GiB granted to binary advice
binary cold checkpoint      47.00244140625 GiB
corresponding Q4 population 188.009765625 GiB
build/amortization          0 (granted free)
selector/address discovery  0 (granted free)
advice reads/decode         0 (granted free)
coefficient uses            >=6,407,133
verification               0 (granted free)
miss/fallback               0 (granted free)
KV/work/runtime state       omitted
```

Even under perfect packing, (11) is only `100,112` 64-bit words or `12,514`
512-bit lines, carrying `800,896` bytes. This is a payload floor, not an
address-granular, cache-line, SSD, PCIe, HBM, or latency theorem: arbitrary
useful bits need not be co-located, while a word may serve multiple arithmetic
uses. No physical claim follows in either direction.

There is also no construction equation below the target. Representation
building, query selection, advice reads, arithmetic, verification, misses,
fallback, and complete branch state remain unspecified for any putative global
code. Since the logical lower bound is below the allowance, both feasibility
and impossibility remain undecided by this model.

## Literature boundary

[Ramamoorthy and Rashtchian, ITCS 2020](https://doi.org/10.4230/LIPIcs.ITCS.2020.35)
prove the exact equivalence between Hamming distance to an advice subspace and
query time in the systematic linear model. Their finite rank-one rigidity
bound is much weaker at linear advice dimension, and the paper explicitly
leaves open whether one linear-dimensional subspace can place every binary
rank-one query at `o(n)` distance. The present localization theorem does not
solve that open rigidity problem.

[Chakraborty, Kamma, and Larsen, STOC 2018](https://doi.org/10.1145/3188745.3188830)
give succinct systematic tradeoffs for Boolean matrix-vector and binary
vector-matrix-vector queries. Their registered redundancy range and theorem do
not yield a jointly preprocessed 883-block direct sum at the VORTEX state
point.

[Hambardzumyan et al., Spiky Rank, 2026](https://arxiv.org/abs/2602.23503)
define a spiky matrix as block diagonal with arbitrary rank-one diagonal
blocks. A registered tuple containing one rank-one coefficient matrix per
block is exactly of this form after padding, which explains why arbitrary
block tuples span the ambient space in at most `k_max` aligned layers. Their
rigidity result derives rigidity from *high* spiky rank of a matrix; it does
not give Hamming rigidity of the complete spiky-query set against one advice
subspace. No target-sized direct sum is imported from that work.

## Decision

```text
ESTABLISH_GLOBAL_LINEAR_ADVICE_LOCALIZATION
REJECT_FREE_REUSE_OF_CROSS_MATRIX_PROJECTION_DIMENSION
DO_NOT_CLAIM_TARGET_SIZED_DIRECT_SUM_OR_GENERAL_IMPOSSIBILITY
NO_GLOBAL_CODE_IS_SPECIFIED_AND_NO_CORE_CANDIDATE_SURVIVES_E0
MOVE_NEXT_TO_A_CAUSAL_QUERY_RESTRICTION_CERTIFICATE
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_MODEL_OR_HARDWARE_WORK
```

The durable result is a new exact accounting rule, not increased mission
feasibility. The remaining general linear rigidity gap is too deep for an
honest target claim. The next cheapest question is whether actual unchanged
Transformer execution restricts the causal residual-query population enough
to replace the arbitrary Cartesian contract with a smaller, independently
certified set without training or future leakage.

## Reproduction

```powershell
$env:PYTHONPATH = ".;.deps"
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_cross_matrix_advice_locality.py `
  --output-dir results\e0_cross_matrix_advice_locality

.deps\exp076-venv\Scripts\python.exe -m pytest -q `
  tests\test_cross_matrix_advice_locality.py
```

Expected focused result: `9 passed`. Authority:

```text
results/e0_cross_matrix_advice_locality/summary.json
results/e0_cross_matrix_advice_locality/checksums.sha256
```
