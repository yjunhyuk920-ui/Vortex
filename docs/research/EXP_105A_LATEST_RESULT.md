# EXP-105A latest result — Activation-Ordered Exact Residual-Bound Head Tournament

## Authoritative decisions

```text
REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE
NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE
```

- deterministic core: `17cebccb51a2fbdadf871767151962b47e462bd14caeba1707cd66a64a468598`
- integrity failures: `[]`

## Exact executable mechanism

The draft head uses signed Q4 integer weights, signed Q8 activations, exact int64 accumulation, dynamic dimension order by descending current `|x_j|`, and exact Cauchy tail bounds. Rows are removed only when their upper bound is strictly below the best surviving lower bound. Lowest token ID is the tie rule. The dense full score appears only in the independent validation path and is excluded from the mechanism ledger.

## Measured finite-word random-dense control

Shape `4096 x 2048`, 12 independent Q8 queries. Q4 payload, group-128 scale/zero reads, row-norm metadata, activation ordering, and per-stage score/residual/active-state traffic are charged.

| block | weight element fraction p50 | head-query/full-head p50 | p95 | early-before-final |
|---:|---:|---:|---:|---:|
| 64 | 82.140731812% | 171.128890094% | 173.654067096% | 9/12 |
| 128 | 83.594512939% | 129.780533734% | 131.685750625% | 2/12 |
| 256 | 87.490844727% | 112.351092170% | 112.925917682% | 0/12 |
| 512 | 99.597167969% | 113.767915614% | 114.082098568% | 0/12 |
| 1024 | 100.000000000% | 107.582720588% | 107.582720588% | 0/12 |
| 2048 | 100.000000000% | 104.273897059% | 104.273897059% | 0/12 |

The smallest p50 query traffic was obtained by the one-stage full-width block: `1.042738970588x` the ordinary Q4 head bytes. Smaller blocks skipped some weights but their exact selector-state traffic cost more than the saved payload. Projected onto the registered head plus one complete layer, this is `2,857,501,696` bytes/token, or `1.190625706667x` the 2.4-GB p50 budget. This projection is labeled `PROJECTED`; it is not target-hardware measurement.

## Target-scale finite-word late-decision Gate

A legal Q4 head/query can keep every row alive until the final dimension block. The best fully charged target ledger then chooses one full-width stage:

- head query bytes: `1,122,065,408`
- one complete realistic-Q4 target-width layer: `1,693,450,240`
- total draft bytes/token: `2,815,515,648`
- p50 budget: `2,400,000,000`
- total/budget: `1.173131520x`
- p50 pass: `False`

The head alone fits the byte budget, but it is not a complete causal transducer. Adding the minimum complete target-width layer makes the path fail before exact target verification. No unimplemented causal state source is credited.

## Positive and causal controls

- A structured Q4 head with decisive high-activation dimensions stopped after 6.25% of weight elements at block 64, proving the algorithm can prune real values rather than being a constant full-scan stub.
- The legal late-decision control read 100% of weight elements at every registered block.
- The causal dependency DAG required `K(L+1)` serial weight-bearing stages: for 8 tokens and 4 layers, 40 stages. Without guessed branches or a precompiled transition operator, token `t+1` cannot enter layer 0 before token `t` head selection.

## Local validation

```text
8 focused tests passed
exact runtime winner separated from dense reference validation
12 random finite-word queries x 6 block sizes, all exact
byte-identical deterministic rerun
Python compile PASS
SHA-256 ledger PASS
GitHub Actions not run
```

## Claim boundary

This closes the registered activation-ordered Cauchy residual-bound head index **when paired with the minimum complete target-width layer** as a p50 core. It does not prove every exact MIPS index impossible. A head-only path has no measured causal state source or `A>=339` continuation and is not promoted. Public-checkpoint execution, 405B execution, physical 8 GiB, CUDA/SASS, and final same-machine latency remain `NOT_TESTED`.
