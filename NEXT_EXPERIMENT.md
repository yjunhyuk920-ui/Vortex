# Next Experiment

## Closed Gate — EXP-064

Exact identical/sign output rows were absent in all 144 dense projections. Four sparse-delta plans survived local dual-cost selection, but population p50/p90 operations and query bytes remained 100%.

```text
REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY
```

## EXP-065 — Pinned Real-Q4 Exact Kronecker-Rearrangement Rank Gate

### Mechanism

For each Q4 dense matrix `W` and every nontrivial factorization `m=m1*m2`, `n=n1*n2`, form the standard Kronecker rearrangement `R(W)` with shape `(m1*n1, m2*n2)`. A rank-`r` rearrangement is necessary for an exact sum of `r` Kronecker products:

```text
W = sum_i A_i tensor B_i
```

Each certified lower bound on `rank(R(W))` yields a lower bound on factor storage and on exact reshape-multiply execution. All row/column factor-order variants are tested. Modular determinant/rank witnesses must be independently verified.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and the frozen EXP-057 Q4 checksums. Analyze all 153 two-dimensional tensors and report promotion statistics on all 144 dense projections.

### Accounting

For every factorization and certified rank lower bound, charge at least:

- `r * (m1*n1 + m2*n2)` factor scalars;
- factor metadata and permutations;
- `r` applications of `B_i X A_i^T`, including all activation/intermediate reads and writes;
- per-row Q4 scales and biases;
- compilation/certificate work and exact reconstruction data.

A low modular rank is not sufficient for promotion: any surviving candidate must reconstruct every Q4 integer exactly over the execution representation.

### Controls

- exact rank-1 and rank-2 Kronecker sums certify correctly;
- one-nibble mutation raises the appropriate rearrangement rank;
- dense-random and forced-unique matrices have high certified rank;
- witnesses verify under at least two primes;
- reshape/order round trips are exact;
- no activation lookup table or approximation is used.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 lower-bound operation fraction <=10%
p90 lower-bound operation fraction <=25%
p50 lower-bound storage fraction <=10%
p90 lower-bound storage fraction <=25%
best dense-random adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
exact integer reconstruction for every promoted candidate
```

Failure decision:

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

### Claim boundary

Phase C weight observation and exact rank certification only. A physical Kronecker kernel, Q4 model-output preservation, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
