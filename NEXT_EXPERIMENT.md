# Next Experiment

## Closed Gate — EXP-065

All 144 dense projections selected a full-rank four-row Kronecker rearrangement. Favorable lower-bound operations exceeded dense execution by more than 2x and static storage did not shrink.

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

## EXP-066 — Pinned Real-Q4 Exact Tensor-Train / MPO Bond-Rank Gate

### Mechanism

Factor matrix dimensions into ordered radix sequences:

```text
m = product_k m_k
n = product_k n_k
```

Pad the shorter sequence only with unit modes, pair `(m_k,n_k)`, and reshape the Q4 matrix into an interleaved Matrix-Product-Operator tensor with physical mode sizes `d_k=m_k*n_k`. For every cut `k`, certify the exact rank of the prefix/suffix unfolding:

```text
R_k = rank(unfold(W, product_{i<=k} d_i, product_{i>k} d_i))
```

These are necessary TT/MPO bond ranks. With `R_0=R_L=1`, exact core storage is lower-bounded by:

```text
sum_k R_{k-1} * m_k * n_k * R_k
```

All admissible radix schedules and deterministic mode-order variants defined before execution must be evaluated. Every selected cut receives independently verified witnesses under at least two primes.

### Population

Use the unchanged TinyStories-1M/3M/8M revisions and frozen EXP-057 Q4 checksums. Analyze all 153 two-dimensional tensors and report promotion statistics over all 144 dense projections.

### Accounting

Charge the bond-rank storage lower bound, mode-order metadata, per-row scales and biases, input reads, MPO contractions, every intermediate tensor read/write, output reductions, compilation and certificate work. Use favorable 4-bit core storage so rejection remains conservative.

### Controls

- exact rank-1 and low-bond MPO tensors certify correctly;
- a one-nibble mutation raises at least one bond rank;
- dense-random and forced-unique tensors produce high bond ranks;
- interleaved reshape/order round trips are exact;
- every selected bond witness verifies under two primes;
- no approximation, training, activation table or changed quantization.

### Promotion Gate

```text
zero checksum/certificate/control mismatch
all 144 dense projections covered
p50 lower-bound operation fraction <=10%
p90 lower-bound operation fraction <=25%
p50 lower-bound storage fraction <=10%
p90 lower-bound storage fraction <=25%
dense-random adversary p50 <=25%
projected static storage <=1 TiB
no largest-model degradation >25%
exact integer MPO reconstruction before operation-replacement promotion
```

Failure decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

### Claim boundary

Phase C weight observation and exact unfolding-rank certification only. Exact MPO cores, Q4 model-output preservation, a physical MPO kernel, real Transformer operation replacement, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT and tokens/sec remain NOT TESTED.
