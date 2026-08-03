# Next Experiment

## Closed Gate — EXP-058

All 144 pinned real-Q4 dense projections were proven full integer/rational rank. Favorable conventional exact two-factor operation and storage lower bounds were 200% at p50 and p90.

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

## EXP-059 — Pinned Real-Q4 Exact Shift-Displacement Rank Gate

### Mechanism

Full-rank Toeplitz-, Hankel-, and circulant-like matrices can still admit fast exact transforms. For every registered Q4 dense projection `W`, certify the exact integer rank of four displacement matrices:

```text
D_zero_diag  = W - shift_zero_down_right(W)
D_zero_anti  = reverse_columns(W) - shift_zero_down_right(reverse_columns(W))
D_cycle_diag = W - shift_cycle_down_right(W)
D_cycle_anti = reverse_columns(W) - shift_cycle_down_right(reverse_columns(W))
```

Use primes 251, 257, and 263. Record every operator certificate and select the most favorable operator only after all four searches are charged.

### Pinned population

Use the unchanged TinyStories-1M/3M/8M revisions, the exact EXP-057 Q4 rule, and all 144 named dense projections. Q4 checksums must match frozen EXP-057 evidence.

### Favorable lower bounds

For displacement rank `r` and shape `m x n`:

```text
query:   r * max(m, n) frequency-domain products
storage: r * (m + n) generator scalars
```

These omit transforms, boundary terms, metadata, bitwidth expansion, and operator-search runtime, so they favor the candidate.

### Controls

- random exact Toeplitz: zero-fill diagonal displacement rank <=2;
- random exact Hankel: zero-fill anti-diagonal displacement rank <=2;
- exact circulant: cyclic diagonal displacement rank 0;
- deterministic dense-random negative control;
- transpose, column-reversal, and cyclic-shift equivalence controls;
- exact EXP-057 Q4 checksum agreement.

### Promotion Gate

```text
zero certificate/control mismatch
zero Q4 checksum mismatch
zero unregistered dense projection
p50 query lower-bound fraction <=10%
p90 query lower-bound fraction <=25%
p50 generator-storage lower-bound fraction <=10%
p90 generator-storage lower-bound fraction <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

Phase C observation only. Q4 output preservation, constructive generators, exact transform kernels, real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
