# EXP-066 — Exact Tensor-Train / MPO Bond-Rank Gate

This experiment is a bounded cheap-kill screen, not an MPO runtime implementation.

It does **not** download the pinned checkpoints or repeat modular Gaussian elimination. EXP-065 already evaluated every ordered nontrivial row/column factorization of the same deterministic Q4 matrices and froze 6,108 validated plan rows. An interleaved TT/MPO cut is byte-for-byte the same rearrangement when its factors are the row/column prefix and suffix products.

EXP-066 therefore:

1. verifies the frozen EXP-065 input checksums;
2. enumerates only the preregistered TT radix and mode-order family;
3. maps each nontrivial TT cut to its frozen EXP-065 rank lower bound;
4. assigns the deliberately favorable universal lower bound one to unit-boundary cuts that EXP-065 did not measure;
5. computes favorable classical MPO storage, operation, and query-byte lower bounds.

No new real-weight rank claim is produced. The scientific result is a deterministic derivation from frozen real-Q4 evidence plus independently tested reshaping equivalence.

Run:

```bash
bash experiments/exp_066/reproduce.sh
```

Default candidate evidence is written to `results/exp_066_candidate/`.

Promotion requires population p50/p90 operation and storage fractions of at most 10%/25%. Failure closes bounded exact classical single-matrix TT/MPO as a primary core direction for the measured population. It does not prove that every conceivable tensor representation is impossible.
