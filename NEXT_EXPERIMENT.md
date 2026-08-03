# Next Experiment

## Closed Gate — EXP-057

Authority: `results/exp_057/summary.json`; workflow `30824957941`; source head `cf9d7099dc11b22ce24ba6e096712d5da1bc3729`; artifact `8860450501`; ZIP SHA-256 `7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0`.

All 144 real dense projections had zero exact repeated/sign-related columns in FP32, Q8, and Q4. Q4 p50/p90 operations were 82.8918%/85.8398% and query bytes 329.0244%/490.6845%. Decision:

```text
REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY
```

## EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

### Mechanism change

Test whether deterministic row-symmetric Q4 projection matrices admit an exact low-rank factorization:

```text
W = A @ B
W x = A @ (B x)
```

No approximation or truncated SVD is allowed. Instead of assuming rank, compute exact modular-rank certificates. A full-rank minor modulo any registered prime proves the integer/rational rank is full and rules out a lower exact factorization rank.

### Pinned evidence

Use the same unchanged TinyStories-1M/3M/8M revisions and the exact EXP-057 Q4 rule. Analyze every named dense-projection matrix. Embeddings/output heads are reported separately.

### Registered primes

```text
251, 257, 263
```

Stop after the first full-rank certificate; test all primes only when a matrix remains deficient. Record pivot rows/columns, certificate prime, rank lower bound, minimum dimension, and checksums.

### Fully accounted lower bounds

For certified rank `r`, any conventional exact two-factor path must perform at least:

```text
r*n + m*r scalar multiply/add terms
```

and store at least `r*(m+n)` factor scalars before metadata. Compare this with the direct `m*n` matrix path. Calculate the maximum rank that could meet 10% and 25% operation budgets, and determine whether the certified lower bound already exceeds them.

### Controls

- known exact low-rank products with registered ranks;
- full-rank identity and random integer controls;
- row/column permutation rank invariance;
- duplicate-row rank-deficient control;
- deterministic Q4 checksum agreement with EXP-057 rules.

### Promotion Gate

```text
zero certificate/control mismatch
zero unregistered dense projections
real-matrix p50 exact-factor operation lower bound <=10%
real-matrix p90 exact-factor operation lower bound <=25%
real-matrix p50 factor-storage lower bound <=10%
real-matrix p90 factor-storage lower bound <=25%
no model-size degradation >25%
```

Failure decision:

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

### Claim boundary

Phase C observation only. Q4 output preservation, factor-kernel execution, actual Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.
