# EXP-060 — Pinned Real-Q4 Exact Zero-Sparsity Streaming Gate

Authority: workflow `30841671707`, source `bf89d087343a4790202126c34562ca0344ebe452`, merge `5f2af394180beaf3e5b5b8c7386d2becdf7eb8e7`, artifact `8867145590`, ZIP SHA-256 `5e5255dbedd779b734876faa027cd2bf5e4a1b00ece7f28cbf35f428fb9a0b05`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 1224 format rows; checksum/reconstruction/control mismatches 0; exact zero fraction p50/p90 17.76%/20.37%; operation fraction 82.22%/85.06%; query-byte fraction 150.93%/200.86%; best matrix 69.90% operations and 190.12% bytes; peak RSS 1,169,112 KiB.

Decision:

```text
REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY
```

Static exact-zero sparse streaming is rejected for this measured population. Q4 output preservation, physical kernels, and target hardware were not tested.
