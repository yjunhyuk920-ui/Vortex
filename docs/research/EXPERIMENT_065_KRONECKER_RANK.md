# EXP-065 — Pinned Real-Q4 Exact Kronecker-Rearrangement Rank Gate

Authority: workflow `30870558294`, source `22fd41697979f0e5aeb570880714a47958270d7f`, merge `2e512e91b5bfcd5e30a19ef163a6438221a134dc`, artifact `8878551394`, ZIP SHA-256 `cf5bfcc53bda4117430c0856b6989704e79bb34fb52c9a4f81869bf20233155d`.

MEASURED: 153 tensors; 144 dense projections; 6,108 factorization plans; 306 selected two-prime certificates; zero checksum, witness or control mismatch. All 144 selected rearrangements were full rank 4. Favorable lower-bound p50/p90 operations were 203.891%/215.385%; storage 100.234%/101.042%; projected storage 202.66 GB.

Decision:

```text
REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY
```

Exact integer factors, Q4 output preservation, physical kernels, 405B, 8 GiB and target hardware were not tested.
