# EXP-059 — Pinned Real-Q4 Exact Shift-Displacement Rank Gate

Authority: workflow `30840432745`, source `cdae6160cd87b537e2f318c16430619736c7c9d9`, merge `82979e393a87845c4c757ce5dfd3fadc4e701d92`, artifact `8866573958`, ZIP SHA-256 `61d0c24ccacd310d7d0e7600cc926a882c74281827d524c4880c6715fad8800d`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 612 operator certificates; checksum/control/certificate mismatches 0; selected displacement-rank p50/p90 100%/100%; favorable query p50/p90 100%/100%; storage p50/p90 200%/200%; peak RSS 1,581,260 KiB.

Decision:

```text
REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES
```

The tested Toeplitz/Hankel/circulant-like route is rejected for this measured population. Q4 output preservation and target hardware were not tested.
