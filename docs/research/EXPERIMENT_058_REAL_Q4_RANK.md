# EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

Authority: workflow `30826618962`, source `8ae03de4cc34317b5536aed42b9b8c22f98c88ea`, merge `3730d6ce8ca89df347079c366a91bcad4d904a85`, artifact `8861905858`, ZIP SHA-256 `851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae`.

MEASURED: 153 two-dimensional tensors; 144 registered dense projections; Q4 checksum mismatches 0; certificate/control mismatches 0; full integer/rational rank proven 144/144; favorable exact two-factor operation/storage p50/p90 200%/200%; peak RSS 886,572 KiB.

Decision:

```text
REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES
```

Ordinary exact low-rank factorization is rejected for this measured population. Q4 output preservation and target hardware were not tested.
