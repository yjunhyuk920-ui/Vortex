# EXP-064 — Pinned Real-Q4 Exact Output-Row Prototype Gate

Authority: workflow `30869720552`, source `a6371c39d85dc39669b98eac6125d9c3bbf4a5dc`, merge `3716584078a91ae307b11b4bf1b2662e1511e9c9`, artifact `8877450455`, ZIP SHA-256 `99c634bd4fb3903d32a1ed45fada7853ea4e1d199b375c129d1d4b8da4f39cb8`.

MEASURED: 153 two-dimensional tensors; 144 dense projections; 1,683 plans; zero checksum, reconstruction, or control mismatch. No dense matrix had identical or sign-related rows. Dense fallback was selected for 140 projections and an exact sparse-delta plan for four. Population p50/p90 operations and query bytes were 100%/100%. Best single projection: 70.522% operations, 93.811% bytes. Projected static storage: 211.31 GB.

Decision:

```text
REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY
```

Q4 model-output preservation, physical kernels, 405B, 8 GiB and target hardware were not tested.
