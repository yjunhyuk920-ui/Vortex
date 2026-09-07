# Validation matrix — 2026-09-07

[Previous matrix retained byte-for-byte](docs/research/history/pre_boundary_convolution_20260907/VALIDATION_MATRIX.md).
[Current algorithm/proof/cost/evidence](experiments/boundary_convolution_20260907/REPORT.md).

| Item | Evidence and exact scope |
|---|---|
| Constructor/source | Exact diagonal equality, streaming BF16 file, BCV1 packed signed+absolute polynomials;16/32 constructed |
| Runtime | Current input only; no original matrix; exact integer convolution, rational certificate; no fallback |
| Numeric theorem | Finite nonzero normal bounded BF16, separate FP32 products, adjacent balanced RNE tree, BF16 store; certified nonzero words only |
| Queries |128 designed-structure queries;86 full vectors,42 unresolved |
| Coordinates |53916/54272 certified with0 false certificates; all FP32 references inside enclosure |
| Inexact accumulation |53937 FP32 sums differ from exact real sums; not integer-only exact-island tests |
| Generality |8 generic+8 individually perturbed matrices rejected; low displacement not proved for original checkpoints |
| Costs | Actual source bytes/operand bits/partial ledger, not physical traffic or latency; no complete10x bound |
| Independent checks |65278 BF16 roundtrips,1000 FP32 rational cases,100 small trees, signed pack/convolution and adversarial tests |
| Unit/reproduction |15 pass;435 generated files+manifest byte-identical |
| Whole mission O1-O6 |OPEN; no core promotion |
| Public HF/Transformer KV/RNG/GPU/405B/4B/TTFT |NOT TESTED |
| Repository suite/Actions |NOT RUN |

Capsule contains14 complete text files and hashes. Large binary/per-coordinate data
are regenerated against the recorded manifest, not all stored in capsule. Archival
XZ is not an inference codec. Remote commit verification is a separate status.
