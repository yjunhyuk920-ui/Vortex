# Validation matrix - 2026-09-07

[Current bounded record](experiments/causal_cut_20260907/REPORT.md).
[Previous matrix unchanged](docs/research/history/pre_causal_cut_20260907/VALIDATION_MATRIX.md).

| Item | Exact evidence/scope |
|---|---|
| Constructor | All vocabulary IDs; same single-token synthetic ABI; no batched equality assumption |
| Query | pread native BF16 Q/K/V row; first-cut weights absent; checkpoint/ABI/payload checks |
| Numeric |48 IDs/2528 coordinates; direct/scalar projection mismatch0 |
| Causal |2 random synthetic3-layer decoders;160 steps/128 generated tokens; logits/KV/RNG mismatch0 |
| History boundary |[1,3] vs[2,3]: first cut same; second cut differs; token+position-only extension refuted |
| Costs |0.0747966466% projection MAC removed at cited405B dimensions;4.4033GiB table; construction paid |
| Three-way gate | token cut/exhaustive branches/lazy KV all nonqualifying; no core promotion |
| Tests/reproduction |20 local tests;32 generated files hash-identical; restored text capsule re-executed |
| O1-O6/full theory |OPEN; no universal native/history/target budget construction |
| Public HF/CUDA/GPU/405B/native4BQ4/TTFT |NOT TESTED |
| Full repo/Actions |NOT RUN |

Capsule preserves17 complete text files and original32-file hashes. Actual binary
inputs/tables/final states and detailed traces regenerate to those hashes; full
originals are in user ZIP, not all in the remote text capsule. XZ is archival only.
All earlier evidence and constraints remain unchanged. Handoff is independently
verified after commit; it is not theory/hardware success.
