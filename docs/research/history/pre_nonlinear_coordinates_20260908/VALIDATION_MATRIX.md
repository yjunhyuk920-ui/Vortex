# Validation matrix — 2026-09-07

[Report](experiments/precision_rank_20260907/REPORT.md), [replay](experiments/precision_rank_20260907/README.md), [fixed validation](experiments/precision_rank_20260907/results/validation.json).
[Prior matrix unchanged](docs/research/history/pre_precision_rank_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and exact scope |
|---|---|
| Constructor | Actual local-ring elimination, LDR reconstruction, mod2 transform invertibility, stored U/V |
| Query | Source-only Python/C; centered residues with strict Q>2B; no dense W argument |
| Native scope | Guarded nonzero integer BF16 W, integer inputs, FP32-exact products/partial sums, zero-sign flag; not generic BF16 |
| Cases |40matrices/320queries/15872coordinates;0FP32/BF16 mismatch |
| Precision cost |128generic r1=1 to rK=128;32768 products;65584/98352B source versus32768B originalBF16 |
| Scope countercontrol |Same Sylvester matrices FWHT896add/sub despite full modular inner rank; no universal rank-cost theorem |
| Other gates |2401dense vectors/12bit one-message;variable-W matrix-vector flatten rank12; separately scoped |
| Replay |18tests/603sciencefiles;manifest b502d79e20c4f10f791737ca43581dfa5ea5f92dbc1f1c49b25253b22d8b8bb3 |
| Full mission |O1-O6OPEN;CORE_ADMISSION=false;threequalifyingnewprinciplesfalse |
| HF/fullKV/RNG/CUDA/405B/8GiB/4BQ4/TTFT/latency |NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite |NOT RUN |

Source/tests/expected results are in a checksummed archive capsule; raw scientific files regenerate and are in userZIP. Direct remote report contains scoped proof and paid costs; detailed Korean prose/development logs are ZIP-only. Persistence is not scientific acceptance.
