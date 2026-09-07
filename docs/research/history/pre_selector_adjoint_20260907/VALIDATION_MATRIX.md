# Validation matrix — 2026-09-07

[Report](experiments/dense_residue_20260907/docs/REPORT_KO.md), [replay](experiments/dense_residue_20260907/README.md), [fixed results](experiments/dense_residue_20260907/RESULTS.md).
[Prior matrix unchanged](docs/research/history/pre_dense_residue_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and scope |
|---|---|
| Constructor/source | Explicit row base,L1 bound,signed bitplanes; finite; every original coefficient read in preparation; no oracle |
| Decoder | Unique integer in certified interval of width<modulus; works with dense error support |
| Native scope | Integer BF16 weights,ternary integer inputs,positive coefficient per row,absolute sums<2^24; general floats excluded |
| Experiments |18matrices288queries19968coordinates:0 mismatch integer/C FP32->BF16;91dense-correction queries |
| Cost | Dense128 source51.60–51.70% of BF16;117.94–118.16% of bitpacked original; source traverses allplanes; not latency |
| Structural gate | L1 modulus uniquely encodes each residual coefficient; not a general query-read lower bound |
| State/decoder scope | Explicit nonlinear congruence and restricted syndrome-only collisions |
| Replay |16tests;111scientificfile manifest a581cff06d5d8247db16c3dc9bc37f98a71ebae60e5cc6a15a81a1ba727769a9 |
| Full mission |O1-O6OPEN;CORE_ADMISSION=false;3qualifyingnewprinciplesfalse |
| HF/fullKV/RNG/CUDA/405B/8GiB/4BQ4/TTFT/latency |NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite |NOT RUN |

Original inputs/programs/traces regenerate from committed code to the fixed aggregate manifest hash and are in the user ZIP. Remote persistence, conditional correctness and target performance remain independent.
