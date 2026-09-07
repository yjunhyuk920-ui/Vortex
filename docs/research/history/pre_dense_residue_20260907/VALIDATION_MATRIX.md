# Validation matrix — 2026-09-07

[Report](experiments/native_decision_geometry_20260907/REPORT.md).
[Replay](experiments/native_decision_geometry_20260907/README.md).
[Prior matrix unchanged](docs/research/history/pre_native_decision_geometry_20260907/VALIDATION_MATRIX.md).

| Item | Evidence and scope |
|---|---|
| Constructor | Exact affine upper hulls/interval tree; no input-response enumeration; coefficient guard |
| Numerical | Exact BF16 product domain; monotone FP32 thenBF16 rounding; earliest numeric tie; raw signedzero |
| Selection | 9matrices67328queries: ID andselectedscore match C/NumPy; Fraction checks |
| Causal | 256scalarsteps output/state/ownLCG match;2historypairs differ; NOT TransformerKV |
| Storage/query | V4096 source~13.24x; p95logicalfields~2.7%; no physicaltraffic/latency claim |
| Cold budget | Originalread+sourcewrite+reload+8queryfields~3.46x; build arithmetic/temp costs additional |
| Whole-model | Freehead leaves99.47954% counted405B projections; core gate FAIL |
| Counterexamples | Realwinner/native tie; multidim native rounding; sampler coupling; current-output quotient |
| Replay | 13tests;51sciencefiles hash-identical;22textcapsule restored from emptyfolder |
| Full mission | O1-O6OPEN;CORE_ADMISSION=false;3qualifyinguniversalprinciplesfalse |
| Publicmodel/fullHF/logitAPI/softmaxsampler/fullKV/CUDA/405B/8GiB/4BQ4/TTFT | NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite | NOT RUN |

Independent C reference compilation is local, not GPU proof. Source archival XZ is
not inference compression. Raw binaries regenerate to preserved manifest and are
in userZIP. Remote commit, theory, and target hardware remain independent axes.
