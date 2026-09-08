# Validation matrix — 2026-09-08

[Report](experiments/nonlinear_coordinates_20260908/REPORT.md), [replay](experiments/nonlinear_coordinates_20260908/README.md), [fixed validation](experiments/nonlinear_coordinates_20260908/results/validation.json).
[Prior matrix unchanged](docs/research/history/pre_nonlinear_coordinates_20260908/VALIDATION_MATRIX.md).

| Item | Evidence and exact scope |
|---|---|
| Finite constructor | Actual T0 orbit, affine coefficient inference, all-state check; 24attempts/8accepted/16refused |
| Direct constructor | Checks reversible primitive semantics/shared prefix/exact inverse suffix/original observer; zero state enumeration |
| Runtime | Serialized-source Python/C step; no target future output; same-size bijective state |
| Exhaustive |43520finite pairs and87040two-observer pairs;0mismatch |
| Causal |6144steps; declared output/full-state relation/each run's LCG and input choice agree;8order-sensitive history witnesses |
| Cost |12bit finite16424B/20488transition calls; direct328B equals compact original;216gate eliminations only with original O=E; O=id72decoder gates remain |
| Preparation |Finite callcount2561x eight original calls; direct initial72gates,720primitive visits/648tuple-comparison upper bound plus all actual storage/address/code costs |
| Scope |Synthetic reversible Boolean programs, not original BF16 neural computation; chart-switch counterexample; noncommuting/noninjective cheap countercontrols |
| Numeric audit |65539FP32patterns around1.5;65537-to-one BF16 fiber;local17bit fixed side-info fact;12edge probes;not full neural bridge |
| Replay |22tests/109sciencefiles;final manifest56c016845ebee52901283b15680f7460270f4a3ef9ecb0465a1bde1c4c3b728d |
| Full mission |O1-O6OPEN;CORE_ADMISSION=false;threequalifyingnewprinciplesfalse |
| HF/fullKV/RNG/CUDA/405B/8GiB/4BQ4/TTFT/latency |NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite |NOT RUN |

Source/tests/prereg/full Korean report/frozenmanifest are in a checksummed archival capsule. Raw scientific binary regenerates and is in userZIP; development history/logs are ZIP-only. The final manifest includes explicitly labeled added scope probes; previous manifest/run/scope files are preserved, not rewritten as if unchanged from first development.
