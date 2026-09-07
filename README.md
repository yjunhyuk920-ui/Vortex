# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** Every preparation, storage, movement, arithmetic and state cost counts. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Precision-dependent modular source](experiments/precision_rank_20260907/REPORT.md): actual local-ring factor constructor, source-only Python/C query and exact signed-output recovery within a guarded BF16-integer/FP32-exact domain. Small mod2 rank is not sufficient output precision. Generic128 cases have r_1=1 but r_K=128;32768 factor products versus16384 direct products. Files65584/98352B versus32768B BF16 original (same-domain bitpack2048/16384B). Dense preparation also counts.

40matrices/320queries/15872coordinates agree;18tests/603scientific files regenerate. The SAME Sylvester controls have full lifted inner rank yet FWHT uses896 add/sub slots. Thus the modular-factor theorem is not a general algorithm/read lower bound. Dense nonlinear scalar packing and variable-W border-rank gates are separately scoped. No full HF/KV/RNG/CUDA/405B/8GiB/4BQ4/TTFT/latency test.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`, `FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md), [contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Replay and preservation scope](experiments/precision_rank_20260907/README.md).
[Prior README unchanged](docs/research/history/pre_precision_rank_20260907/README.md). Core source/tests/expected results are checksummed in the capsule; raw science regenerates and is in the user ZIP. Detailed Korean report/development logs are ZIP-only. No universal performance claim.
