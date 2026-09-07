# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** Every preparation, storage, movement, arithmetic and state cost counts. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Dense interval–residue source](experiments/dense_residue_20260907/docs/REPORT_KO.md): exact dense corrections without a sparse-support assumption. An actual lossless bitplane source generates residues and a bounded interval decoder reconstructs every coordinate. The L1-based modulus retains all residual coefficient information; this query implementation reads every residual plane. It is not a new cheap source for arbitrary native dense models.

18 integer/BF16 synthetic matrices,288 queries,19968 coordinates match independent exact integer and C FP32/BF16 references;91 queries have nonzero corrections everywhere. Generic dense128 source reads51.60–51.70% of original BF16 bytes and is larger than bitpacked original integer payload. Restricted nonlinear congruence and joint-decoder witnesses are preserved.16 tests/111 scientific files replay to a fixed manifest hash. No full HF/KV/RNG/GPU/405B/8GiB/4BQ4/TTFT or latency test.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`, `FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md), [contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Replay and preservation scope](experiments/dense_residue_20260907/README.md).
[Prior README unchanged](docs/research/history/pre_dense_residue_20260907/README.md). Scoped decisions, proof obligations and costs are in the report. Existing policies, evidence and other draft branches remain unchanged.
