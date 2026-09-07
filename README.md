# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch one, one GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** All preparation, storage,
movement, arithmetic and state costs count. No training, weight or mission change.

## Current bounded record - 2026-09-07
[Token-addressed causal cut and coverage gate](experiments/causal_cut_20260907/REPORT.md).
A complete vocabulary source supplies first pre-RoPE Q/K/V without its original
weights at query time. All downstream work remains. In the declared synthetic ABI,
48 token IDs/2528 coordinates and160 causal steps match, including KV/RNG.
History witnesses forbid a token+position-only extension to the tested second layer.

At the official405B example, only0.0747966466% of projection MAC is removed;
table payload4.4033203125GiB and construction work are extra. This is not latency
or whole-model compression.20 tests and32 generated-file reproduction pass.
All three compared principles fail core admission; no large backend was built.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md),[next](NEXT_EXPERIMENT.md),[validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md),[mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_causal_cut_20260907/README.md).
Historical relative links retain root meaning. Previous policies/evidence/ledgers
are preserved; the report supplies the current addendum. Remote verification is
independent of scientific status. Parent PR133 head7cf9c4ac579060ecfbb0dd91a0ba2bda27d2b490.
