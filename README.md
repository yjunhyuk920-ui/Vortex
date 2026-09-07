# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch one, one GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** All preparation, storage,
movement, arithmetic and state costs count. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Native context-response source and deferred zero tags](experiments/context_response_20260907/REPORT.md).
A checked finite-geometry attention source updates original KV and reads matching
positions only when native nonmatching softmax weights are exactly zero. Original
PV grouping and signed zero are preserved; full HF/CUDA/RNG is not established.
640 conditional queries/10240 coordinates match. Generic/perturbed keys are rejected.
Favourable1024-context query-byte p50 is6.96%/7.79%, but p95>10%, state is88.32%,
online init/update/query62.99%-78.32%, and all dense projections remain unchanged.
At context4096 even free QK/PV leaves95.98% of counted405B MAC; not a latency bound.
16 tests and both241-file stages reproduce, including capsule restoration.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md),[next](NEXT_EXPERIMENT.md),[validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md),[mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_context_response_20260907/README.md).
Prior policies, evidence and root ledgers remain intact. Full report supplies the
scoped new addendum. Remote handoff is verified separately from scientific success.
