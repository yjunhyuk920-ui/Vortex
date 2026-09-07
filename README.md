# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, single GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** All preparation, storage,
movement, arithmetic and state costs count. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Exact residual absorption and live-KV history quotient](experiments/residual_absorption_20260907/REPORT.md).
A concrete original-weight envelope can certify a whole residual branch before
reading its matrices. K/V remain live. A global certificate across tokens/layers
implies a token-indexed KV/logit transducer without enumerating histories.
This succeeded on4 deliberately high-scale, inherently context-blind synthetic
controls; ordinary controls did not qualify. 256 reference/guarded steps and128
table steps preserve logits/KV/RNG. No RoPE/full HF/public model/GPU test.
Token-table query reads192B vs96896B source in the small case, NOT a latency ratio.
Preparation/state costs remain; target geometry tables96.20GiB and current builder
~8.12e14 MAC. No general cheap context-dependent source or target closure.
19 tests,455 run files+2 derived files replay exactly from the source capsule.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_residual_absorption_20260907/README.md).
The report links scoped decisions/assumptions/failures/architecture. Existing policies,
prior evidence and other draft branches remain. Remote persistence is a separate axis.
