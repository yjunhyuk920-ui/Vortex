# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, single GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** Every preparation, storage,
movement, arithmetic and state cost counts. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Native rounding-aware decision geometry](experiments/native_decision_geometry_20260907/REPORT.md).
A concrete interval-envelope constructor selects the earliest native BF16 maximum
without evaluating every scalar affine score. No input-response enumeration or free
selector. Original coefficient information remains in the index. Scalar exactness
and short scalar context continuation are proved/tested, not arbitrary native dots,
full logits, reference sampling or HF/KV. 67328 registered queries match C/NumPy;
256 scalar causal steps match. 13 tests and51 science files replay from the capsule.
V4096 p95 logical query bytes2.69-2.78% of coefficient scan, but index~13.24x and
prep+8 minimum accounted bytes~3.46x. NOT physical traffic or measured speed.
Even free LM head leaves99.48% of counted405B projections. This is auxiliary only.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_native_decision_geometry_20260907/README.md).
Scoped decisions/assumptions/failures/architecture are linked from the new report;
existing policies, prior evidence and other draft branches remain unchanged.
