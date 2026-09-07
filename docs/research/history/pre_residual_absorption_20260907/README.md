# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, single GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** All preparation, storage,
movement, arithmetic and state costs count. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Non-enumerative symbolic native source and compositional guards](experiments/symbolic_source_20260907/REPORT.md).
A finite source grammar plus IEEE SMT checks replaces exhaustive input-response
tables for small expressions. Two UNSAT-approved rewrites; two UNKNOWN cases retain
source. Separate all-finite-BF16 exact-product guards justify local FP32 rewrites.
448 matrix queries /1664 FP32 and1664 BF16 coordinates match C and Fraction references.
FMA reduces expression instruction count, not multiplication/addition work or
weight-literal reads. This is bounded auxiliary compiler work, not a new core engine.
22 tests,56 scientific files and a fresh source-capsule restore reproduce exactly.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_symbolic_source_20260907/README.md).
The report indexes scoped decisions/assumptions/failures/architecture. Prior policies,
evidence and root ledgers remain. Remote persistence is verified separately in the PR.
