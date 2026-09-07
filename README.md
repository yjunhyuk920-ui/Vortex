# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total
peak<=8GiB, original output/RNG/required successor state, same-machine native4BQ4
p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** All preparation, storage,
movement, arithmetic and state costs count. No training, weight or mission change.

## Current bounded record — 2026-09-07
[Native response-code constructor and C evaluator](experiments/native_response_code_20260907/REPORT.md).
Explicit small BF16 input domains permit whole-SwiGLU output-bit compilation.
Original dense operations and activation lookup disappear from the loaded query,
but exhaustive source creation grows as Q^n and output atoms grow rapidly.
148000 queries/592000 outputs match the declared CPU ABI; no all-input/HF claim.
For h32,768-byte weights compile to208/2160/33368/468328-byte programs as Q grows.
High response rank is not a full-read query lower bound: the chosen basis reads one
atom in generic fixtures. Storage/construction, not a universal impossibility, fail.
20 tests and171 generated files reproduce; no public model/full KV-RNG/GPU testing.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md),[next](NEXT_EXPERIMENT.md),[validation](VALIDATION_MATRIX.md),
[AGENTS](AGENTS.md),[mission](MISSION_AND_WORKING_PRINCIPLES.md),
[contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Previous README unchanged](docs/research/history/pre_native_response_code_20260907/README.md).
Prior policies/positive-negative evidence/root ledgers remain; the current report is
the indexed scoped addendum. Remote commit persistence is separate from theory success.
