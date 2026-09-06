# VORTEX

Fixed mission: arbitrary public unmodified HF dense 405B, batch one, one total-8-GiB GPU, original output/RNG/required successor state, same-machine native 4B Q4 p50 <=1.2x, p95 <=1.5x, and the existing TTFT requirement. **Not achieved.** All preparation, original/transformed storage, movement, arithmetic, metadata and state costs count.

## Current bounded construction — 2026-09-07

[Native signed-orbit compiler and C executor](experiments/signed_orbit_20260907/REPORT.md). It shares original ordered rounded subtrees and their sign-mirrored evaluations, preserving signed zero and local nonfinite exceptions in a declared BF16-input/FP32-tree CPU ABI. It does not assume sparse residuals or small input changes. It is not a verified Hugging Face/CUDA replacement.

Deliberately structured dense full-rank 2048 Walsh controls use 2.442% code bytes and at most about 6.11% logical code/data traffic relative to original BF16 weight payload. This is not latency. General BF16 controls do not obtain these reductions, and integer/address work prevents a certified whole-work 10x claim. No core engine is admitted.

192 C queries / 87,040 output coordinates match FP32/BF16 references. Twenty unit tests pass; 188 generated evidence files and two manifests reproduce byte-exactly. The checked capsule contains all sources, tests, preregistrations, raw text observations, hashes and full Korean report; binary arrays are regenerated and hash-checked. See the report for commands and scope.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`, `FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`, `README_CURRENT=true`.

[State](RESEARCH_STATE.md), [next obligation](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [mission](MISSION_AND_WORKING_PRINCIPLES.md), [constructive contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).

[Previous README preserved byte-for-byte](docs/research/history/pre_signed_orbit_20260907/README.md). Historical relative links retain their original root meaning. Prior policies, experiments and negative evidence are unchanged. Parent: PR #131 head `8262178842ec4699a45093f3c29a669cfc158481`. A remote commit records evidence; it does not complete the theory or hardware mission.
