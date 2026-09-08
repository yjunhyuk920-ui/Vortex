# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state; same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** No training/weight/mission change; every preparation, storage, movement, arithmetic and state cost counts.

## Current bounded record — 2026-09-08
[Native whole-state transition constructors](experiments/native_global_transition_20260908/REPORT.md)
actually load pinned original SmolLM2-135M and execute HF generation using each
candidate's own logits, complete30-layer KV and original sampler/RNG. A exports
and reloads a guarded native graph; B implements a full byte/layout state codec;
C executes an explicit TOP/singleton demand graph. All observed logits/KV/RNG
and layouts match, including an explicit-mask clarification and fresh replay.

**No core acceleration:** A/C keep100% of matrix MACs; B calls the complete
original forward every step and adds5114880B minimum codec RW in the registered
12steps. Full original checkpoint remains necessary. Constructor full-forward
calls and graph storage are charged.14tests and3510science-file replay establish
only bounded correctness/reproduction, not target hardware/latency or a universal
theory. [Protocol/cost/obligations](experiments/native_global_transition_20260908/REPORT.md).

## Previous packet screen (unchanged scope)
[Native output-envelope screen](experiments/output_envelope_20260908/docs/REPORT_KO.md)
constructs a shared BF16-output certificate from row-packet extrema in the original
declared FP32 tree. Uncertified rows are actually read and calculated; the whole
vector is returned in the declared numerical domain.

Pinned original SmolLM2-135M weights:12matrices,96syntheticinputs,101376outputs,
0mismatch,36independent Fractiondots. All432packets had different actual outputs;
0certified/broadcastable. Original reads100%, coefficient/FPwork100.78%-101.04% plus
other overhead. This fails the fixed broadcast format, not every algorithm or
the checkpoint's reachable activation population. No HF forward or CUDA ran.

[Reproduction](experiments/output_envelope_20260908/README.md):12tests,
38numerical files regenerate. Raw upstream weights and derived min/max payloads
are ignored; code, hashes, inputs, outputs and traces are retained. No latency claim.
[Prior row-frontier evidence](experiments/native_row_frontier_20260908/REPORT.md)
remains unchanged. Concurrent PR147 correlation-source is committed on another
branch; [combined frontier](experiments/output_envelope_20260908/FRONTIER_SYNC.md)
records both failures without importing or overwriting that work.

THEORY_STATUS=NOT_ESTABLISHED; CORE_ADMISSION=false; TARGET_HARDWARE_STATUS=NOT_TESTED; FULL_MISSION_O1_O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false; README_CURRENT=true.

[State](RESEARCH_STATE.md), [next](NEXT_EXPERIMENT.md), [validation](VALIDATION_MATRIX.md), [AGENTS](AGENTS.md), [contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md).
[Prior README unchanged](docs/research/history/pre_output_envelope_20260908/README.md).
[Pre-native-global README unchanged](docs/research/history/pre_native_global_transition_20260908/README.md).
