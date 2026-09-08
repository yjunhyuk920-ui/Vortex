# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state; same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** No training/weight/mission change; every preparation, storage, movement, arithmetic and state cost counts.

## Current research frontier — 2026-09-08

[Nonlinear adaptive router cover Gate](experiments/nonlinear_router_frontier_20260908/REPORT.md)
now closes the previous `25x108`,50-word,two-probe arbitrary-nonlinear local
capacity survivor. The theorem allows every stored word to be an arbitrary
checkpoint function, every later address to depend on prior returned values,
and arbitrary deterministic final decoding; exact largest-fiber plus Segre
geometry still limits one route's query span to `t*w`.

All 8,256 target-feasible side<=128 64-bit local rectangles are rejected. The
first local points not rejected by the fully adaptive Gate are `24x225` and
`25x216`,99 words/four probes at favorable traffic `8/675`. They are **not
constructions**. Fixed four-word supports and one-value-stage adaptive routing
are also rejected. A stronger independently-selectable 32-query Gate forces a
`64/675` encoded-word union on some abstract tuple—exactly 8x the registered
`8/675` line—even with fully nonlinear/value-adaptive words. The missing bridge
is now causal: that adversarial tuple has not been proved reachable along one
legal batch-1 Transformer continuation. Global cross-matrix encoding and native
numerical/state lifting remain open.

## Prior bounded native record — 2026-09-08
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
