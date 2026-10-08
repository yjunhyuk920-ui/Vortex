# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state; same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** No training/weight/mission change; every preparation, storage, movement, arithmetic and state cost counts.

## Native product-cut boundary — 2026-10-01

An [exact symbolic BF16 product-cut result](experiments/native_gate_innovation_rank_20261001/REPORT.md)
shows fixed bit-affine representations need nonlinear injection rank 15m for
m independent product lanes when every product word is exposed under the declared
signed-zero-preserving ABI. Fifteen exact scalar rectangles prove the result
without a native/model run. This excludes a narrow fixed additive port on that
standalone domain; it proves no HF joint reachability, composed-graph rank or
latency lower bound. High rank alone does not imply expensive computation.

## Paid reachable-state construction — 2026-10-01

A [synthetic Krylov-feedback constructor](experiments/paid_krylov_feedback_20261001/REPORT.md)
derives a reachable basis for arbitrary binary A and a rank-one nonlinear
feedback port. Its causal encoded update preserves the declared output, RNG
and successor-state relation without rereading A at each hot step. Original
source, construction, records and requested full-state readback remain paid.
This works for a specific n<=64 BF16/FP32 circuit whose original program reduces
exact small integer dots to parity. It is not an HF operation replacement or
performance result; a cheap original-native innovation/observer source remains
missing. Full-mission O1-O6 and target hardware remain unresolved.

## Nonlinear cell bound — 2026-10-01

A [reviewed address-fiber bound](experiments/nonlinear_cell_address_fibers_20261001/REPORT.md)
closes the isolated 25x108 binary-source interface with 50 nonlinear 64-bit cells:
with no free source-dependent advice, exact worst-case decoding needs at least
five probes. Five probes are not a construction. An exact small-integer bridge
covers an exposed dense BF16 MatVec primitive only. Released-HF reachability,
global paid state, full-mission O1-O6 and performance remain unresolved. This is
an auxiliary lower bound, with no admitted core or target-hardware result.

## Cloud continuation — 2026-09-30

[Latest text-only result and evidence limits](experiments/cloud_continuation_20260930/PUBLIC_SUMMARY.md).

[CPU environment and classification repair](experiments/cloud_continuation_20260930/REPORT.md): the existing 14 native
controls and 16 focused EXP100A tests pass; 3,510 frozen file hashes are unchanged.
The public model is downloaded/hash-verified. A later full Linux replay preserves
within-runtime candidate equality but FAILS the original Windows frozen numerical
comparison; matching sampled tokens are insufficient. No acceleration is claimed. Resource-empty frontiers are now separated from
missing/corrupt search coverage. Historical results remain intact. No qualifying
new core, theory closure or target hardware result has been established.

## Constructive scalar result — 2026-09-30

[Guarded native transition maps](experiments/binade_transition_20260930/REPORT.md) replace a literal accumulator lookup
table by two offsets and parity intervals inside a fixed binade. A fixed dyadic
itinerary has at most three periodic levels with constructive inverses. Independent
local reruns pass the declared scalar checks. The [complete scalar native-chart
constructor](experiments/binade_transition_20260930/native_charts/REPORT.md) is now
implemented and independently rechecked, including corrected inverse accounting.
Every coefficient/product remains paid. This is a bounded representation theorem, **not** a 10x producer or mission
completion. Target hardware and universal HF equivalence remain untested.

## Heterogeneous source frontier — 2026-09-30

[Suffix synchronization](experiments/heterogeneous_source_20260930/REPORT.md) gives a finite, causal scalar method: bound a skipped prefix, execute the original suffix on both endpoints, and accept only identical native words; otherwise pay full fallback. Its prefix bound and anti-coalescence screen were independently reviewed and corrected with original drafts preserved. Broad intervals fail the screen; a cheap narrow source and sufficient workload coverage are still missing. This is a conditional theory construction, not a runtime or core admission.

[Packed MatVec source audit](experiments/static_matvec_literature_20260930/SOURCE_NOTE.md) distinguishes current-vector bit-column arithmetic from future-token batching. The inspected finite implementation still reads all matrix bits and does not preserve prescribed native accumulation. No target execution or new numerical experiment was performed in either source study.

## Whole-program source boundary — 2026-09-30

[Whole-program relational construction](experiments/whole_program_causal_source_20260930/REPORT.md) can generate exposed outputs and required successor state without reconstructing hidden accumulator values. Its explicit compiler pays prohibitive full-domain tables; ordinary input-conditioned substitution retains dense traversal. Continuation minimization and exact trace repair also supply no qualifying bound. Independent symbolic review preserves these narrow scopes: none is a universal impossibility result. This constructive round found no affordable source or qualified next implementation; the mission is still unresolved.

## Additional source attempt — 2026-09-30

[Conditional native field bridge](experiments/causal_cost_construction_20260930/NATIVE_FIELD_WRAPPER.md) gives a paid row guard and exact integer/field-to-native conversion on its accepted scalar domain. It does not supply the needed cheap numeric decoder or causal coverage. The additional constructive attempt therefore admits no new core, model run or target improvement. The full cost/time objective is controlling; separate tenfold reductions of both traffic and arithmetic are not imposed as a universal extra requirement. Original notes and parent-requested corrections are preserved.

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


## Primary source review October 7 2026

[New review and public-constant evidence](experiments/primary_synthesis_20261007/REPORT.md):
STENSO's displayed unguarded x/sqrt(x) rewrite differs by one FP32 ULP at x=3.
This is an auxiliary compatibility counterexample, not a new acceleration method.
[Follow-on review](experiments/primary_synthesis_20261007/LITERATURE_SUPPLEMENT.md)
keeps a verified soft-gate manuscript as an unread source-construction lead.
No qualifying new producer, target run, speedup or O1-O6 closure. Existing CSE,
codec, scalar and quotient exclusions and fixed mission remain.

## Native producer continuation — 2026-10-08

[Public primary/code audit](experiments/native_source_continuation_20261008/REPORT.md) compares native reachability slicing, compact observer-state execution and original-native surrogate certification. The inspected interval producer retains dense gate coefficient work and proves real enclosures; a pinned CAP certifier omits the paper's recursive residual check; certified attention distinguishes its custom unquantized reference from the original SDPA arithmetic path. These are scoped source/manual findings, not universal impossibility or new core evidence. The missing causal native producer and complete paid target upper bound remain open. No new numerical, solver, model or GPU run. [Prior source review](experiments/source_frontier_20261008/REPORT.md) remains unchanged. README_CURRENT=true; O1–O6 OPEN; CORE_ADMISSION=false.
