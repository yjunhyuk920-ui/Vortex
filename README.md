# VORTEX

Fixed mission: arbitrary public unmodified HF dense405B, batch1, one GPU total peak<=8GiB, original output/RNG/required successor state; same-machine native4BQ4 p50<=1.2x,p95<=1.5x and existing TTFT. **Not achieved.** No training/weight/mission change; every preparation, storage, movement, arithmetic and state cost counts.

## Current research frontier — 2026-09-09

[Direct global finite-word producer frontier](experiments/direct_global_producer_20260909/REPORT.md)
now attacks the remaining direct GF(2)/native data-structure hole with three
frozen principles. Global reconstructive coding is rejected: even giving all
8 GiB to exact advice removes at most `4.255098%` of an arbitrary Q4 source and
then leaves `100%` of dense arithmetic.

Exact mathematical summation also does not make native rounding a small
independent problem. An eight-leaf balanced-FP32 gadget has exact sum zero while
its rounded result is `-b`; composing the gadget encodes arbitrary binary MatVec
counts entirely in rounding error.

The strongest new theorem is global rather than matrix-local. For arbitrary
globally mixed nonlinear 64-bit cells and fully adaptive addresses, a final
route answering simultaneous full-Mv tuples has total coefficient span
`<=t*64`; no 8-GiB advice split by matrix is assumed. On all 883 registered
non-embedding matrices the proof-safe route-cover floor is `578,619` words.
That is only `0.193471%` of the favorable registered target allowance, so it is
**far too weak to prove target impossibility**.

The round also constructed a real finite producer for every GF(2) matrix:
Gauss-Jordan compiles `R=E W` plus a row-XOR stream and runtime exactly recovers
`Wv` by reverse replay. It is not fast enough. At width 16,384 the frozen
producer adversary costs exactly `16385/32768 = 50.0030518%` of dense scalar
bit operations and at least `447.97 MiB` of row-program metadata for that one
matrix. The arbitrary-native globally nonlinear producer therefore remains
OPEN rather than being renamed as this binary construction.

[Causal/global producer frontier](experiments/causal_global_bridge_20260908/REPORT.md)
compared the frozen legal-causal exposure, global nonlinear producer, and paid
dynamic-summary principles. The ordinary `transformers.LlamaForCausalLM` bridge
now exposes arbitrary binary `v_proj` information bit-exactly in required
`DynamicCache.values`; a native GQA `32x224` control checked 7,168 source/cache
coordinates with zero mismatch and exact full/incremental cache equality.

The same restricted binary/basis family now has a **real finite producer**:
lossless 64-bit column packing plus paid RMSNorm/RoPE metadata emits exact native
logits/K/V/RNG successor state without calling an original dense `v_proj` kernel.
For two `32x224` variable sources it reads 128 source bits/token versus 229,376
bits for a full BF16 variable-`v_proj` read (`99.944%` removed). This is not an
arbitrary-checkpoint constructor: binary weights, basis states and the declared
zero paths are essential.

A second standard-causal sign/delta compiler placed 32 independently selected
right factors in one legal batch-1 trace at width 8. All 8,192 exhaustive binary
output-row parity decodes, native full/incremental query logits and RNG controls
matched. The exact `25x216` area-5,400 source passed four native right-factor
queries with zero row/left-mask mismatch. The **area-5,400 32-query native trace
has not been executed**, and globally mixed 8 GiB advice is not a proved direct
sum. Literal `y'=y+W(s'-s)` column-delta summaries return to full dense effect
when legal successive right factors differ in every coordinate.

A post-persistence [Boolean-MatVec exact-lift Gate](experiments/causal_global_bridge_20260908/BOOLEAN_MATVEC_LIFT_RESULT.md)
now closes two tempting Principle-B detours without overclaiming a general
cell-probe impossibility. Any deterministic adaptive black-box conversion from
complete Boolean-semiring `Mv` answers to exact GF(2) `Mv` needs at least
`|supp(v)|` Boolean products; full-support queries need `n`. A stronger one-shot
escape that permits arbitrary nonlinear row/query feature maps still needs
exactly `2^d-1` Boolean features to realize `d`-bit GF(2) inner product. At the
`25x216` area-5,400 bridge this alone is about `4.8755691e62x` source storage.
Direct GF(2)/native globally mixed data structures remain OPEN; these theorems
must not be promoted into a universal arbitrary-data-structure lower bound.

The previous [nonlinear adaptive router Gate](experiments/nonlinear_router_frontier_20260908/REPORT.md)
remains authoritative: `25x108`/two-probe and all side<=128 local routes are
closed, the first single-query survivors are area 5,400, and the abstract strong
independent-32 interface is 8x over target. The remaining constructive hole is
an explicit zero-error arbitrary finite-word matrix-vector producer with globally
paid advice, native ordered arithmetic, and a complete causal state/cost lift.

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
