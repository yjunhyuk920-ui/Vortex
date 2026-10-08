# Validation matrix — 2026-09-08

[Current native proof/code/replay](experiments/native_global_transition_20260908/REPORT.md).
[Pre-native-global matrix unchanged](docs/research/history/pre_native_global_transition_20260908/VALIDATION_MATRIX.md).

|Current item|Actual evidence and scope|
|---|---|
|Original|Complete pinned SmolLM2-135M BF16 file269060552B, upstream LFS SHA verified; torch2.8.0+cpu/HF4.55.4|
|A|Guarded make_fx graph, exact pure CSE/DCE,61live roots, serialized/reloaded original tensor-name bindings|
|B|Actual byte/layout cache encoding and decoding; full original forward every step, costs charged|
|C|Actual TOP/singleton root-demand evaluation,0backward narrowing,no supplied desired answers|
|Generation|3fixed prefixes,12forward steps perarm; own native sampler;589824logits/760320KV coordinates allmatch|
|Additional roots|Full prefill196608logits plus60KVroots match; shape/stride/offset/cachefields/RNG match|
|Mask clarification|Explicitall-one mask/original auto-pad0 vs old automaticmask reference, unchanged outputs/RNG/sequences|
|Costs|A/C100%matrixMACs; A5/C4captureforwards; B12fullforwards+5114880BmincodecRW; program11775553B+originalweights|
|Scope witness|Direct nativeRoPE inverse fails144/211/263 of768keycoords at synthetic positions1/7/31; not allcharts|
|Tests/replay|14tests;3510fixedsciencefiles replay;3volatiletimefields excluded, originaltimings retained|
|OPEN|Universal O1-O6,alllegal masks/context/platforms,CUDA/405B/8GiB/native4BQ4/TTFT/fullrepositorysuite|

## Prior packet screen (unchanged scope)
[Prior matrix unchanged](docs/research/history/pre_output_envelope_20260908/VALIDATION_MATRIX.md).

|Item|Evidence and scope|
|---|---|
|Constructor|Paid entire-matrix extrema scan; row-packet lower/upper BF16 arrays, originals retained|
|Runtime|Sign-aware native interval tree; singleton broadcast or direct original rows; Python/NumPy|
|Native proof|FiniteBF16/separateFP32RNE/fixed balanced/+0padding/canonicalNaN/finalBF16; no real HF ABI claim|
|Corpus|Pinned originalSmolLM2-135M12matrices,96syntheticinputs,101376BF16coordinates,0mismatch|
|Independent check|36full-length FractionRNE dot products; main candidate/reference share NumPy elementary operations|
|Tests|12passed; source/load, signedzero, nonfinite intermediates, nonpower2, read-accounting controls|
|Costs|accepted_groups=0;oracle_broadcastable_groups=0 of432;originalreads100%;coefficient/FPwork100.78%-101.04%|
|CPU observation|No candidate latency benchmark; recorded acquisition/build durations are diagnostic only|
|Reproduction|Initial39files hashed;38numerical result files regenerate excluding later source-snapshot metadata|
|Remote evidence|Source/input/output/packettrace/hash/log preserved;upstream weight and derived plan payloads ignored/reacquired|
|Whole mission|O1-O6OPEN;CORE_ADMISSION=false;THEORY_STATUS=NOT_ESTABLISHED;targetHARDWARE_STATUS=NOT_TESTED|
|Not tested|HF_FORWARD=NOT_TESTED;FULL_KV_RNG=NOT_TESTED;TARGET_405B=NOT_TESTED;realactivations/CUDA/8GiB/native4BQ4/TTFT/fullrepositorysuite|

Scientific acceptance and persistence are independent. No claim of three new qualifying principles, no universal low-coherence assumption, no full-model speed inference.


## 2026-09-30 supporting validation

MEASURED: 14 unchanged native controls and 16 focused EXP100A tests pass; 3,510 canonical comparison hashes match. Metadata audit covers 50 recorded searches. NOT TESTED: new full-generation replay, full suite, target GPU, 405B latency/VRAM.

[Evidence and boundaries](experiments/cloud_continuation_20260930/REPORT.md).

Later full Linux replay was executed: within-runtime candidates match, but the
frozen Windows comparator fails (3,064/3,510 files differ; genuine logits/KV
bit differences). Original baseline remains untouched. See continuation report
for numerical counts, raw replay archive and registration timing deviation.


## Guarded scalar transition checks

Parent rerun: 34,026 leaves, 77,274 ordered sequences, 8,748 composition cases, 38,420 escape cases, and66 FP32/BF16 edge witnesses; zero mismatches. Fixed-itinerary replay:3,375 itineraries,118,125 output checks,33,750 inverse checks. Scope is the declared scalar/dyadic reference; not GEMM/HF/target evidence.

[Proof, costs and evidence](experiments/binade_transition_20260930/REPORT.md).


## 2026-09-30 complete scalar constructor and source boundary

Independent corrected rerun: 342,000 tiny prefix-word comparisons, 114 tiny identity checks, 3,520 FP32 identity/prefix comparisons, 192 BF16 wrapper comparisons, and six expected rejection checks; zero mismatches. Dynamic inverse-accounting audit covers 3,050 append steps with 164,931 calls and 347,198 level terms; zero accounting mismatches. FP32 subset totals are 52,281 inverse calls and 116,086 level terms. Earlier 47,515/105,036 counts covered intersections only and are explicitly superseded. Actual GEMM/HF ABI, target memory/latency and whole repository suite remain NOT TESTED.

[Native construction and accounting correction](experiments/binade_transition_20260930/native_charts/REPORT.md).
[Grammar-source accounting](experiments/tree_grammar_source_accounting_20260930/REPORT.md).


## 2026-09-30 heterogeneous source construction and limits

These two source packages received symbolic proof/arithmetic/source/link checks only. No numerical, model or GPU tests were run. Independent suffix audit verified the first-overflow proof, bounded integer representation, signed-zero/infinity monotonicity, conservative inventories and b=256 anti-coalescence condition after disclosed corrections. This is a bounded manual audit, not a whole-theorem formal verification or target result.

[Suffix construction and proof review](experiments/heterogeneous_source_20260930/REPORT.md).
[Static MatVec primary-source audit](experiments/static_matvec_literature_20260930/SOURCE_NOTE.md).


## 2026-09-30 whole-program source round

Independent read-only symbolic review verified output-only reverse witness recovery, the K(N,N) minor and >=2^(16N)-bit allocated-message bound for the stated unsimplified full-domain word-table schedule, and the exact fully dirty scalar-DAG witness. Recursive hashes and local links pass. These are manual proofs/document checks only, with no formal checker, numerical/model/GPU run or target performance evidence.

[Whole-program construction, scoped obstructions and audit](experiments/whole_program_causal_source_20260930/REPORT.md).


## 2026-10-01 nonlinear cell certificate

DERIVED: independent manual proof review of the modal-tree/rank argument and
greedy rank-one ruling bound. Exact integer checks reject t=0,...,4 and do not
reject t=5 at S=50,w=64,h=0; t=2 rejects h=1383 and does not reject h=1384.
With query-accessible original BF16 cells, S=725 rejects t<=2 but not t=3.
The specified three-bit nonlinear bijection passes all 64 source/query integer
checks. Fresh isolated regeneration leaves all six package files byte-identical;
SHA-256 checks and relative links pass. These are certificate/document checks,
not proof-assistant verification, HF execution or native kernel tests. Full
repository suite and target hardware remain NOT TESTED.

[Proof, exact certificate and scope](experiments/nonlinear_cell_address_fibers_20261001/REPORT.md).


## 2026-10-01 paid Krylov-feedback reference

The registered four synthetic sources have ranks 29,63,1,0. Their separate
native C reference and serialized-record executor match across 4,096 causal
steps, including decoded state, BF16 output words, inputs, tokens and RNG.
Independent review reproduced 32 scientific files and added 4,128 small
compiles / 20,576 transitions plus rank-64 and native-boundary checks. These
extra checks are disclosed as post-registration audit work. Earlier incomplete
counters and their corrected successor are preserved. The fixed-array word-RAM
cost bound is derived/manual-reviewed; Python timing, service latency, HF,
GPU and full repository suite remain NOT TESTED. Publication checks separately
verify the 36-file text manifest, Python/C syntax, links and cost constants;
no new numerical or model run was performed by the integrating writer.

[Construction, proof, costs and evidence](experiments/paid_krylov_feedback_20261001/REPORT.md).


## 2026-10-01 native BF16 product-cut certificate

Exact integers/Fraction verify 15 legal scalar affine rectangles, exact
normal/zero product representability and 15 independent output magnitude bits.
Parent independently reviewed the proof and checked the rectangles with separate
code. The 15m width extension and sign upper bound are symbolic under the stated
ABI. Existing graph/source hashes are pinned; their inspection is not a new HF
execution. The plan discloses symbolic discovery before registration. Publication
checks verify the eight-file manifest plus manifest file, source pins, Python
syntax, JSON and relative links. No native floating backend, SiLU, model, GPU,
latency or full repository test was run for this result.

[Proof, exact-word certificate and scope](experiments/native_gate_innovation_rank_20261001/REPORT.md).


## October 7 2026 public constant compatibility

STENSO displayed rule: x=3 differs by one FP32 ULP; x=4 matches. Four Fraction
midpoint inequalities pass, no ties. One strict-FP GCC14.2/libm cloud CPU process,
exit0, empty build/run stderr, original preregistration/source/expectation hashes
unchanged. C FE_ALL_EXCEPT flags match for these inputs; fenv API restoration
success is not full raw x87/MXCSR comparison. Target model/tensor/GPU, user
computer and performance tests NOT RUN.
[Evidence](experiments/primary_synthesis_20261007/REPORT.md).

## 2026-10-08 source/manual checks only

Pinned public files and primary sections were read; TorchLean's displayed operation inventory, [0,1] fallback and real-node theorem scope were independently rechecked. The CAP residual omission and direct three-step witness received two independent source/manual readings. Observer original-plant calls, recurrent branch full-QK/unnormalized return, and attention custom-reference versus production-native distinction were inspected. These are source/manual findings, not executed third-party code, formal-verifier certificates, numerical experiments or latency measurements. Documentation, JSON, local links, unchanged canonical prefixes and content hashes are checked in the package's LOCAL_VALIDATION.json. Full repository suite, original model/solver/GPU/user-computer and target hardware tests NOT RUN.

[Detailed evidence](experiments/native_source_continuation_20261008/REPORT.md).
