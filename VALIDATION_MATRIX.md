# Validation matrix — 2026-09-09

[Current causal/global proof/code/result](experiments/causal_global_bridge_20260908/REPORT.md).
[Prior nonlinear router proof/code/result](experiments/nonlinear_router_frontier_20260908/REPORT.md).
[Prior native proof/code/replay](experiments/native_global_transition_20260908/REPORT.md).
[Pre-native-global matrix unchanged](docs/research/history/pre_native_global_transition_20260908/VALIDATION_MATRIX.md).

|Latest E1 causal/global item|Actual evidence and scope|
|---|---|
|Three-principle round|A legal-causal exposure constructed; B global nonlinear producer remained without arbitrary-checkpoint encoder/address/decoder; C literal column-delta summary rejected under dense legal right-factor changes|
|Native square KV exposure|Actual BF16 `LlamaForCausalLM`/`DynamicCache`, 32x32x3 binary `v_proj` source, 3,072 source/cache coordinates, 0 mismatch, full/incremental cache bit equal|
|Native rectangular GQA exposure|`v_proj=32x224`, head_dim2, Q112/KV16; 7,168 exact source/cache coordinates, 0 mismatch, checkpoint/RNG unchanged|
|Restricted finite producer|64-bit binary column store + exact native RMSNorm/RoPE metadata; exact logits/K/V/RNG against native incremental path; 0 original dense `v_proj` calls|
|Restricted producer cost|2-layer 32x224: 128 source payload bits/token vs 229,376 variable-vproj BF16 full-read bits/token; 99.944196% removed in this restricted source only; whole-model/latency not implied|
|Standard-causal independent-right trace|n=8,r=4,K=32, legal sequence length536; 0 sign/common-magnitude failure; 0 full/incremental query-logit mismatch; RNG unchanged|
|Exhaustive parity decoder|32 queries x all 256 binary output rows = 8,192 native BF16 row evaluations; 0 same-parity codebook collision, 0 row/scalar decode mismatch|
|Area-5400 bridge|25x216 binary lm-head source, K=4, sequence length1516; 0 sign/common-magnitude, full/incremental, row-parity or left-mask mismatch; 32-query native trace NOT EXECUTED|
|Focused validation|14/14 causal-global tests PASS at final source state; canonical raw-result hashes in `results/e1_causal_global_bridge_20260908/checksums.sha256`|
|Related/standard validation|28/28 nonlinear/geometry regression PASS; `scripts/run_validation.py` exit0; complete tests directory attempted but BLOCKED by missing pytest/SciPy and Windows-incompatible historical test dependencies, so no full-suite PASS claim|
|OPEN|Area-5400 32-query native lift, globally nonlinear <=8GiB advice/direct-sum, arbitrary native finite-word MatVec producer, complete causal state, O1-O5, whole-theory O6|
|Not tested|405B weights/execution, CUDA, <=8GiB physical GPU, PCIe/SSD/HBM schedule, native4BQ4 p50/p95, TTFT|

|Post-persistence Principle-B E0 item|Actual evidence and scope|
|---|---|
|Adaptive Boolean-Mv black-box lift|Exact deletion-witness theorem: deterministic exact Boolean-semiring `Mv` -> GF(2) `Mv` requires >=`|supp(v)|` complete Boolean products; full support >=`n`|
|Finite transcript controls|All 37,067 distinct `<n` query sets for `n<=5` checked; false parity determinations `0`; canonical `n` isolating queries are tight on the witness family|
|One-shot nonlinear Boolean feature lift|For arbitrary nonlinear `E,Phi`, representing GF(2) inner product as one Boolean OR/AND product has exact Boolean rank `2^d-1`|
|Rectangle controls|All nonempty left subsets for `d<=4` checked; maximum all-one rectangle sizes `1,2,4,8`, exactly `2^(d-1)`|
|Area-5400 storage projection|`25x216`: minimum one-shot row features `25*(2^216-1)` bits, about `4.8755691e62x` the binary source before DS/query/native state|
|Validation|11/11 tests PASS; canonical v2 SHA `2d8a91800d574d32ec7611da6af6046d37cd4230a8c897f5c33ff268238695f6`|
|OPEN|Direct GF(2) or native finite-word data structure, Boolean-index internal reinterpretation that becomes such a direct source, global cross-matrix advice, native arithmetic/state, O1-O5|
|Not claimed|No universal nonlinear cell-probe impossibility; no target hardware/latency result|

|Latest E0 item|Actual evidence and scope|
|---|---|
|Three-principle screen|Checkpoint-space nonlinear words selected; joint restriction and native transition maps not promoted with undefined generators|
|General theorem|Arbitrary nonlinear `S` stored `w`-bit cells + fully value-adaptive deterministic `t` probes imply <=`S^t` query subspaces of dimension <=`tw`|
|Exact geometry|Independent deficit DP + closed binary Segre/simplex-product intersection formula; matches prior small exact controls and 24x225,d256 direct DP|
|Old nonlinear survivor|25x108,50 words,2 probes: exact coverage upper `7.45058081896844e-05`; REJECTED|
|Complete local scan|8,256 normalized side<=128 rectangles: 2,316 zero-probe-budget, 5,940 cover-rejected, 0 unclosed|
|New local frontier|First single-query area 5,400: 24x225 and25x216,99 padded words,4 probes,traffic8/675; no constructor|
|Routing refinement|Nonadaptive ratios `0.2243743/0.1121872`; one-value-stage adaptive `0.8974972/0.4487486`; all REJECTED|
|Strong independent-32 union|Some source needs >=84 active words over all rank-one queries; some 32-query tuple touches >=32 words=2048 bits=`64/675`=8x target; newer causal trace narrows reachability but exact area-5400 32-query/native/global-advice transfer remains unproved|
|Validation|15 focused tests PASS; 28 related tests PASS; authoritative summary SHA `af113c03b561827fc4ac93b1beb7216eb9de9cc9de17c95e6660b681c66f03d2`|
|OPEN|Legal-causal reachability bridge or causal-specific escape, global nonlinear cross-matrix producer, native arithmetic/state, O1-O5, whole-theory O6, target hardware|
|Not tested|405B weights/execution, CUDA, <=8GiB physical GPU, PCIe/SSD schedule, native4BQ4 p50/p95, TTFT|

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
