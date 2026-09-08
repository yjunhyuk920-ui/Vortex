# Validation matrix — 2026-09-09

[Current implicit direct-query proof/code/result](experiments/implicit_nonlinear_direct_query_20260909/REPORT.md).
[Prior direct-global producer proof/code/result](experiments/direct_global_producer_20260909/REPORT.md).
[Prior causal/global proof/code/result](experiments/causal_global_bridge_20260908/REPORT.md).
[Prior nonlinear router proof/code/result](experiments/nonlinear_router_frontier_20260908/REPORT.md).
[Prior native proof/code/replay](experiments/native_global_transition_20260908/REPORT.md).
[Pre-native-global matrix unchanged](docs/research/history/pre_native_global_transition_20260908/VALIDATION_MATRIX.md).

|Implicit-direct-query E0 item|Actual evidence and scope|
|---|---|
|Three-principle execution|P1 query-side complete image frame constructed; P2 exact value/state coalescing constructed then rejected at cost; P3 safe gauge rejected in product-preserving class|
|P1 arbitrary-GF2 producer|Finite block-image compiler; current query block bits are the addresses; exhaustive small-query controls exact|
|P1 registered cost|403,747,897,344 binary source bits -> 41,384,749,318,656 transformed image bits =4,817.8189 GiB; worst query 40,365,964,800 bits =4.6992 GiB|
|P1 traffic gate|query/source `0.09997814246350742`, so >=90% raw binary source removal, but `8.435655770358439x` the registered `8/675` line and no native lift|
|P1 general linear-frame bound|Complete one-sided linear image frames need >34x source storage even at 90% radius for registered widths; target-radius lower is ~1e23x; does not cover general nonlinear cells|
|P2 static exactness/cost|Declared ordered finite-word leaf ABI exact; best square case `16385/32768=50.0030518%`; high distinctness 100%; memberships `mn`|
|P2 dynamic coalescing adversary|FP32-exact integer family keeps all row accumulators distinct; `(accumulator,weight)` pair updates `268,435,456/268,435,456=1.0`|
|P3 arbitrary nonlinear safe gauge|Every Boolean-cube meet-preserving bijection is a coordinate permutation; exhaustive all-bijection controls through n=3, 40,346 total checked|
|P3 independent linear gauges|`C(x AND y)=A(x) AND B(y)` with invertible A/B/C forces one aligned coordinate permutation; n=2 exhaustive 216 triples, 2 survivors|
|P3 dense support|Safe row/column coordinate permutations preserve all 268,435,456 nonzeros of the 16,384-square dense adversary|
|Focused validation|15/15 current implicit tests PASS; canonical SHA `78644fd5ef4b3e0131bf78867e8986718ed02faeaf37b4228fd1c37909a7af29`|
|Deliberately not rerun|direct-global 10/10, prior nonlinear 15/15, geometry 28/28, causal-global 14/14, Boolean exhaustive controls, native-global 3510-file replay|
|OPEN|General implicit nonlinear finite-word direct-query source outside these three classes, arbitrary native ordered lift, complete causal logits/KV/RNG executor, O1-O5, whole-theory O6|
|Not tested|405B, CUDA, <=8GiB physical GPU, PCIe/SSD/HBM, native4BQ4 p50/p95, TTFT|

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

|Direct-global E0 item|Actual evidence and scope|
|---|---|
|Three-principle execution|P1 global reconstruction rejected; P2 exact-sum/witness reduced to direct MatVec by rounding gadget; P3 global nonlinear class remains open after explicit GF2 producer failure|
|P1 information Gate|All 8 GiB can remove at most binary `17.020392%`, Q4 `4.255098%`, BF16 `1.063775%` of arbitrary reconstruct-all source bits; original dense arithmetic remains 100%|
|P2 rounding Gate|Balanced IEEE-FP32 RNE 8-leaf gadget exact sum=0 but rounded result=`-b`; aligned gadgets return `-popcount(row&query)`; exhaustive widths<=5 pass|
|P3 global route theorem|Arbitrary global nonlinear cells/value-adaptive addresses allowed; one final route has `sum_i m_i*k_i<=t*64`, no per-matrix advice split|
|P3 registered bound|883 matrices; single simultaneous tuple >=312,468 words; proof-safe full route-cover >=578,619 words=4.4145 MiB|
|P3 target relation|578,619 / 299,072,516 favorable words = `0.193471%`; theorem DOES NOT reject target and complete causal tuple reachability/native lift remain unproved|
|Actual arbitrary-GF2 producer|Deterministic Gauss-Jordan compile `R=EW` + reverse row-XOR runtime; random rectangular/square controls exact|
|GF2 producer cost rejection|n=16,384 frozen producer adversary: `134,225,920/268,435,456 = 16385/32768 = 50.0030518%` operations; row-op metadata lower `3,757,867,008` bits =447.97 MiB for one matrix|
|Focused validation|10/10 new tests PASS; canonical SHA `a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe`|
|Deliberately not rerun|Prior 15/15, 28/28, 14/14, Boolean exhaustive and 3510-file native-global replay|
|OPEN|Direct implicit nonlinear arbitrary-native producer, native ordered lift, complete causal logits/KV/RNG executor, O1-O5, whole-theory O6|
|Not tested|405B, CUDA, <=8GiB physical GPU, PCIe/SSD/HBM, native4BQ4 p50/p95, TTFT|

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
