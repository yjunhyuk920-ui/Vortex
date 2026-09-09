# Validation matrix — 2026-09-09

## Native residual lift validation

[Actual evidence](experiments/codex_native_residual_lift_20260909/REPORT.md):
106 ordered FP32 cases,2,097,280 first integer pairs/16,768 residual pairs,
nine domain refusals and24 small CPU torch linear/mv calls. The native output
words and CPU RNG state match their declared checks. Incomplete source defects
were actually reproduced and corrected. No full HF generation/KV/405B/CUDA/
8GiB/latency measurement or whole-repository test run occurred.

## EXP-100A current CPU validation

[Actual validation](experiments/codex_fmm_integrity_20260909/VALIDATION.md):
full red, first-corrected and hardened gates,50 cells/132 controls,11 existing functions plus6
unittest regressions; exact old-field comparison and independent workspace sum.
This was not pytest or a repository-wide suite. Windows RSS unavailable/null.
No native405B/GPU/baseline/TTFT run; O1-O5 OPEN/O6 PARTIAL.

## Restricted BF16 producer extension — 2026-09-09

Current focused evidence is [experiments/codex_native_sparse_extension_20260909/VALIDATION.json](experiments/codex_native_sparse_extension_20260909/VALIDATION.json).
The scalar agent ran 1,566,724 frozen controls and five refusals; the primary
ran an independent 124,423-case integer oracle and the actual final packaged
CPU HF comparison. All 40 incremental logits/serialized cache states and eight
sampled generation logits/probabilities/tokens/RNG states match. Full-prefill
logits/KV values also match. HF v2's first-stride failure remains preserved.
Scalar historical v1 source bytes were not retained; do not claim a v1 red run.

Scope is Phase B native synthetic control and a declared scalar theorem.
No full repository suite, real public checkpoint, CUDA, 405B, <=8-GiB peak,
native4BQ4 latency ratio, TTFT or GitHub Actions ran this round. O1-O5 OPEN,
O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED.

## Native-fiber evidence — scoped CPU/proof only

| Item | Evidence and boundary |
|---|---|
| Native dense collision | 65536 n=3 vectors / 196608 FP32 words; zero mismatch |
| Positive-zero pruning | 16 full-tree comparisons at n=2,3,16,64; zero mismatch |
| Width 16384 | Two restricted pruned-tree controls; not full dense/hardware execution |
| Proof | Native node induction; invariant under bijections; later binary corollary is proof-only |
| OPEN | Whole O1-O5, full HF reachable-state/RNG/KV proof, native4BQ4/8GiB/TTFT and target hardware |

[Report](experiments/codex_native_fiber_20260909/REPORT.md), [raw summary](experiments/codex_native_fiber_20260909/results/summary.json).
This audit does not rerun or claim ownership of the preceding carrier's 11-test
receipt or earlier suites; their recorded provenance remains unchanged below.

## Independent native-gauge validation — 2026-09-09

| Item | Actual evidence and scope |
|---|---|
| Distinct field gauges | Finite certificate/basis-witness constructor; 30,564 invertible A,B pairs, forced C, zero classification disagreement; universal proof separate |
| Native schedule transport | 144 fixed cases / 288 output words, zero transported mismatch; naive permutation differs in 48 cases under declared balanced FP32 ABI |
| Shared decoder | GF2 four-output control: per-output ranks sum 12, actual shared products 4 plus 8 XORs; 256 cases exact |
| Native opaque-bit gate | 65,536 roundtrip vectors, 144 product vectors / 576 words, 8 known word controls, 20 gate-state steps; zero mismatch |
| Focused tests | 7 tests PASS; no complete repository/HF/target suite claim |
| Original frontier replay | Separately reran 10 direct-global tests and regenerated canonical a907cb1d... byte-identically before concurrent head advanced |
| Full mission | O1-O5 OPEN, O6 PARTIAL; no arbitrary dense-work removal, full HF/KV/RNG, GPU, 405B, native4BQ4 latency or TTFT evidence |

[Proof/source/raw results](experiments/codex_gauge_transport_20260909/REPORT.md). Plain-ABI boundary transforms are
explicitly added in the canonical bit-gauge v2 cost ledger; packed algorithmic
word storage is not a measured physical Python/GPU peak.

[Current implicit program-carrier proof/code/result](experiments/implicit_program_carrier_20260909/REPORT.md).
[Prior implicit direct-query proof/code/result](experiments/implicit_nonlinear_direct_query_20260909/REPORT.md).
The implicit-direct-query rows below preserve the preceding `9259497` execution
receipt. Its 15-test result and "deliberately not rerun" list belong to that
execution. The concurrent native-gauge audit above independently reran its own
7 focused tests and also replayed 10 older direct-global tests; this program-
carrier round itself did not rerun those completed suites.
[Prior direct-global producer proof/code/result](experiments/direct_global_producer_20260909/REPORT.md).
[Prior causal/global proof/code/result](experiments/causal_global_bridge_20260908/REPORT.md).
[Prior nonlinear router proof/code/result](experiments/nonlinear_router_frontier_20260908/REPORT.md).
[Prior native proof/code/replay](experiments/native_global_transition_20260908/REPORT.md).
[Pre-native-global matrix unchanged](docs/research/history/pre_native_global_transition_20260908/VALIDATION_MATRIX.md).

|Implicit-program-carrier E0 item|Actual evidence and scope|
|---|---|
|Three-principle execution|P1 address-only routing, P2 Patricia synthesis and P3 rank-normal state with paid transformed AND all actually constructed|
|P1 exactness|Arbitrary-GF2 alias compiler/runtime; exhaustive small-query equality; logical runtime addresses are fixed row/block pairs|
|P1 square arithmetic|28,516,762 table+row-XOR ops / 536,870,912 leaf+add slots = `0.053116608411073685`|
|P1 registered program carrier|40,365,964,800 aliases; explicit 64-bit descriptors `300.74987411499023 GiB`=`6.398601117664475x` binary source; does not fit 8 GiB|
|P1 scope|Rejects explicit descriptor/page-routing carrier and free-routing premise, not all nonlinear address encodings|
|P2 exactness|Arbitrary-GF2 compressed row-pattern program; exhaustive query controls incl. duplicate patterns|
|P2 square arithmetic|12,582,400 edge+leaf events / 536,870,912 baseline = `0.023436546325683594`|
|P2 program traffic|Preregistered one-64-bit-label/edge realization `536,838,144` bits=`1.9998779296875x` raw binary source before topology/memberships|
|P2 scope|Rejects explicit word-label Patricia realization, not every succinct exact query-time program synthesizer|
|P3 rank normal|Deterministic arbitrary rectangular GF2 `A W B=J_r` compiler verified; encoded query equality exact|
|P3 transformed AND|Exhaustive 4-bit control exact; isolated full-rank map fraction `1/n`, but literal two-projection AND micrograph `1.0000305171124708x` baseline and restores both dense maps|
|Focused validation|11/11 new tests PASS; canonical SHA `f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070`|
|Deliberately not rerun|implicit-direct 15/15, direct-global 10/10, prior nonlinear 15/15, geometry 28/28, causal-global 14/14, Boolean exhaustive, native-global 3510-file replay|
|OPEN|General nonlinear implicit program source, multi-query/time exact dynamic state, arbitrary native ordered lift, complete causal logits/KV/RNG executor, O1-O5, whole-theory O6|
|Not tested|405B, CUDA, <=8GiB physical GPU, PCIe/SSD/HBM, native4BQ4 p50/p95, TTFT|

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
