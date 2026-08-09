# VORTEX Decision Log

Append-only decisions. Authoritative run identities are read from committed result JSON.

## D-001 — Final target fixed

Arbitrary public unmodified Hugging Face dense model; runtime only; real 405B; total GPU VRAM <=8 GiB; original contract preserved; 4B-class user experience; independent reproduction.

Status: ACTIVE.

## D-002 — Current environment is not Phase D

GitHub Actions CPU cannot be labeled target GPU, 405B, CUDA, PCIe, SSD, TTFT, tokens/second, power, physical block reuse, or peak-VRAM evidence.

Status: ACTIVE. Phase D NOT TESTED.

## D-003 — Phase A/B/C/D adopted

Status: ACTIVE.

## D-004 — E0–E7 adopted

Status: ACTIVE.

## D-005 — MEASURED/DERIVED/PROJECTED/UNVERIFIED separation adopted

Status: ACTIVE.

## D-006 — mmap/index/DAG components are auxiliary

Evidence: PR #50/#52/#54.

Status: ACTIVE.

## D-007 — Raw prefix enumeration rejected as core

Evidence: 64/64 unique nodes excluding duplicate; held-out start coverage 0%.

Status: REJECTED.

## D-008 — Exact future-DAG accepted only as body compression

Evidence: 64->38 nodes; causal held-out start coverage 0%.

Status: AUXILIARY.

## D-009 — Core research must skip or amortize original operations causally on unseen prompts

Status: ACTIVE.

## D-010 — EXP-047 correctness primitive accepted at E1

Authority `results/exp_047/summary.json`, workflow `30793232558`.

Status: AUXILIARY.

## D-011 — Global-range and range-rescue CPTC rejected as core

EXP-047 certified 4/525 with 99.238% fallback. EXP-047R exact realized range oracle median/p90 was 100%.

Decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

Status: REJECTED CORE.

## D-012 — Completed evidence is immutable

Completed workflows write isolated reproduction output. Frozen result directories and raw checksums are authoritative.

Status: ACTIVE.

## D-013 — Exact longest-prefix block verifier accepted at E1

EXP-048 verified proposal blocks left-to-right, committed matching prefix plus exact first-mismatch correction, and never committed later predictions.

Status: AUXILIARY.

## D-014 — Perfect proposal proves verifier arithmetic only

96 exact future tokens / one target pass =1.0416667%, but future information true.

Status: NON-DEPLOYABLE UPPER BOUND.

## D-015 — Hard Jacobi and same-checkpoint partial-layer self-draft rejected

EXP-048 hard Jacobi p50 181.25%; partial-layer draft p50 committed 1 and p90 2893.843%.

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

Status: REJECTED CORE.

## D-016 — Target-only continuous fixed-point generation rejected

Authority `results/exp_049/summary.json`, workflow `30803672059`, artifact `8851957250`.

Favorable reference-selected result: p50 prefix 4.5, maximum 6, p90 fraction 168.778596%, Anderson/Jacobi 0.25x. Hidden triangular targets preserved transcript indistinguishability and the one-new-position-per-round barrier.

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Status: REJECTED CORE; solver/verifier references retained.

## D-017 — EXP-050 target-independent external draft Gate executed

Authority:

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact 8852817664
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
```

MEASURED correctness/causality:

```text
9 EXP-050 tests passed
repository validation passed
18 target/prompt cases
36 target/draft/prompt pairs
108 K rows
exact committed-output mismatches 0
target-future uses 0
E3 future-oracle failures 0
```

Status: EXECUTED, E1.

## D-018 — Fixed target-independent draft universal guarantee rejected

A deterministic draft proposed first token 7. An arbitrary causal target chose token 8 for the same prompt. Exact verification reported matching proposal prefix zero and committed only exact correction token 8.

Therefore a fixed target-independent draft cannot guarantee even one matching proposal token for every arbitrary target.

Status: UNIVERSAL FIRST-TOKEN COUNTEREXAMPLE ACCEPTED WITHIN DECLARED INTERFACE.

## D-019 — Tested TinyStories external draft pool rejected as practical core

The exact target reference selected the best eligible external draft and K per target/prompt.

MEASURED:

```text
p50 exact proposal prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
matching prefix zero 72/108 rows
all selected K=64
Korean useful acceptance false
structured JSON useful acceptance false
target median prefixes 1.0 / 0.0 / 0.5
```

Gate failures: prefix, traffic, family coverage, target-size trend, and universal counterexample. Exactness/causality passed.

Decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested pool is also rejected as a restricted practical core. Proposal-tree continuation is prohibited.

Status: REJECTED CORE.

## D-020 — Actual 4B draft requires 507 exact proposal tokens before overhead

PROJECTED:

```text
4B/405B draft ratio = 0.0098765432
required total fraction = 0.01185185185
4/405 + 1/K <= required
K >=507
```

The older 85-token requirement applies only to a zero-cost proposal.

Status: ACTIVE RESOURCE EQUATION.

## D-021 — Next Gate changes skip axis from future tokens to Transformer depth

EXP-051 will use exact greedy prefixes and audit every intermediate target block depth for the current next-token decision.

Primary oracle: earliest suffix-stable depth whose intermediate final-norm/LM-head token equals the exact final token at every later depth.

A late-decision adversarial residual chain tests the universal fixed-depth boundary. No selector/certificate is built unless the non-deployable suffix-stable oracle survives lenient 10%/25% traffic Gates.

Status: ACTIVE NEXT GATE — EXP-051.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## D-022/D-023/D-024 — Close EXP-051/052 and select EXP-053

D-022 rejects EXP-051 layer-tail skipping: median suffix-stable depth 25%, p90 37.5%, median/p90 favorable traffic 82.2069%/99.8011%, with a final-layer adversary.

D-023 records EXP-052 authority `results/exp_052/summary.json`; workflow `30811429049`; source head `d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`. 1,152 exact warm states and 36 leave-one-family-out rows produced zero wrong hits and zero build/evaluation leakage, but P0 prefix and S0 KV-state held-out hit rates were 0% in every family. Fallback was 100%, natural exact reuse median/max was 1/1, and p90 fully-accounted target fraction was 6.0 (600%). Same-state replay was 100% exact and required at least 85 repetitions. Under 8 GiB hot index plus 1 TiB cold advice, combined coverage of 2^48 independent states was 6.357828752356909e-7, leaving fallback 0.9999993642171248. Decision: `REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY`.

D-024 requires the next mechanism to be non-enumerative and weight-derived; EXP-053 is active.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## D-025/D-026 — Reject AIG structural hashing and select EXP-054

D-025 records EXP-053 authority `results/exp_053/summary.json`; workflow `30814648709`; source head `325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge `4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`. 24 weight-derived circuits were exhaustively checked over 4,506,624 inputs with zero output-bit mismatch and no truth-table representation. Structural hashing left p50/p90 reachable fractions 0.84168345/0.94107229; dense-random p50 was 0.92452096. The maximum 405B source-parameter circuit projection was 255.5966 TiB. Late-bit controls simplified to zero AND nodes, but sparse controls still retained 65–78% of the exact bit-blast and projected 3.17–7.45 TiB. Growth and compile-amortization Gates passed; node, byte, storage, and random-dense Gates failed. Decision: `REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY`.

D-026 selects exact reduced ordered decision diagrams as the next representation class. They must be compiled from weights/residual arithmetic states, not a stored truth table, and must charge compile-state visits, unique nodes, bytes, path probes, variable-order search, and fallback.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## D-027/D-028 — Reject reduced diagrams and select EXP-055

D-027 records EXP-054 authority `results/exp_054/summary.json`; workflow `30816333096`; source head `2c63da85050afcedad6a00698a6f8fddd3bc99d2`; artifact `8856906303`; ZIP SHA-256 `0dc642f306cea99ce01095758a5f49151092d530efb94d36985553e408596edf`. 24 operators were compiled in natural and weight-magnitude orders: 48 completed diagrams, zero ceiling/fallback, zero mismatches across 9,013,248 validations, and zero truth-table representations. Selected global p50/p90 path fractions were 35%/95%. Dense-random growth was 1.6872587x per added input bit, maximum projected storage was 202.2479 TiB, and maximum order-search amortization was 1,185,055 queries. Late-bit controls reached 5–12.5% paths, but dense, low-rank, and sparse families failed the universal Gate. Decision: `REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY`.

D-028 forbids variable-order-only continuation and selects exact word-level column-signature/popcount aggregation as EXP-055.

<!-- EXP-055-AUTHORITATIVE-FINAL -->
## D-029/D-030 — Reject exact column grouping as universal core and select EXP-056

D-029 records EXP-055 authority `results/exp_055/summary.json`; workflow `30820909775`; source head `c15b1bb94496ad629bf8911d30d47a7cbe792595`; artifact `8858805996`; ZIP SHA-256 `983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8`. Across 48 cases and 96 compiled plans, 248,832 scalar validations plus packed controls produced zero score, top-1, or packed mismatches and no truth-table representation. Ideal repeated/sign-related columns improved monotonically to 7.8125%/9.375% logical operations at n=64. However global p50/p90 operation fractions were 62.5%/250%, p50/p90 query-byte fractions were 63.64%/200%, dense/unique p50 was 250%, and 21 cases had no positive compile amortization. Projected logical 405B-Q4 storage peaked at 0.7597 TiB and passed its isolated Gate. Decision: `REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY`.

D-030 retains exact grouping as an auxiliary optimization only when real weight extraction proves repetition. EXP-056 tests automatically derived exact prototype-plus-sparse-residual dictionaries, charging prototype, membership, residual, compile, and query costs.

<!-- EXP-056-AUTHORITATIVE-FINAL -->
## D-031/D-032 — Reject prototype-residual dictionaries as universal core and select real-weight extraction

D-031 records EXP-056 authority `results/exp_056/summary.json`; workflow `30823042599`; source head `73655fc216340d9bd1d452d779951c28ac1b3d3b`; artifact `8859665874`; ZIP SHA-256 `9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8`. 56 cases and 448 exact plans produced 1,161,216 scalar validations with zero score, top-1, or packed mismatch and no runtime table. Repeated columns reached 7.8125% logical work at n=64, sparse prototype perturbations reached 10.9375%, and sign clusters reached 15.625%. The universal Gate failed: p50/p90 operations 62.5%/131.25%, p50/p90 bytes 62.115%/169.643%, dense/unique p50 123.4375%, and 24 cases did not amortize. Projected logical 405B-Q4 storage peaked at 0.6791 TiB and passed only its isolated Gate. Decision: `REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY`.

D-032 stops synthetic dictionary elaboration until real pinned checkpoint matrices are measured. EXP-057 extracts exact FP bit patterns and deterministic Q8/Q4 weight columns from unchanged pinned TinyStories checkpoints and applies the retained EXP-055/056 analyzers with full per-matrix accounting.

<!-- EXP-057-AUTHORITATIVE-FINAL -->
## D-033/D-034 — Reject exact real-weight grouping/dictionaries and select algebraic-rank Gate

D-033 records EXP-057 authority `results/exp_057/summary.json`; workflow `30824957941`; source head `cf9d7099dc11b22ce24ba6e096712d5da1bc3729`; artifact `8860450501`; ZIP SHA-256 `7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0`. Three unchanged pinned TinyStories checkpoints exposed 327 learned tensors, including 153 analyzed 2-D tensors and 54,205,312 named 2-D scalars. Across all 144 named dense-projection matrices, exact repeated or sign-related column coverage was zero in loaded FP32, deterministic Q8, and deterministic Q4. Q4 retained p50/p90 logical operations of 82.8918%/85.8398%, p50/p90 query bytes of 329.0244%/490.6845%, and median/p90 exact residual density of 81.4087%/84.2834%. Even the best real matrix retained 70.2866% operations. Storage projection passed narrowly at 0.9300 TiB and compile amortization passed at 377 queries. Decision: `REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY`.

D-034 closes exact column repetition and sparse residual dictionaries as a universal direction for the measured real checkpoints. EXP-058 tests a different exact representation: algebraic low-rank factorization. Modular-rank certificates on the same pinned Q4 matrices establish exact rank lower bounds before any factorization is promoted.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## D-035/D-036 — Reject exact low-rank factorization and select shift-displacement Gate

EXP-058 authority: `results/exp_058/summary.json`; workflow `30826618962`; source head `8ae03de4cc34317b5536aed42b9b8c22f98c88ea`; workflow merge `3730d6ce8ca89df347079c366a91bcad4d904a85`; artifact `8861905858`; ZIP SHA-256 `851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae`. All 144 registered Q4 dense projections were proven full integer/rational rank with zero certificate, control, registration, or EXP-057 checksum mismatch. Prime 251 certified 143 matrices and prime 257 certified one. Favorable conventional exact two-factor operation and storage lower bounds were p50/p90 200%/200%. Decision: `REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES`.

Full rank does not rule out fast full-rank structured transforms. EXP-059 therefore tests exact zero-fill and cyclic diagonal/anti-diagonal shift-displacement rank rather than another factor search.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## D-037/D-038 — Reject exact shift-displacement structure and select zero-sparsity Gate

EXP-059 authority: `results/exp_059/summary.json`; workflow `30840432745`; source head `cdae6160cd87b537e2f318c16430619736c7c9d9`; workflow merge `82979e393a87845c4c757ce5dfd3fadc4e701d92`; artifact `8866573958`; ZIP SHA-256 `61d0c24ccacd310d7d0e7600cc926a882c74281827d524c4880c6715fad8800d`. Four registered exact displacement operators were certified for every two-dimensional tensor. For all 144 dense projections, even the favorable selected displacement rank was 100% of the minimum dimension. Favorable query lower bounds were p50/p90 100%/100%, generator storage was 200%/200%, and the best real matrix still required 100% query work and 125% storage. All controls, registration, and EXP-057 Q4 checksums passed. Decision: `REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES`.

D-038 closes the tested Toeplitz/Hankel/circulant-like exact route. EXP-060 measures a simpler orthogonal possibility: exact scalar zeros and all-zero blocks in the same pinned Q4 matrices, with index and byte costs fully charged.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## D-039/D-040 — Reject exact Q4 zero-sparsity streaming and select activation-sparsity Gate

EXP-060 authority: `results/exp_060/summary.json`; workflow `30841671707`; source head `bf89d087343a4790202126c34562ca0344ebe452`; workflow merge `5f2af394180beaf3e5b5b8c7386d2becdf7eb8e7`; artifact `8867145590`; ZIP SHA-256 `5e5255dbedd779b734876faa027cd2bf5e4a1b00ece7f28cbf35f428fb9a0b05`. Across 144 pinned real-Q4 dense projections, exact zero-scalar fraction was p50 17.7612%, p90 20.3674%, and maximum 30.1041%. Favorable row-run selection left p50/p90 operation fractions 82.2205%/85.0586%, while indexes and run metadata raised query bytes to 150.9277%/200.8606%. The best matrix still required 69.896% operations and 190.118% bytes. Reconstruction, controls, registration, and EXP-057 checksums passed. Decision: `REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY`.

D-040 closes static exact-zero weight streaming for this measured Q4 population. EXP-061 moves to runtime state and measures exact zeros at inputs to every dense projection during causal prefill and decode.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## D-041/D-042 — Reject causal exact activation-zero skipping and select attention-probability Gate

EXP-061 authority: `results/exp_061/summary.json`; workflow `30843404056`; source head `15097a9b0323aa992679214173aaac0e7a98821c`; workflow merge `44c3d6691d78714dc975e46e19bb8fdfe97a22cf`; artifact `8867731496`; ZIP SHA-256 `a01d31b012badd7d06087df576279b852db07813a0c7fb50d65c3a7283e9ca65`. Across 18 pinned model/prompt cases, 1,152 hooked generation tokens matched unhooked references exactly. 147 unique projection registrations produced 56,448 calls, including 54,684 warm-decode calls and 12,165,888 warm-decode input scalars. Exact positive/negative-zero count was zero in prefill, first decode, and warm decode. A full zero scan raised weighted p50/p90 operation fractions to 100.00199%/100.390625% and query-byte fractions to 100.00404%/101.56555%. Decision: `REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY`.

D-042 closes exact zero-coordinate skipping at registered dense-projection inputs for this causal population. EXP-062 measures a different runtime structure: exact non-mask zero probabilities after attention softmax and the fully accounted effect on Value accumulation and total Transformer work.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## D-043/D-044 — Reject exact non-mask attention-zero skipping and select KV equivalence Gate

EXP-062 authority: `results/exp_062/summary.json`; workflow `30844873182`; source head `c38baa187e41760ef07676326c6a14f08635acc3`; workflow merge `891868c186eb22869925ad20cba43ef32d371589`; artifact `8868287407`; ZIP SHA-256 `497816dcca7e6b8c40e9222ed8511efa266fe2358aab847a93795d7c04637390`. Across 18 causal cases, 1,152 observed forwards and 9,216 attention rows matched reference generation with zero registration/control mismatch. Causal and local-window mask zeros were excluded. Warm decode contained 2,564 exact non-mask zeros among 8,404,224 eligible probabilities: aggregate 0.030508%, weighted p50 0%, p90 0.075301%, and maximum single-row 7.1629%. QK, softmax, probability scan, metadata, unchanged Linear work and bytes were charged, yielding whole-model p50/p90 operation fractions 100.0484%/100.1541% and byte fractions 100.0930%/100.3031%. Decision: `REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY`.

D-044 closes exact post-softmax zero skipping for this measured population. EXP-063 tests a separate exact reuse condition: bit-identical cached Key and Key-Value vectors at causally eligible warm-decode positions, which could reuse QK scores and identical contribution products without approximation.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## D-045/D-046 — Reject exact cached-KV equivalence reuse and select real-Q4 output-row Gate

EXP-063 authority: workflow `30846082964`, source `979bde3a23b76270f740740fbf511c7f90900a7c`, merge `488fa0e3785885bbcea25681aae55bb361fa0f84`, artifact `8868770832`, ZIP SHA-256 `b900a7019d8527d6f67d0eb412bb2fb7a0331188d84cd74444ca10762a105a14`. Across 18 causal cases, 1,152 forwards and 147,456 layer/head rows, exact Key duplicates and exact Key-Value duplicates were both zero. Token, registration and control mismatches were zero. Fully accounted warm whole-model p50/p90 operation fractions were 100.0211%/100.0273%; query-byte fractions were 106.2629%/119.4005%. Decision: `REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY`.

D-046 opens EXP-064: inspect exact identical, sign-related, and prototype-plus-sparse-delta output rows in the pinned real Q4 dense matrices. This is the row-space dual not covered by EXP-057's column grouping.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## D-047/D-048 — Reject exact Q4 output-row reuse and open Kronecker-rank Gate

EXP-064 authority: workflow `30869720552`, source `a6371c39d85dc39669b98eac6125d9c3bbf4a5dc`, merge `3716584078a91ae307b11b4bf1b2662e1511e9c9`, artifact `8877450455`, ZIP SHA-256 `99c634bd4fb3903d32a1ed45fada7853ea4e1d199b375c129d1d4b8da4f39cb8`. All 153 two-dimensional tensors and 144 dense projections matched frozen Q4 checksums; all 1,683 plans reconstructed exactly. No dense matrix contained identical or sign-related rows. The deployable selector retained dense execution for 140/144 projections and exact sparse-delta plans for four. Dense-projection p50/p90 operation and query-byte fractions were all 100%; the best single matrix reached 70.522% operations and 93.811% bytes. Decision: `REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY`.

D-048 opens EXP-065: exact Kronecker-rearrangement rank certificates on the same pinned real-Q4 matrices.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## D-049/D-050 — Reject exact Kronecker sums and open Tensor-Train/MPO bond-rank Gate

EXP-065 authority: workflow `30870558294`, source `22fd41697979f0e5aeb570880714a47958270d7f`, merge `2e512e91b5bfcd5e30a19ef163a6438221a134dc`, artifact `8878551394`, ZIP SHA-256 `cf5bfcc53bda4117430c0856b6989704e79bb34fb52c9a4f81869bf20233155d`. All 153 two-dimensional tensors and 144 dense projections matched frozen Q4 checksums. Across 6,108 ordered factorization plans, selected two-prime certificates had zero witness mismatch. Every dense projection selected a full-rank 4-row rearrangement. Favorable lower-bound p50/p90 operation fractions were 203.891%/215.385%; storage fractions 100.234%/101.042%. Decision: `REJECT_REAL_Q4_KRONECKER_RANK_AS_CORE_RETAIN_TENSOR_CERTIFIER_AUXILIARY`.

D-050 opens EXP-066: exact Tensor-Train/Matrix-Product-Operator unfolding ranks, which strictly generalize one-cut Kronecker structure.

<!-- EXP-066-072A-AUTHORITATIVE-CATCHUP -->
## D-051 through D-056 — Close EXP-066 through EXP-071

- D-051 rejects exact classical TT/MPO as core: p50 operation/query `3.8941%/2.9984%`, but p50/p90 static lower bounds `11.0524%/22.9883%` and projected 405B lower-bound representation about `14.315 GB`, above 8 GiB. Retain the certifier only.
- D-052 rejects exact joint Q/K/V row/common-right reuse: zero reusable rows across 10,752 rows; p50/p90 work 100%; storage above 107%.
- D-053 rejects absolute-unread global demand certificates: even with preceding work and the winning output-head row free, output-head-only p50/p90 mandatory fractions were `13.7697%/19.2524%`.
- D-054 rejects causal exact temporal-span replay: p50/p90 mandatory full passes 100%, no exact replay hits, and p50 basis cache 391.97% of Q4 projection storage.
- D-055 rejects exact Q4 local-pattern tables: p50/p90 operations `88.4856%/91.4423%`; query/static `111.0294%/112.7907%`.
- D-056 records EXP-071 as insufficient for an impossibility claim. CKL18 covers 0/9 tensor families under the full 8 GiB side state; no model-wide direct sum or finite constants were established.

Status: REJECTED/RETAINED AUXILIARY AS RECORDED; EXP-071 IS A CLAIM RESTRICTION, NOT FEASIBILITY EVIDENCE.

## D-057 — Reject self-contained exact Q4 DAG as universal hot core

EXP-072A authority: `results/exp_072a/summary.json`; source `468f297925e10bdc541fe48f19c2f72a1e3f5e14`; evidence commit `f9ac26befb01fd9a71c7c6e1efed4c4b4df31389`; deterministic core SHA-256 `112f0490e9e21efc8df3da04f71d92adacba4c493d2eae07d94dbf0960ac8c66`.

Exact outputs on standard-basis inputs uniquely recover the coefficient matrix. Exhaustive finite controls produced 272 unique signatures for 272 matrices with zero collision/control failure. The arbitrary 405B Q4 class requires `1,623,396,974,592` worst-case artifact bits (`188.98828125 GiB`) before overhead, `23.62353515625x` the 8 GiB hot allowance. The former 10% static threshold alone is `18.898828125 GiB`.

Decision:

```text
REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE
RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY
```

This does not rule out an online runtime that queries the original checkpoint or another lossless cold representation. Such a runtime belongs to a different class and must charge all cold probes.

## D-058 — Calibrate the private target before another cold-backed core Gate

EXP-073 is a sanitized, read-only target inventory followed—only with separate authorization—by same-machine storage, transfer, and native 4B Q4 baselines. It may establish resource facts only; it cannot promote a VORTEX mechanism. Private connection details remain out of repository evidence.

Status: ACTIVE NEXT CALIBRATION GATE; PHASE D STILL NOT TESTED.

## D-059 — Accept EXP-073 Stage 1 inventory and hold Stage 2 for separate authorization

EXP-073 Stage 1 collected a complete sanitized read-only inventory with no remote mutation, no saved private identifier, and zero validation/checksum mismatch. The target exposes one Quadro M5000 with 8,192 MiB VRAM and maximum PCIe Gen2 x16, 23.4983 GiB host RAM, and 97.6183 GiB free on the root filesystem. fio and nvcc are not available; Python 3.12.3 and an active Ollama 0.30.6 service are present.

The registered packed 405B-Q4 information is 91.3700 GiB larger than current root free capacity before overhead. The favorable Gen2 x16 signaling ceiling implies at least 25.3656 seconds for one packed-Q4-equivalent transfer, but physical bandwidth and loaded-link behavior are not measured.

Decision:

```text
COMPLETE_SANITIZED_READ_ONLY_TARGET_INVENTORY
HOLD_STAGE_2_PENDING_SEPARATE_AUTHORIZATION
```

Authority: `results/exp_073/summary.json`; source `d3b1d2e4dd08e73781c969814cb4d181377a054d`; checkout-stable evidence `4e35afb9648dc0c513c3a90c32d604f4c0b0fd21`; core SHA-256 `aa9cae0457a6b92fcb75da35fedc1a2a2a9f3da115808d341a5a498ca4722da2`.

Status: STAGE 1 COMPLETE; EXP-073 ACTIVE; PHASE-D RUNTIME VALIDATION, E6, AND E7 NOT ACHIEVED.

## D-060 — Reject MTP-1 plus expert paging; revise long-block candidate

EXP-074 authority: `results/exp_074/summary.json`; source
`8abc06e73c884b839927cf41d5f4fa6cbb8fc051`; evidence
`c1d778af011672ec7fadfa66935ba2548de8e115`; deterministic core SHA-256
`404b43088448eaafc3f3d9631cdc3271dc9ebc16b635e27c9211cf9aa0459a65`.

MTP-1 plus expert paging retained `10.0x` the 1B baseline-equivalent target
traffic under free drafting and fixed expert reuse, failing the `1.2x` p50 Gate.
The ideal fixed-route/free-proposal long block reaches the Gate at nine accepted
tokens, while independent-uniform expected routing requires 98.

Decision:

```text
REVISE_MTP1_AND_EXPERT_PAGING_INSUFFICIENT_REQUIRE_LONG_CAUSAL_PROPOSAL_AND_ROUTING_LOCALITY_GATES
```

The MTP-1/paging combination is rejected as a 1B-class core. Reference block
accounting and the exact verifier remain auxiliary. A metadata-only MTP surface
audit and then a small-checkpoint accepted-prefix Gate are required before any
35B/122B model download or scheduler implementation. The Qwen MoE surrogate
does not replace the dense 405B acceptance target.

Status: REVISE; E1 REFERENCE ACCOUNTING ONLY; PHASE D/E6/E7 NOT ACHIEVED.

## D-061 — Accept the pinned native-MTP surface and open small acceptance Gate

EXP-075 authority: `results/exp_075/summary.json`; source
`f85ac583a129070247992987d1b3c63634e6447f`; evidence
`2fcf7315cf9da491a5ca361536eb0f07e325e74c`; deterministic core SHA-256
`2515bb53a0e2967cfd23e15d18720142337c7d25dc658db057e86e1aa45b5674`.

At the pinned official `Qwen/Qwen3.5-0.8B` revision, the config declares one
MTP hidden layer and the weight index contains all 15 registered `mtp.*` keys.
Pinned vLLM source exposes the registered Qwen3.5 MTP config rewrite, model
classes, loader remap, quantization inheritance, and recursive step surface.
Five metadata controls passed.

Decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

This confirms static surface availability only. It does not confirm that any
proposal is accepted, that recursive hybrid state is correct, that a quantized
artifact preserves the head, or that the runtime works on current hardware.
No checkpoint weight or server command was used.

Status: METADATA PREREQUISITE PASS; E1 STATIC INTERFACE; EXP-076 AUTHORIZED FOR
PREREGISTRATION; PHASE D/E6/E7 NOT ACHIEVED.

## D-062 -- Reject native-MTP long blocks as the surrogate core

EXP-076 authority: `results/exp_076/summary.json`; source
`5e331137f8e03250cc74aa796abbf69f49ef87a5`; evidence
`55b79937c1f21887ae76b7e56ad61ba7dde8322a`; deterministic core SHA-256
`199db6f8fc0dedd32d7b38be8ff1d05c29aced3388a87ddecccd1235038bd22d`.

The build split selected `K=4`. The held-out population accepted p05/p50/p95
prefixes of `0/4/4`, against shape-required p05/p50 minima of `9/11`. Two of
18 evaluation cases accepted no draft token. The reference recorded zero
future-target reads, wrong accepts, committed-state mutations, commit-replay
mismatches, and rollback-recompute mismatches.

Decision:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

The 35B-A3B route-union trace is not promoted. Do not rescue the branch with
post-selected prompts or K, another small Qwen checkpoint, quantization, a
35B/122B download, or a page scheduler. Return candidate selection to a
materially different query-time information source. EXP-073 Stage 2 remains a
separately authorized calibration, not a core mechanism.

Status: SCIENTIFIC REJECTION AT E1 CPU OBSERVATION; REFERENCE INTEGRITY PASS;
NO PHASE D/E2-E7 OR DENSE-405B EVIDENCE.

## D-063 -- Reject activation-norm Fractal MLP at the 10% favorable ceiling

EXP-077A authority: `results/exp_077a/summary.json`; source
`33fed17ed6abed7c8c14eec543efc620e4fe537d`; evidence
`0970c6626ff848c5026b684b3e2d1bb479e96603`; deterministic core SHA-256
`e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411`.

The unchanged target replayed all 192 registered causal positions with zero
mismatch. The oracle used the complete current SwiGLU intermediate and free
selection, yet retaining 358/3,584 channels (`9.988839%`) produced held-out
top-1 agreement `71.5278%`, mean KL `0.884161`, and p95 KL `3.080752`.
All six families failed the registered `95%` top-1 minimum. At 20%, population
top-1 agreement was still only `83.3333%`.

Decision:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

Do not rescue this score with post-selected fractions, layers, prompts,
channel blocks, a trained router, physical sparse kernels, or a larger model.
A continuation must supply a materially different execution dependency that
recovers or amortizes omitted nonzero contribution under fully charged costs.

Status: SCIENTIFIC REJECTION AT E1 FAVORABLE-ORACLE OBSERVATION; CONTROL PASS;
NO PHASE D/E2-E7 OR DENSE-405B EVIDENCE.

## D-064 -- Open a corrected construction-amortization Gate for frozen tangent macroblocks

The complete anchor-conditioned SwiGLU mapping is a materially different
recovery dependency from EXP-077A channel selection. However, counting only its
hidden-by-hidden hot application omitted the cost of constructing
`W_down diag(c_anchor) W_up`.

EXP-078A therefore freezes the direct-construction equation before model
execution. The 0.8B screen requires `13,821/6,249` p50/p05 hot positions; the
nine-path 122B surrogate screen requires `115,299/26,354`. A seven-position
unchanged-checkpoint run is authorized only as a cheap early-failure Gate.
Perfect survival is right-censored, not promotion.

Decision:

```text
PREREGISTER_EXP_078A_FROZEN_TANGENT_MACROBLOCK_LIFETIME_GATE
```

Status: SOURCE/CONTRACT CONTROLS PENDING; NO MODEL RESULT OR TARGET-SERVER ACTION.

## D-065 -- Reject frozen tangent macroblocks and require a new delta-construction source

EXP-078A reproduced the unchanged target with zero baseline mismatch, but the
complete frozen anchor operator failed at the first later token for all 18
held-out cases. Top-1 agreement was `0/126`; mean/p95 KL was
`14.898423/25.335417`; valid-prefix p05/p50/p95 was `0/0/0`.

Decision:

```text
REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH
```

Do not spend work on longer traces, rank sweeps, selected layers/prompts,
sentinels, or physical macro kernels for this path. A delta-updated operator is
not promoted; it requires a fresh E0 information source and fully charged update
and repair equations.

Status: SCIENTIFIC REJECTION AT E1; CONTROL PASS; NO PHASE D/E2-E7 OR
DENSE-405B EVIDENCE.

## D-066 -- Preregister the DCT block-zonotope causal proof-state Gate

EXP-079A replaces frozen prior-token operators with current-token cold queries
and a fail-closed correlated proof state. Its favorable dense-405B shape plan
charges p50 logical traffic/operations of `1.08883712%/0.976002844%`; all other
operators, dual oracle selectors, nonlinear propagation, and failed fallback
are free. This is sufficient for E0 admission but not promotion.

Decision:

```text
PREREGISTER_EXP_079A_CAUSAL_PROOF_STATE_GATE
```

Status: SOURCE/CONTRACT CONTROLS PENDING; NO MODEL RESULT, TARGET-SERVER ACTION,
PHYSICAL SPEED, 8 GIB PEAK, 122B/405B RUN, OR E2-E7 EVIDENCE.

## D-067 -- Reject the fixed DCT block-zonotope proof-state path

EXP-079A reproduced all 192 unchanged decisions and met the p50 logical traffic
and operation budgets. Its two free oracles nevertheless failed independently:
the quality-optimal row blocks yielded `6/144` held-out top-1 with mean/p95 KL
`7.544862/12.616226`, while the proof-optimal blocks left minimum local radius
`48.663918x/57.778748x` the signal.

Decision:

```text
REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH
```

Do not invest in a selector, nonlinear propagator, fallback engine, fixed-basis
or block sweep, kernel, larger checkpoint, or target-server run for this path.
The next E0 proposal must provide new information about the dense residual, not
rename its enclosure.

Status: SCIENTIFIC REJECTION AT E1; CONTROL/BUDGET PASS; NO PHASE D/E2-E7 OR
DENSE-405B EXECUTION EVIDENCE.

Authority: `results/exp_079a/summary.json`; source `e0c661e`; evidence
`38bfd6c`; core
`c8d794bea8f17b31e40b9f667836d2198ce80137e23080260f81ea6866112eeb`.

## D-068 -- Preregister Hyperblock exact rectangular arithmetic Gate

EXP-048/049/050 already reject hard Jacobi, target-only fixed-point, and fixed
external drafting. EXP-080A therefore does not rerun proposal quality. It
tests whether exact subcubic arithmetic across a granted future block can close
the dense-operation budget that those experiments left unchanged.

Decision:

```text
PREREGISTER_EXP_080A_HYPERBLOCK_EXACT_RECTANGULAR_MULTIPLICATION_GATE
```

The authoritative constructive arm is standard recursive Strassen with every
scalar multiply/add, padding tile, and output accumulation charged. Passing a
unit-constant exponent-only control is insufficient. No backend, checkpoint,
or target-hardware work is authorized before the arithmetic Gate survives.

Status: E0 PREREGISTERED; NO RESULT OR CORE PROMOTION.

## D-069 -- Reject standard recursive Strassen Hyperblocks as the core

EXP-080A passed all 80 exact integer controls and granted perfect future
activations, zero proposal cost, one target weight sweep, and the best classical
leaf width. Nevertheless, no registered block length passed both final
fractions. At `K=16,384`, logical traffic was `0.006103516%`, but constructive
arithmetic was `36.111580%` against a `1.185185%` allowance: a `30.469145x`
miss before physical overhead.

Decision:

```text
REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY
```

Retain the exact rectangular cost model and unit-constant exponent row as
falsification/target-setting auxiliaries. Do not build standard Strassen
kernels or use the non-constructive exponent control as evidence of an
executor. This decision does not reject every exact low-constant FMM algorithm;
reopening requires an explicit fully charged construction beyond the measured
gap and a separately credible causal future-block source.

Status: SCIENTIFIC REJECTION AT E1 SYNTHETIC/REFERENCE; NO MODEL, PHASE D,
TARGET HARDWARE, PHYSICAL KERNEL, OR E2-E7 EVIDENCE.

Authority: `results/exp_080a/summary.json`; source `7f1c661`; evidence
`98e9089`; core
`7578c4c9f463da8135f3c320df9d7fb920ffc172d31fdd2f60b30af9778280ce`.

## D-070 -- Preregister syndrome-recovered nonlinear lookup MatVec

The OMv boundary rules out claiming a universal arbitrary-query improvement.
EXP-081A instead defines an exact conditional fast path: a nonlinear candidate,
rank-eight syndrome recovery, six independent field fingerprints, and unchanged
fallback. Its success variable is the measured residual-code coverage of real
causal projection inputs.

Decision:

```text
PREREGISTER_EXP_081A_SYNDROME_RECOVERED_LOOKUP_MATVEC_GATE
```

Only synthetic controls and the pinned small-checkpoint necessary Gate are
authorized. No decoder integration, backend, kernel, larger model, or hardware
run is authorized before `99.75%` held-out exact recovery coverage and the
quality/resource Gates pass.

## D-071 -- Reject the tiny residual-code fast path and change the proof target

EXP-081A passed all 321 finite-field correctness and fail-closed controls. Its
pre-fallback resource equation also passed. The necessary held-out premise did
not: exact rank-eight residual-code coverage was `8.681672%` instead of
`99.75%`, every family was below `10.72%`, and favorable corrected relative-L2
p50/p95 was `0.351269/1.213828` instead of `0.01/0.05`.

Decision:

```text
REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH
```

Do not tune tree depth, leaves, residual rank, field, layer, projection, prompt,
or tolerance around this path. The observed fallback population yields derived
logical traffic `92.244986%` of dense and cannot be repaired by a kernel.

The next E0 direction may change the proof target from exact local MatVec
recovery to an exact final-token/fixed-RNG decision certificate. It is not yet
an experiment or promoted mechanism. Its free-oracle minimum-page Gate and full
resource equation must be committed before implementation.

Status: SCIENTIFIC REJECTION AT E1; STRUCTURAL ALGEBRA CONTROL PASS; NO
OPERATION REPLACEMENT, TARGET HARDWARE, PHASE D, OR E2-E7 EVIDENCE.

Authority: `results/exp_081a/summary.json`; source `1d3e91f`; evidence
`ee9573d`; core
`8621f6357536b6fc3396872668484c52103e28d2af8291b96575bc4d007c2ccc`.

## D-072 -- Reject verification without a local result source

The post-EXP-081A final-decision direction was compared with EXP-068 before an
experiment number was opened. Absolute unread bounds are already closed, and
even deleting the entire registered 405B output head saves only
`0.520459999%` of non-embedding coefficient work.

Random linear trace verification itself is favorable: six checks project to
`0.058120260%` operations, `0.266838924%` traffic, and `0.427185 GiB` sidecar.
But the local dense proposer makes the total `100.058120260%` operations and
`100.266838924%` traffic. Delegation moves rather than removes target work.

Decision:

```text
REJECT_PROOF_CARRYING_TRACE_AS_A_STANDALONE_LOCAL_CORE
```

No implementation beyond a throwaway E0 equation prototype is authorized.

## D-073 -- Preregister exact differential spanning-tree MatVec

EXP-082A admits a full row/column Hamming tree rather than reopening the
bounded-prototype sweep from EXP-064. Each edge is an exact sparse integer
difference, and a low-weight tree gives a published subquadratic online MatVec
route for structured matrices.

Decision:

```text
PREREGISTER_EXP_082A_DIFFERENTIAL_SPANNING_TREE_GATE
```

Only the exact block-pattern MST lower bound is initially authorized. If
coefficient work alone exceeds the final fraction, stop before constructing a
tree or runtime. Status: E0 PREREGISTERED; NO RESULT OR CORE PROMOTION.

## D-074 -- Reject exact row/column differential spanning trees

EXP-082A evaluated the preregistered collision-free block-pattern lower bound
on 21 pinned Q4 matrices. The best-orientation weighted coefficient lower bound
was `1.562367394%`, or `1.318247489x` the `1.185185185%` target, before all
positive runtime and storage costs. Seventy-two exact small-matrix controls
passed, and independent replay matched the deterministic artifacts.

Decision:

```text
REJECT_DIFFERENTIAL_SPANNING_TREE_FROM_CERTIFIED_LOWER_BOUND
```

Per the frozen stop rule, do not construct or benchmark a terminal-only tree.
Synthetic-intermediate circuits are not rejected by this result, but receive
no experiment number until an E0 novelty audit distinguishes them from the
closed exact circuit/DAG families.

## D-075 -- Define the positive research milestone and versioned investigation map

The persistent research request is operationalized without weakening E7. A
`Certain Research Milestone` requires a materially new candidate to pass E0
and E1 and then perform actual fail-closed dense-operation replacement at E2
on the pinned small checkpoint. Negative family closures remain valuable but
do not satisfy this milestone.

Decision:

```text
ADOPT_VERSIONED_WAYFINDING_MAP_FOR_POSITIVE_E2_MILESTONE
```

The map and dependency-ordered child tickets live under
`.scratch/vortex-certain-milestone/`. Map creation does not promote a core or
authorize EXP-083, hardware, a large-model download, or a target-server action.

## D-076 -- Reject static synthetic intermediates as a renamed core

An exact synthetic Hamming/Steiner tree computes each node from a parent plus
an exact coefficient delta. It therefore expands into a static linear
straight-line program and is strictly less general than the shared synthetic
linear forms already archived in EXP-072B. Putting that program in cold
storage changes capacity placement but does not create a causal query-time
information source.

The EXP-082A terminal certificate cannot by itself reject Steiner nodes:
`SMT >= MST/2` gives only `0.781183697%`. The class still fails E0 promotion
because it is not novel and has no fully charged sub-`1.185185185%` static
operation/traffic schedule. Almost-all hypercube Steiner and binary
linear-circuit results provide strong adverse distributional evidence but are
not promoted to finite real-weight claims.

Decision:

```text
REJECT_STATIC_SYNTHETIC_INTERMEDIATE_TREE_OR_DAG_AS_NEW_CORE
KEEP_QUERY_ADAPTIVE_COLD_BACKED_INFORMATION_SOURCE_OPEN_AT_E0
```

Do not open EXP-083 or build a static synthesizer/runtime. The next admissible
work is an E0 equation and causal information-source search for a genuinely
query-adaptive cold-backed executor. Authority:
`docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

## D-077 -- Fix the query-adaptive cold-backed conservation frontier

For either logical operations or traffic, the complete favorable branch
equation is `R = g + kappa/N + rho*h + (1-rho)*(m+1)`. On the registered
405B non-embedding shape, even a free successful path needs
`98.814814815%` coverage to keep mean dense fallback below the p50
`1.185185185%` allowance. The known verifier alone raises that frontier to
`99.081653739%`; positive selector, decode, miss, or build costs raise it
again.

The same Gate requires at least `84.375x` query-time useful information
amplification with no overhead, or `108.891389x` after the known verifier.
Selecting raw Q4 pages without any coded information about omitted weights
cannot determine exact arbitrary `W*x`; a surviving class must supply a causal
checkpoint-derived code or certificate and charge it fully.

Decision:

```text
DERIVE_QUERY_ADAPTIVE_COLD_BACKED_FEASIBILITY_FRONTIER
REJECT_RAW_Q4_PAGE_SELECTION_WITHOUT_OMITTED_CONTRIBUTION_SOURCE
KEEP_CODED_CAUSAL_COLD_SOURCE_OPEN_FOR_INFORMATION_SOURCE_SEARCH
```

This resolves an E0 equation, not the missing mechanism. Do not assign
EXP-083, download a model, or enter hardware/runtime work. Authority:
`docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`.

## D-078 -- Admit committed-prefix residual coding to Gate design

Exact committed-prefix pairs contain query-specific checkpoint information:
orthogonalizing their inputs into `Q` and applying the same combinations to
their already computed images produces `Z = WQ` without target-future tokens
or another checkpoint scan. A later query is `x = Qa + u`; a legal runtime may
evaluate `Za`, reveal cold pages for `Wu`, and retain every unread page in a
sound residual enclosure or exact fallback.

The named class is **Causal Residual Atlas**. It is not exact temporal replay,
because a nonzero `u` is allowed but never ignored. It is not a fixed pilot,
because the code is formed only from the current request's committed prefix.
It is not raw page selection, because prefix images and block certificates
carry information about omitted contributions.

A favorable rank-16/64-column/0.2%-requested/64-token registered screen has
`1.085025716%` amortized traffic and `0.557958575%` amortized operations, but
requires `99.899840530%` traffic-governed certificate coverage. Real residual
concentration, full numerical propagation, page latency, and complete branch
memory are unverified.

Decision:

```text
IDENTIFY_CAUSAL_RESIDUAL_ATLAS_AS_A_CONCRETE_CODED_CAUSAL_COLD_SOURCE
REJECT_TABLE_ENUMERATION_AND_BOOLEAN_ABSORPTION_AS_DIRECT_NUMERICAL_SOURCES
AUTHORIZE_ONLY_PREREGISTRATION_OF_THE_CHEAPEST_REAL_WEIGHT_GATE
```

No Core Candidate is promoted and no EXP-083, runtime, model download, Ubuntu
action, or hardware work is authorized. Authority:
`docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`.

## D-079 -- Freeze the smallest real-weight Atlas falsification

The first real-weight Gate uses only the already pinned Qwen3.5-0.8B payload
and evaluation trace. Prompt-only top-16 bases are frozen before teacher
position 1. Layer-11 `q_proj` and `down_proj` enumerate all 16/56 contiguous
64-column pages under a native-anchored exact-reference center, for 1,296
candidate page runs across 18 prompts.

The target equation's `99.899840530%` coverage requires 18/18 token states,
3/3 per family, and 36/36 projection branches to retain unchanged greedy
top-1. The selected branch population must also meet mean/p95 KL
`<=0.02/0.05`. A single token failure gives only `94.444444%` coverage and
fires the scientific stop rule.

Decision:

```text
PREREGISTER_CAUSAL_RESIDUAL_ATLAS_CHEAPEST_REAL_WEIGHT_GATE
ON_PASS_AUTHORIZE_ONLY_CAUSAL_PAIR_AND_OUTWARD_BOUND_GATE
ON_FAILURE_REJECT_RANK16_PAGE64_CAUSAL_RESIDUAL_ATLAS_FAST_PATH
```

No result, experiment number, model execution, hardware action, or Core
Candidate promotion is recorded. Authority:
`docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`.

## D-080 -- Accept favorable one-page existence and open the legal pair/bound Gate

EXP-083A completed the exact frozen population on the unchanged pinned
Qwen3.5-0.8B checkpoint. It preserved top-1 for 18/18 token states, 3/3 in each
family, and 36/36 q/down branches. Mean/p95 KL was
`0.007225545020063708/0.037660752986209814`, all 234 controls passed, and the
separate model replay reproduced deterministic-core SHA-256
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`.

The favorable existence premise therefore survives E1 on this finite small-
model population. It is not a runtime result: the native dense output anchors
the center and exact-reference logits choose among every page. A post-hoc
minimum-radius selector fails 2/18 tokens, and the present bound remains at
least `19.852750x` looser than exact unread error even on selected branches.

Decision:

```text
ACCEPT_FAVORABLE_ONE_PAGE_EXISTENCE_AT_E1_ON_THE_PINNED_SMALL_MODEL
PROMOTE_ONLY_TO_A_PREREGISTERED_LEGAL_CAUSAL_PAIR_AND_BOUND_GATE
KEEP_RUNTIME_HARDWARE_SCALE_AND_E2_E7_CLOSED
```

No Core Candidate is admitted to production architecture and no private-server
or larger-model action is authorized. Authority:
`docs/research/EXPERIMENT_083A_CAUSAL_RESIDUAL_ATLAS_GATE.md`.

## D-081 -- Freeze a leakage-safe legal last-down certificate

The observed EXP-083A rows are quarantined from selector and bound design. A
new 24-prompt population was written and hashed before model execution. The
first executable slice is layer-23 `down_proj` at teacher position 1 because
only residual addition, final RMSNorm, and the tied LM head remain after it.

Two-pass causal Gram-Schmidt derives `Q_hat/Z_hat` from committed pairs only.
The target-free selector removes the page with maximum current residual energy.
A verified full-matrix operator bound encloses the unread residual; pair/native
rounding, final RMSNorm, and every LM-head row margin are outward. Failure to
prove a strict greedy winner executes dense completion or aborts.

The fully charged favorable component equation, including a 20M-token static
compile amortization, is `1.093706271797%` traffic and `0.928746620379%`
operations. Traffic requires `99.908521086612%` coverage, so the finite Gate
permits no failure.

Decision:

```text
PREREGISTER_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE
USE_ONLY_NEW_24_PROMPT_POPULATION_FOR_THE_SCIENTIFIC_DECISION
ON_PASS_AUTHORIZE_ONLY_BACKWARD_LAYER_AND_POSITION_EXPANSION
ON_FAILURE_REJECT_THE_LEGAL_ATLAS_PRIMARY_PATH
```

No new model result, experiment number, E2 claim, hardware action, or scale
claim is recorded. Authority:
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`.

## D-082 -- Reject the legal Causal Residual Atlas primary path

EXP-083B executed the frozen untouched population exactly once and stopped on
its first row. The pair compiler reached rank 16 and the maximum-residual-
energy selector legally chose page 0 before native current output or logits
existed. The final certificate was unresolved, exact dense completion replayed,
and the required zero-fallback Gate failed.

All 19 controls passed with zero leakage, malformed state, false accept, or
dense-replay mismatch. Independent verification used zero model forwards and
rebuilt deterministic-core SHA-256
`57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.
The frozen down radius was `23.4205200666`, dominated by the unread
common-spectral radius `16.3298572850` and pair radius `6.2376633562`.

A post-hoc necessary-condition audit cannot alter the rejection: actual
candidate/native hidden separation was `38.9079080403`, while the top-two
row-margin certificate allows less than `4.1978252811` even with rounding set
to zero. Therefore tightening only the large RMSNorm implementation term does
not reopen the same certificate family.

Decision:

```text
REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH
DO_NOT_OPEN_BACKWARD_LAYER_POSITION_OR_E2_ATLAS_WORK
REQUIRE_A_NEW_CAUSAL_INFORMATION_SOURCE_AND_E0_EQUATION
KEEP_HARDWARE_SCALE_AND_E2_E7_CLOSED
```

Authority: `results/exp_083b`,
`docs/research/EXPERIMENT_083B_LEGAL_PAIR_OUTWARD_LAST_DOWN_GATE.md`, and
`docs/research/EXP083B_POSTHOC_NECESSARY_CONDITION_AUDIT.md`.

## D-083 -- Reject the two constructible post-Atlas dual sources at E0

The post-Atlas search changed the proof target from full intermediate vectors
to signed decision functionals. Exact primal and dual images determine two
terms, but a bilinear cross residual `r^T W u` remains. The perturbation
`Delta W=lambda r u^T` proves that the two image sets alone cannot determine
that residual for arbitrary operators.

The raw-checkpoint dynamic constructor fails the registered 64-token equation:
one fresh full-model dual direction costs `1.5625%`, above the complete
`1.185185185%` allowance with every other term free. The static final-slice
constructor needs a favorable `12.720703125 GiB` table per last down and a
`6.765979992%` full scan. The EXP-083B post-hoc direction screen resolves zero
of `248,319` competitors under both the actual and frozen radii.

Decision:

```text
REJECT_FORWARD_TRACE_ONLY_POST_ATLAS_SOURCE_AS_ATLAS_NORMAL_FORM
REJECT_DYNAMIC_CAUSAL_DECISION_DUAL_CODE_UNDER_REGISTERED_64_TOKEN_CONSTRUCTOR
REJECT_STATIC_FULL_VOCABULARY_DUAL_SCAN_AS_CORE
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_HARDWARE
KEEP_ONLY_A_CONCRETE_LOSSLESS_SUBDENSE_CROSS_RESIDUAL_SOURCE_OPEN
```

This is a scoped E0 closure, not a universal online bilinear-query lower bound.
Authority: `docs/research/E0_POST_ATLAS_CAUSAL_INFORMATION_SOURCE_AUDIT.md`.

## D-084 -- Reject matrix-local separable cross-residual codes

The Bilinear Cross Residual audit generalized fixed primal/dual bases to
arbitrary matrix-local binary linear covering codes. Exact cached images have
rank `a*n + m*b - a*b`; every remaining query is repaired from the product of
the two coordinate residual supports.

The rational sphere-covering witness `(rate,radius)=(3/10,3/16)` has
`rate + H_2(radius) < 1`. Granting the complete `8 GiB` hot allowance to a
single binary bit plane still forces `6,141,198,336` raw cross-coordinate
probes, `1.52104775688%` of the registered non-embedding population and
`1.28338404487x` the complete p50 budget before all other costs.

Decision:

```text
REJECT_MATRIX_LOCAL_SEPARABLE_LINEAR_RESIDUAL_CODE_AS_CORE
DO_NOT_REOPEN_ATLAS_WITH_LARGER_OR_COVERING_BASES
KEEP_NONSEPARABLE_GLOBAL_OR_CAUSALLY_RESTRICTED_SOURCE_OPEN
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_HARDWARE
```

The result assumes matrix-local image state and a Cartesian tuple of arbitrary
per-matrix query pairs. General nonlinear/cross-matrix data structures,
word-packed probes, and real causal reachability remain unproved. Authority:
`docs/research/E0_BILINEAR_CROSS_RESIDUAL_SEPARABLE_CODE_BOUND.md`.

## D-085 -- Localize globally mixed linear advice without claiming direct sum

For global advice rowspace `U`, exact systematic-linear query recovery writes
`q_i=a_i+e_i`. Because `q_i` is supported only on matrix block `i`, its outside
residual equals the outside part of `a_i`. Therefore zero-outside-probe advice
is the shortened space `U intersect V_i`, whose dimensions direct-sum inside
`U`; a fixed outside support `T` increases the projected local dimension by at
most `|T|`.

This does not collapse query-dependent supports. The diagonal space
`{(x,x)}` has full projection into two blocks but zero shortened dimension in
each, proving that summing projection ranks is invalid. A finite global
span-to-cover argument survives this issue but certifies only `6,407,133`
coefficient uses (`0.001586914271%`), while the target permits
`4,785,160,264.82`. The allowance is `746.8489x` the lower bound.

Decision:

```text
ESTABLISH_JOINT_ADVICE_LOCALIZATION_IN_THE_SYSTEMATIC_LINEAR_MODEL
REJECT_FREE_CROSS_MATRIX_PROJECTION_REUSE
DO_NOT_CLAIM_TARGET_SIZED_DIRECT_SUM_OR_GENERAL_IMPOSSIBILITY
KEEP_GLOBAL_NONLINEAR_AND_DATA_DEPENDENT_STRUCTURES_OPEN
OPEN_CAUSAL_QUERY_RESTRICTION_CERTIFICATION_AT_E0
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_HARDWARE
```

This is a durable accounting theorem, not a Core Candidate and not increased
mission feasibility. Authority:
`docs/research/E0_CROSS_MATRIX_ADVICE_LOCALITY.md`.

## D-086 -- Freeze the causal bilinear span rank Gate

Actual causal rank-one query tensors can invalidate an arbitrary Cartesian
lower-bound population only if their restriction survives the complete
fallback equation. For a factor-scanned ledger, one aligned basis tuple over
all 883 matrices has `39,109,888` left/right coordinates. After favorable
two-byte factors, full metadata, the retained six-check verifier, and 46
dense-equivalent build passes over 20M service tokens, dimension 23 consumes
`1.159413233%` traffic and `0.281295494%` operations; dimension 24 is already
traffic-infeasible.

For any held-out population of rank `R`, even the best post-hoc
`B`-dimensional subspace misses at least `R-B` rows. On the frozen 36-row last-
down screen, remaining traffic permits only four registered-down-equivalent
fallbacks. Certified rank 28 therefore rejects every `B<=23` factor-scan
ledger.

Decision:

```text
DERIVE_B23_SHAPE_LIMIT_FOR_A_FULL_FACTOR_SCAN
PREREGISTER_LAST_DOWN_CAUSAL_BILINEAR_RANK_GATE_ONLY
DO_NOT_CALL_MODULAR_NONINCREASE_AN_EXACT_HIT
KEEP_PAIR_EXTRACTION_AND_NATIVE_NUMERICAL_RECONSTRUCTION_UNSOLVED
DO_NOT_ASSIGN_AN_EXPERIMENT_NUMBER_OR_PROMOTE_HARDWARE
```

The current pair extractor and local result source remain favorable free
oracles, so no Core Candidate survives E0. Authority:
`docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md`.

## D-087 -- Reject the frozen causal bilinear factor-span ledger

EXP-084A executed the exact E1 Gate on the pinned unchanged Qwen3.5-0.8B
last `down_proj`. A first batched-capture attempt failed closed with zero query
rows and was preserved. The authoritative sequential committed-prefix run
kept all scientific settings fixed and passed 114 controls.

All 24 build queries were independent under primes `65521/65519/65497`; the
ledger stored the first 23 exact independent rows. The first five evaluation
queries produced zero exact rational hits and five exact misses. Each fixed-
ledger modular rank was 24 under all primes. The fifth miss exceeded the
registered four-row fallback allowance and stopped the run before the separate
held-out rank could reach 28.

Decision:

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
DO_NOT_SWEEP_LEDGER_OR_POPULATION_PARAMETERS
DO_NOT_CLAIM_EVALUATION_RANK_28
REQUIRE_A_MATERIALLY_NONLINEAR_OR_IMPLICIT_EXACT_QUERY_CODE_AT_E0
KEEP_NO_SURVIVING_CANDIDATE
KEEP_HARDWARE_SCALE_AND_E2_E7_CLOSED
```

The independent verifier performed zero model forwards and rebuilt core SHA
`1e79550fb66fe050338b2eedaf069728fd959583052dee2032fdd57f5cd0a7c4`.
This rejects the frozen automatic full-factor ledger, not every conceivable
post-hoc subspace or nonlinear exact data structure. Authority:
`results/exp_084a` and
`docs/research/EXPERIMENT_084A_CAUSAL_BILINEAR_RANK_GATE.md`.

## D-088 -- Reject trace-built query-adaptive linear code unions as core

A union of exact linear leaves is nonlinear at the router boundary and can be
strictly more expressive than one scanned span. However, routing supplies no
new cached scalar answer. For any globally independent finite query
population, total leaf dimension `A` permits at most `A` exact hits.

With all selector and lookup costs free, registered 405B construction
amortization permits at most 87,958 independent cached directions at leaf
dimension 1. This covers `0.43979%` of 20M independent queries while misses
require `99.999992826%` coverage; charged traffic is `85.0039x` target. The
complete 24-row EXP-084A build span also has zero membership hits among all
five frozen evaluation rows, certified by rank 24-to-25 increases under every
registered prime.

Decision:

```text
REJECT_TRACE_BUILT_QUERY_ADAPTIVE_LINEAR_CODE_UNION_AS_CORE
RETAIN_ONLY_AS_A_REPEATED_OR_CLUSTERED_QUERY_AUXILIARY
DO_NOT_GENERALIZE_TO_ALL_NONLINEAR_OR_IMPLICIT_DATA_STRUCTURES
OPEN_ONLY_A_CONCRETE_IMPLICIT_CHECKPOINT_DERIVED_SOURCE_AT_E0
KEEP_NO_SURVIVING_CANDIDATE
KEEP_MODEL_HARDWARE_AND_E2_E7_CLOSED
```

The generic independent population is a finite interface counterexample, not
a measurement that every causal Transformer stream is independent. No new
model forward was performed. Authority:
`docs/research/E0_QUERY_ADAPTIVE_CODE_UNION_BOUND.md` and
`results/e0_query_adaptive_code_union`.
