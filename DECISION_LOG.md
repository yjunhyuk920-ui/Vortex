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
