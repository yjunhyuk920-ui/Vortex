# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- total peak GPU VRAM <=8 GiB;
- no target retraining, distillation, fine-tuning, LoRA, or user-authored target-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same target machine;
- independently reproducible evidence.

The objective is unchanged.

## Current environment truth

MEASURED capability: GitHub repository, GitHub Actions CPU, Python, and downloadable small checkpoints.

Unavailable and `NOT TESTED`: target 8 GiB GPU, 405B storage/execution, CUDA, PCIe, target SSD, target power, TTFT, tokens/second, physical block weight reuse, combined target/draft residency, and target peak VRAM.

Phase D remains `NOT TESTED`. E6/E7 are not achieved.

## Evidence contract

- Phase A: theory and structure;
- Phase B: synthetic/reference;
- Phase C: small-real-model falsification or actual operation replacement;
- Phase D: target hardware.

Evidence E0–E7 and `MEASURED / DERIVED / PROJECTED / UNVERIFIED` labels are mandatory.

## Component classification

Auxiliary accepted:

- exact/checksummed mmap pointer VM;
- bounded exact compiler/DAG components in finite tested domains;
- CPTC certificate, metadata fault rejection, and exact fallback at E1;
- exact longest-prefix plus first-mismatch block verifier at E1;
- damped Picard/Anderson reference and fail-closed numerical machinery at E1;
- adversarial triangular and first-token constructions for universal-claim audits.

Rejected as core:

- raw prefix/future routing;
- static compression, deterministic residual, recurrent-program, and sparse-repair families;
- global/oracle-tight/stratified range CPTC;
- hard target-only Jacobi;
- sequential same-checkpoint partial-layer self-draft;
- target-only continuous Picard/Anderson generation;
- target-independent external drafting for the arbitrary-model exact mission;
- the tested TinyStories fixed external-draft pool even as a restricted practical core.

See `FAILED_APPROACHES.md`.

## EXP-049 closed evidence

Authority:

```text
results/exp_049/summary.json
workflow 30803672059
artifact 8851957250
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
```

MEASURED favorable target-only solver upper bound:

```text
p50 exact proposal prefix 4.5
maximum prefix 6
p90 target-equivalent fraction 168.778596%
Anderson/Jacobi exact-prefix improvement 0.25x
hidden triangular round barrier true
```

Decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

## EXP-050 authoritative evidence

Machine-readable authority:

```text
results/exp_050/summary.json
results/exp_050/raw/pair_rows.jsonl
results/exp_050/raw/case_rows.jsonl
results/exp_050/checksums.sha256
```

Frozen provenance:

```text
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact 8852817664
artifact size 34225 bytes
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
phase A/B/C-observation
evidence E1
```

Pinned unchanged models were TinyStories-1M/3M/8M with the common GPT-Neo tokenizer. Every model served as target and every other model as draft across six held-out families.

### MEASURED correctness and causal accounting

```text
EXP-050 accounting/counterexample tests: 9 passed
repository validation: passed
3 models
18 target/prompt cases
36 target/draft/prompt pairs
108 pair/K rows for K=64/128/256
excluded states 0
all-pair exact committed-output mismatches 0
all-pair target-future information uses 0
E3 exact future-target oracle failures 0
peak RSS 871824 KiB
```

Every 256-token draft continuation was generated causally with its own KV cache. Every cross-checkpoint proposal was verified by one exact target block pass; shorter K rows were causal prefixes of the same pass.

### MEASURED favorable fixed-pool upper bound

The exact target reference selected the best eligible draft and K per target/prompt. This selector is non-deployable and deliberately favorable.

```text
favorable-pool p50 exact proposal prefix 0.5
favorable-pool maximum exact proposal prefix 3
favorable-pool p90 4B/405B-normalized fraction 1.6320987654
                                                     = 163.20987654%
selected K distribution: K=64 for 18/18 cases
selected draft distribution: 1M 12 / 3M 4 / 8M 2
```

All 108 pair rows:

```text
matching prefix 0: 72 rows
matching prefix 1: 24 rows
matching prefix 2: 6 rows
matching prefix 3: 6 rows
```

Required-family coverage:

```text
English narrative: some acceptance
Code: some acceptance
Mathematics: some acceptance
Identifier boundary: some acceptance
Korean: zero matching proposal tokens in every selected target case
Structured JSON: zero matching proposal tokens in every selected target case
```

Target median prefixes:

```text
TinyStories-1M target: 1.0
TinyStories-3M target: 0.0
TinyStories-8M target: 0.5
```

### MEASURED universal first-token audit

A fixed draft proposed token 7. An arbitrary causal target chose token 8 for the same prompt.

```text
matching proposal prefix 0
exact correction committed 1 token
exact output match true
counterexample succeeds true
```

Therefore a fixed target-independent proposal cannot guarantee a nonzero exact prefix for every arbitrary target.

### PROJECTED traffic boundary

```text
405B Q4 target stream 188.592821 GiB
4B Q4 draft stream 1.862645 GiB
required total fraction 1.185185%
4B/405B draft ratio 0.98765432%
minimum completely correct proposal after draft cost 507 tokens
EXP-050 favorable p90 / required fraction 137.7083x
```

These are parameter-count projections, not target hardware measurements.

## EXP-050 scientific decision

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested fixed TinyStories pool also fails restricted practical continuation:

- prefix Gate: 0.5 <16;
- traffic Gate: 163.21% >10%;
- family coverage Gate: Korean and structured JSON fail;
- target-size trend Gate fails;
- universal first-token Gate fails;
- exactness and causality pass but do not rescue usefulness.

Required wording:

> EXP-050, E1: every cross-checkpoint proposal was generated causally and exact verification preserved target output, but even exact-reference selection over the fixed draft pool achieved only p50 0.5 matching tokens and maximum 3, with p90 1.6321 normalized target streams per committed token. Korean and structured JSON had zero useful acceptance, and an arbitrary target contradicted a fixed draft at the first token. Target-independent external drafting is rejected as a universal core and the tested pool is rejected as a restricted practical core. 405B and target hardware remain NOT TESTED.

## Primary unresolved bottleneck

No tested proposal source supplies long exact future tokens:

- target-only rounds are causally depth-limited;
- same-checkpoint partial layers diverge immediately;
- external independent models disagree at or near the first token;
- perfect future proposals prove verifier arithmetic only.

The next direct operation-skipping axis is **Transformer depth** rather than future-token proposal. With the exact target prefix fixed, intermediate layer states can be tested against the final target token without recursive draft drift.

## Current frontier

`EXP-051 — Oracle Layer-Finalization and Tail-Skip Gate`, defined in `NEXT_EXPERIMENT.md`.

For every exact greedy target state on pinned small checkpoints, EXP-051 will:

1. capture the current token hidden state after every Transformer block;
2. apply the target final normalization and LM head at every depth;
3. identify the earliest depth whose token equals the final target and remains equal at every later depth;
4. compute a favorable non-deployable tail-skip traffic upper bound including one full LM-head probe;
5. compare first-match versus suffix-stable depth;
6. test a late-decision adversarial residual chain where only the final layer flips the target token.

This differs from EXP-048 B3: it uses the exact target prefix at every token and audits current-token layer finalization, rather than recursively generating future tokens with partial layers.

If even suffix-stable oracle depth remains far above the 1.185% budget, layer skipping is rejected before selector/certificate engineering. If the oracle survives, the next Gate must derive a causal sound tail certificate and perform actual operation replacement.

## Reproduction

```bash
git checkout research/exp-050-external-draft-advice
python -m pytest -q tests/exp_050
python scripts/run_validation.py
bash experiments/exp_050/reproduce.sh
cd results/exp_050 && sha256sum -c checksums.sha256
```

## Next-session reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `ARCHITECTURE.md`
9. `REPRODUCIBILITY.md`
10. EXP-050 document and frozen summary
11. PR #60

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-052 authoritative result and EXP-053 frontier

Authority: `results/exp_052/summary.json`; workflow `30811429049`; source head `d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`.

1,152 exact warm states and 36 leave-one-family-out rows produced zero wrong hits and zero build/evaluation leakage, but P0 prefix and S0 KV-state held-out hit rates were 0% in every family. Fallback was 100%, natural exact reuse median/max was 1/1, and p90 fully-accounted target fraction was 6.0 (600%). Same-state replay was 100% exact and required at least 85 repetitions. Under 8 GiB hot index plus 1 TiB cold advice, combined coverage of 2^48 independent states was 6.357828752356909e-7, leaving fallback 0.9999993642171248.

Decision: `REJECT_ENUMERATIVE_EXACT_ADVICE_AS_CORE_RETAIN_FAIL_CLOSED_TABLE_AUXILIARY`. Exact tables are auxiliary only. The active frontier is `EXP-053 — Automatic Bit-Exact Decision-Circuit Compiler Gate`, which compiles bounded quantized operators from weights rather than enumerating states. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053 authoritative result and EXP-054 frontier

Authority: `results/exp_053/summary.json`; workflow `30814648709`; source head `325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge `4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`.

24 weight-derived circuits were exhaustively checked over 4,506,624 inputs with zero output-bit mismatch and no truth-table representation. Structural hashing left p50/p90 reachable fractions 0.84168345/0.94107229; dense-random p50 was 0.92452096. The maximum 405B source-parameter circuit projection was 255.5966 TiB. Late-bit controls simplified to zero AND nodes, but sparse controls still retained 65–78% of the exact bit-blast and projected 3.17–7.45 TiB. Growth and compile-amortization Gates passed; node, byte, storage, and random-dense Gates failed.

Decision: `REJECT_BIT_EXACT_DECISION_CIRCUIT_COMPILER_AS_CORE_RETAIN_AIG_REFERENCE_AUXILIARY`. The exact AIG compiler, evaluator, binary format, and exhaustive validator remain E1 auxiliary reference machinery. The active frontier is `EXP-054 — Exact Reduced Ordered Decision-Diagram Gate`, which replaces all-gate AIG evaluation with one exact input-adaptive decision path. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054 authoritative result and EXP-055 frontier

Authority: `results/exp_054/summary.json`; workflow `30816333096`; source head `2c63da85050afcedad6a00698a6f8fddd3bc99d2`; artifact `8856906303`; ZIP SHA-256 `0dc642f306cea99ce01095758a5f49151092d530efb94d36985553e408596edf`.

24 operators were compiled in natural and weight-magnitude orders: 48 completed diagrams, zero ceiling/fallback, zero mismatches across 9,013,248 validations, and zero truth-table representations. Selected global p50/p90 path fractions were 35%/95%. Dense-random growth was 1.6872587x per added input bit, maximum projected storage was 202.2479 TiB, and maximum order-search amortization was 1,185,055 queries. Late-bit controls reached 5–12.5% paths, but dense, low-rank, and sparse families failed the universal Gate.

Decision: `REJECT_EXACT_REDUCED_DECISION_DIAGRAM_AS_CORE_RETAIN_BDD_REFERENCE_AUXILIARY`. Exact reduced diagrams remain E1 auxiliary reference machinery. The active frontier is `EXP-055 — Exact Column-Signature Popcount Aggregation Gate`, a word-level compiler that groups identical or sign-related weight columns and computes group activation counts rather than evaluating bit-level gates or paths. Real Transformer operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, Phase D, E6, and E7 remain NOT TESTED.

<!-- EXP-055-AUTHORITATIVE-FINAL -->
## EXP-055 closed — Exact column-signature popcount aggregation

Authority: `results/exp_055/summary.json`; workflow `30820909775`; artifact `8858805996`; ZIP SHA-256 `983962faf329f2ccef2bd3f52c33116b146b0070fd350b1edee6c0f99923c6a8`.

MEASURED E1: 48 cases, 96 plans, 248,832 scalar validations, zero exact mismatches, zero runtime tables. Repeated/sign-related n=64 controls reached 7.8125%/9.375% logical work, proving a real exact compression fragment under strong repetition. The universal Gate failed: p50/p90 operations 62.5%/250%, p50/p90 bytes 63.64%/200%, dense/unique p50 250%, and 21 non-amortizing cases. Projected logical storage maximum 0.7597 TiB passed only the storage Gate.

Decision: `REJECT_EXACT_COLUMN_SIGNATURE_AGGREGATION_AS_CORE_RETAIN_GROUPING_REFERENCE_AUXILIARY`. Real Transformer extraction, operation replacement, 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/sec remain NOT TESTED.

Current frontier: EXP-056 exact prototype-plus-sparse-residual dictionary Gate.

<!-- EXP-056-AUTHORITATIVE-FINAL -->
## EXP-056 closed — Exact prototype plus sparse-residual dictionaries

Authority: `results/exp_056/summary.json`; workflow `30823042599`; artifact `8859665874`; ZIP SHA-256 `9fa7816c124069590aadf6746923b4ca1103800b333c110c30a74c3fb7b4c9e8`.

MEASURED E1: 56 cases, 448 plans, 1,161,216 scalar validations, zero exact mismatches, zero runtime tables. Repeated n=64 reached 7.8125%, exact sparse prototype perturbations 10.9375%, and sign clusters 15.625%. General p50/p90 work was 62.5%/131.25%; bytes 62.115%/169.643%; dense/unique p50 123.4375%; 24 cases did not amortize. Projected logical storage maximum 0.6791 TiB passed only storage.

Decision: `REJECT_EXACT_PROTOTYPE_RESIDUAL_DICTIONARY_AS_CORE_RETAIN_DICTIONARY_REFERENCE_AUXILIARY`. Current frontier is EXP-057 pinned real-checkpoint weight-structure extraction. 405B, 8 GiB, actual operation replacement, and target hardware remain NOT TESTED.

<!-- EXP-057-AUTHORITATIVE-FINAL -->
## EXP-057 closed — Pinned real-checkpoint exact weight structure

Authority: `results/exp_057/summary.json`; workflow `30824957941`; artifact `8860450501`; ZIP SHA-256 `7e2d91fb1af2d77c7cb87732557e8c42c22e23771264cfb000d29536d76172f0`.

MEASURED Phase C observation: 3 pinned unchanged models, 327 tensors, 153 two-dimensional tensors, 54,205,312 named 2-D scalars, and zero unregistered matrices. All 144 dense projections had zero exact repeated/sign-related columns in FP32, Q8, and Q4. Q4 p50/p90 operations were 82.8918%/85.8398%; bytes 329.0244%/490.6845%; median residual density 81.4087%; best matrix 70.2866%. Reconstruction and controls passed; projected storage was 0.9300 TiB.

Decision: `REJECT_REAL_WEIGHT_EXACT_GROUPING_DICTIONARY_AS_CORE_RETAIN_MEASURED_AUXILIARY_ONLY`. Q4 output preservation, actual operation replacement, 405B, 8 GiB, and hardware remain NOT TESTED. Current frontier: EXP-058 pinned real-Q4 exact algebraic-rank certificates.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## EXP-058 closed — Exact algebraic rank of pinned real Q4 matrices

Authority: workflow `30826618962`, artifact `8861905858`, ZIP SHA-256 `851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae`. All 144 registered dense projections were proven full integer/rational rank. Favorable exact two-factor operation/storage lower bounds were 2.0x at p50 and p90. Decision: `REJECT_REAL_Q4_EXACT_LOW_RANK_FACTORIZATION_AS_CORE_RETAIN_RANK_CERTIFICATES`. Q4 output preservation, constructive factor kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-059 exact shift-displacement rank.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## EXP-059 closed — Exact shift-displacement rank

Authority: workflow `30840432745`, artifact `8866573958`, ZIP SHA-256 `61d0c24ccacd310d7d0e7600cc926a882c74281827d524c4880c6715fad8800d`. All 144 registered real-Q4 dense projections had selected exact displacement-rank fraction 1.0. Favorable query and generator-storage lower bounds were p50/p90 1.0/1.0 and 2.0/2.0. Decision: `REJECT_REAL_Q4_EXACT_SHIFT_DISPLACEMENT_STRUCTURE_AS_CORE_RETAIN_CERTIFICATES`. Q4 output preservation, constructive generators, transform kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-060 exact zero-sparsity streaming.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## EXP-060 closed — Exact Q4 zero-sparsity streaming

Authority: workflow `30841671707`, artifact `8867145590`, ZIP SHA-256 `5e5255dbedd779b734876faa027cd2bf5e4a1b00ece7f28cbf35f428fb9a0b05`. The 144 real-Q4 dense projections had p50/p90 exact zero fractions 17.76%/20.37%. Exact sparse execution retained p50/p90 82.22%/85.06% operations and 150.93%/200.86% query bytes. Decision: `REJECT_REAL_Q4_EXACT_ZERO_SPARSITY_STREAMING_AS_CORE_RETAIN_SPARSE_AUXILIARY`. Q4 output preservation, physical sparse kernels, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-061 causal exact activation sparsity.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## EXP-061 closed — Causal exact activation sparsity

Authority: workflow `30843404056`, artifact `8867731496`, ZIP SHA-256 `a01d31b012badd7d06087df576279b852db07813a0c7fb50d65c3a7283e9ca65`. Hooked and unhooked generation matched for all 1,152 tokens. Exact zero count was 0 over 56,448 projection calls; warm-decode p50/p90 fully accounted work was 100.002%/100.391% and bytes 100.004%/101.566%. Decision: `REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY`. Physical sparse kernels, 405B activation statistics, 405B, 8 GiB, and target hardware remain NOT TESTED. Current frontier: EXP-062 exact non-mask attention-probability sparsity.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## EXP-062 closed — Exact non-mask attention probability sparsity

Authority: workflow `30844873182`, artifact `8868287407`, ZIP SHA-256 `497816dcca7e6b8c40e9222ed8511efa266fe2358aab847a93795d7c04637390`. Warm decode had 2,564 exact eligible zeros among 8,404,224 probabilities. Whole-model p50/p90 work was 100.048%/100.154% and bytes 100.093%/100.303%. Decision: `REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY`. Physical kernels, 405B statistics, 405B, 8 GiB and target hardware remain NOT TESTED. Current frontier: EXP-063 exact cached KV equivalence.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## EXP-063 closed — Exact cached-KV equivalence

Authority: workflow `30846082964`, artifact `8868770832`, ZIP SHA-256 `b900a7019d8527d6f67d0eb412bb2fb7a0331188d84cd74444ca10762a105a14`. Exact K/KV duplicate counts were zero across 147,456 group rows. Warm p50/p90 work was 100.021%/100.027%; bytes 106.263%/119.401%. Decision: `REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY`. Current frontier: EXP-064 pinned real-Q4 output-row equivalence and sparse-delta structure. 405B, 8 GiB, physical kernels and target hardware remain NOT TESTED.

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## EXP-064 closed — Real-Q4 exact output-row structure

Authority: workflow `30869720552`, artifact `8877450455`, ZIP SHA-256 `99c634bd4fb3903d32a1ed45fada7853ea4e1d199b375c129d1d4b8da4f39cb8`. 153 tensors, 144 dense projections, 1,683 plans, zero checksum/reconstruction/control mismatch. Identical/sign-related dense matrices: 0/0. Selected: dense 140, sparse-delta 4. p50/p90 operations and bytes: 100%/100%. Decision: `REJECT_REAL_Q4_OUTPUT_ROW_PROTOTYPE_AS_CORE_RETAIN_ROW_DICTIONARY_AUXILIARY`. Current frontier: EXP-065 exact Kronecker-rearrangement rank. 405B, 8 GiB, Q4 model-output preservation and hardware remain NOT TESTED.
