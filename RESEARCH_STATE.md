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
