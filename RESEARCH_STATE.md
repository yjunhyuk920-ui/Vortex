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
