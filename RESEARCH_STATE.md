# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored target-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same target machine;
- independently reproducible evidence.

The objective is unchanged.

## Current environment truth

MEASURED capability: GitHub repository, GitHub Actions CPU, Python, and downloadable small checkpoints.

Unavailable and `NOT TESTED`: target 8 GiB GPU, 405B storage/execution, CUDA, PCIe, target SSD, target power, TTFT, tokens/second, physical block weight reuse, and target peak VRAM.

Phase D remains `NOT TESTED`. E6/E7 are not achieved.

## Validation and evidence contract

- Phase A: theory and structure;
- Phase B: synthetic/reference;
- Phase C: small-real-model falsification or operation replacement;
- Phase D: actual target hardware.

Evidence levels E0–E7 and provenance labels `MEASURED / DERIVED / PROJECTED / UNVERIFIED` are mandatory.

## Component classification

Auxiliary accepted:

- exact/checksummed mmap pointer VM;
- bounded exact compiler/DAG components in their finite tested domains;
- CPTC causal certificate, metadata fault rejection, and exact fallback at E1;
- exact longest-prefix plus first-mismatch block verifier at E1;
- damped Picard/Anderson reference solvers and numerical fail-closed machinery at E1.

Rejected as core:

- raw prefix/future routing for unseen prompts;
- prior static compression, residual, recurrent-program, and sparse-repair families;
- global/oracle-tight/stratified range-based CPTC;
- hard target-only Jacobi under charged target-pass accounting;
- sequential same-checkpoint partial-layer self-draft with repeated LM-head work;
- target-only continuous Picard/Anderson block proposal generation under the declared arbitrary-model exact contract.

See `FAILED_APPROACHES.md` for permanent anti-repetition rules.

## EXP-048 closed evidence

Authoritative source:

```text
results/exp_048/summary.json
workflow 30798936320
artifact 8850040445
artifact ZIP SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
```

MEASURED:

```text
B1 perfect future oracle: 96 exact tokens / one target pass = 1.0416667%
B2 hard Jacobi p50: 58 target passes / 32 tokens = 181.25%
B3 partial-layer draft p50 committed tokens: 1
B3 p90 fully accounted fraction: 2893.843%
exact committed-output mismatches: 0
future information in deployable B3: 0
```

Decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

## EXP-049 authoritative evidence

Machine-readable authority:

```text
results/exp_049/summary.json
results/exp_049/raw/cases.jsonl
results/exp_049/raw/triangular_audit.json
results/exp_049/checksums.sha256
```

Frozen provenance:

```text
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact 8851957250
artifact size 105493 bytes
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
phase A/B/C-observation
evidence E1
```

Pinned small checkpoints were TinyStories-1M/3M/8M at the revisions recorded in the raw manifest. The corpus contains 3 models × 6 held-out families = 18 cases and 1,458 fixed solver trajectory rows. No context state was excluded.

### MEASURED correctness, causality, and numerical safety

```text
EXP-049 solver/lower-bound tests: 9 passed
repository validation: passed
selected exact-output mismatches: 0
selected deployable future-information uses: 0
unhandled numerical failures: 0
selected numerical fallbacks: 0
S3 exact future-state oracle alignment failures: 0
peak RSS: 684684 KiB
```

### MEASURED favorable upper-bound performance

The experiment allowed the exact reference to choose the best pre-registered S1/S2 variant, block size, and pass count per case. This selector is non-deployable and deliberately favors the hypothesis.

```text
oracle-best S1/S2 p50 matching prefix: 4.5 tokens
oracle-best S1/S2 maximum matching prefix: 6 tokens
oracle-best S1/S2 p90 target-equivalent fraction: 1.68778596
                                                   = 168.778596%
model median prefixes: 1M 4.5 / 3M 5.0 / 8M 4.0
```

All 18 favorable selections used four target solver passes and a 64-token block. Seventeen selected the hard top-1 Picard control; one selected top-k-8 Picard. No Anderson variant was selected.

Controls after four passes:

```text
S0 hard Jacobi p50 matching prefix: 4
S2 Anderson p50 matching prefix: 1
S2 / S0 improvement: 0.25x
```

### MEASURED triangular lower-bound audit

Two hidden-predecessor causal chains produced:

```text
Picard exact prefixes by round:   1, 2, 3, 4
Anderson exact prefixes by round: 1, 2, 3, 3
unresolved hidden suffix transcript indistinguishability: true
one-new-exact-position-per-round barrier observed: true
```

Under the declared black-box target-round interface, continuous arithmetic and Anderson history extrapolate observed outputs but do not reveal an adversarial hidden suffix before its exact predecessor is resolved.

### PROJECTED target comparison

```text
405B Q4 full stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent fraction: 1.185185%
EXP-049 favorable p90 fraction: 168.778596%
EXP-049 favorable p90 / required: about 142.405x
```

These are logical same-bit projections, not target-hardware measurements.

## EXP-049 scientific decision

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

Reasons:

- prefix Gate failed: 4.5 <16;
- traffic Gate failed: 168.78% >10%;
- Anderson improvement Gate failed: 0.25x <4x;
- the hidden triangular family established the pre-registered worst-case round barrier;
- exactness, causality, numerical-safety, and model-size-trend checks passed, but they do not rescue performance or universality.

Required wording:

> EXP-049, E1: fixed Picard/Anderson block solvers and the exact verifier remained causal, numerically controlled, and exact on the committed small-checkpoint corpus. Even a reference-selected favorable variant achieved only p50 4.5 exact proposal tokens with p90 1.6878 target-equivalent streams per committed token, while adversarial triangular models preserved the one-new-position-per-round barrier. Target-only fixed-point proposal generation is therefore rejected as the core runtime. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED.

## Primary unresolved bottleneck

The exact verifier is not the bottleneck. A successful exact block runtime needs future-token information that is not obtainable from a small number of target-only synchronous causal rounds for every arbitrary target.

The remaining information sources are:

1. an external target-independent proposal model;
2. automatically compiled target-specific transition advice/metadata;
3. a relaxed non-exact behavioral contract;
4. target hardware capable of retaining or rereading substantially more weights.

Only the first option preserves the current no-training/no-target-modification rule without immediately requiring exponential target-state compilation or relaxing correctness. It must still survive a universal no-free-lunch audit and fully charged practical tests.

## Current frontier

`EXP-050 — Target-Independent External Draft Advice Gate`, defined in `NEXT_EXPERIMENT.md`.

EXP-050 tests whether a fixed pool of already published, unmodified draft checkpoints can provide long exact target proposals without using target future tokens or target-specific training. It includes:

- a formal first-token counterexample for universal target-independent advice;
- cross-checkpoint draft proposals among the pinned TinyStories models;
- a favorable oracle selection over the fixed draft pool;
- exact block verification and full draft/target-equivalent accounting;
- an 85-token minimum exact-prefix requirement before real draft overhead, plus the dynamic requirement after charging draft streams.

A negative favorable-pool result rejects fixed external drafting as a universal core mechanism. A positive result would remain below E2 until a causal fixed selector and complete generation loop are implemented.

## Reproduction

```bash
git checkout research/exp-049-anderson-continuous-fixed-point
python -m pytest -q tests/exp_049
python scripts/run_validation.py
bash experiments/exp_049/reproduce.sh
cd results/exp_049 && sha256sum -c checksums.sha256
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
10. EXP-049 document and frozen summary
11. PR #59
