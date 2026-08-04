# VORTEX Research State

Last updated: 2026-08-05 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- total peak GPU VRAM <=8 GiB;
- no target retraining, distillation, fine-tuning, LoRA, or target-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same target machine;
- independently reproducible evidence.

The objective is unchanged. A small-checkpoint result, theoretical mechanism, or projected number is not target completion.

## Current environment truth

Measured and available:

```text
GitHub repository and GitHub Actions CPU
Python/PyTorch
pinned downloadable TinyStories checkpoints
synthetic/reference controls
frozen machine-readable evidence
```

Unavailable and `NOT TESTED`:

```text
405B download/storage/execution
target 8 GiB GPU
CUDA and physical kernels
PCIe and target SSD
TTFT and tokens/second
target peak VRAM and power
```

Phase D remains `NOT TESTED`. E6/E7 are not achieved.

## Evidence contract

- Phase A: theory and mechanism;
- Phase B: synthetic/reference controls;
- Phase C: small-real-model falsification or actual operation replacement;
- Phase D: target hardware.

Every result must distinguish `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED`. Estimates may not be presented as measurements.

## Current scientific position

The pinned populations have repeatedly behaved like general dense computation under the registered exact tests:

```text
EXP-058 ordinary exact low rank: full rank
EXP-059 shift-displacement structure: full displacement rank
EXP-060 exact-zero Q4 sparsity: insufficient
EXP-061 exact-zero activation skipping: no observed zeros
EXP-062 non-mask attention zeros: negligible
EXP-063 exact cached K/KV equivalence: no duplicates
EXP-064 exact output-row reuse: population failure
EXP-065 Kronecker rearrangement rank: unfavorable full-rank cuts
EXP-066 exact TT/MPO: storage Gate failure
EXP-067 joint Q/K/V arithmetic: zero reusable rows
EXP-068 absolute-unread demand certificate: output-head-only p50 Gate failure
EXP-069 temporal exact span replay: mandatory p50/p90 full passes 100%
EXP-070 Q4 local-pattern tables: operation remained about 88-91%; bytes/storage exceeded dense
```

These failures do not prove the final objective impossible. They close only the registered mechanisms and prohibit cosmetic variants without a new fact, theorem, or information source. Permanent restrictions are in `FAILED_APPROACHES.md` and `FAILED_APPROACHES_RECENT.md`; machine-readable authority is under `results/exp_NNN/`.

## Recent authoritative closures

### EXP-066 — TT/MPO bond-rank core

```text
144 dense projections
4,384 preregistered plans
operation p50/p90 3.8941% / 6.7788%
storage p50/p90 11.0524% / 22.9883%
```

The favorable storage lower bound failed the 10% p50 Gate before unresolved ranks and implementation costs.

Decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

Authority: `results/exp_066/summary.json`.

### EXP-067 — Joint Q/K/V exact arithmetic

```text
24 complete Q/K/V groups
10,752 Q4 rows
exact reusable rows 0
operation p50/p90 100% / 100%
storage p50/p90 107.4142% / 114.1204%
```

Decision:

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Authority: `results/exp_067/summary.json`.

### EXP-068 — Absolute-unread demand certificates

After granting all preceding Transformer work, the winning output-head row, metadata, and a separate optimal reveal order per competitor for free:

```text
output-head-only p50 weight/operation lower bound 13.7697%
p90 lower bound 19.2524%
```

Decision:

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

Authority: `results/exp_068/summary.json`.

### EXP-069 — Causal exact temporal-span replay

```text
147 registered projections
833 warm projection traces
p50/p90 mandatory weight and operation fractions 100% / 100%
TinyStories-1M/3M/8M model p50 69.244% / 100% / 100%
verified exact replay hits 0
p50 basis cache / Q4 projection population 391.97%
```

Decision:

```text
REJECT_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_AS_CORE
RETAIN_DYADIC_RANK_AUDITOR_AUXILIARY
```

Authority: `results/exp_069/summary.json`.

### EXP-070 — Exact Q4 local-pattern table circuits

Every frozen dense projection was evaluated under the preregistered block widths and deterministic column orders. One joint plan per matrix minimized the maximum of operation, query-byte, and static-representation fractions; no cost-axis cherry-picking was allowed.

```text
3 models
144 dense projections
3,024 plans
checksum/reconstruction/collision/control failures 0
operation p50/p90 88.4856% / 91.4423%
query-byte p50/p90 111.0294% / 112.7907%
static representation p50/p90 111.0294% / 112.7907%
minimum joint worst-axis fraction 105.4244%
```

Decision:

```text
REJECT_EXACT_Q4_LOCAL_PATTERN_TABLE_AS_CORE
RETAIN_BLOCK_PATTERN_ANALYZER_AUXILIARY
```

Authority:

```text
results/exp_070/summary.json
results/exp_070/raw/plan_rows.jsonl
results/exp_070/raw/selected_rows.jsonl
workflow 30930542616
artifact 8901017649
artifact ZIP SHA-256 0e3e60f959af852759b9aac8dd6af1a28524cdcbb6c736cd8e32ad00d6c29987
```

## Auxiliary infrastructure retained

- checksum and provenance tooling;
- exact modular-rank and witness certifiers;
- Kronecker/MPO and dyadic temporal-rank auditors;
- exact row/group/block-pattern analyzers;
- exact verifier and fail-closed fallback components;
- output-head absolute-unread bound auditor;
- adversarial random, forced-unique, triangular, recurrence, and late-flip controls.

Auxiliary classification does not mean the final runtime objective is achieved.

## Primary unresolved bottleneck

No tested mechanism has supplied a universal exact way to avoid almost all dense weight information and arithmetic. Static exact structure, sparsity, duplication, current-input certification, temporal replay, and short Q4 table circuits all failed by large margins or storage limits.

Continuing to enumerate classical decompositions is no longer efficient. Before proposing another executor mechanism, the project must determine whether known online matrix-vector data-structure lower bounds can rigorously constrain the registered conventional execution model.

## Current frontier

`EXP-071 — Universal Exact Dense Runtime Lower-Bound Applicability Audit`, preregistered in `NEXT_EXPERIMENT.md`.

It will:

- verify primary theorem statements and every hypothesis;
- formalize preprocessing, cold representation, <=8 GiB side information, online query, exactness, word/cell size, and randomization assumptions;
- verify a small exact reduction from Boolean matrix-vector multiplication into dense float projections;
- audit square/rectangular padding and model-wide direct-sum composition rather than assume them;
- apply only certified formulas to the registered 405B tensor plan;
- state either a rigorously bounded conventional model or `INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY`.

A theorem audit is not a hardware measurement. Actual 405B execution and target hardware remain `NOT TESTED` regardless of the EXP-071 outcome.
