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

The objective is unchanged. A small-checkpoint result, theorem, or projection is not target completion.

## Current environment truth

Available and measured:

```text
GitHub repository and GitHub Actions CPU
Python/PyTorch
pinned downloadable small checkpoints
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

Every result must distinguish `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED`. Estimates and asymptotic theorems may not be presented as measurements.

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
EXP-070 Q4 local-pattern tables: operation 88-91%; bytes/storage exceeded dense
EXP-071 known online-Mv lower bounds: insufficient for an 8 GiB model-wide impossibility claim
```

These results do not prove the final objective feasible or impossible. They close only the registered mechanisms or claims. Permanent restrictions are in `FAILED_APPROACHES.md` and `FAILED_APPROACHES_RECENT.md`; machine-readable authority is under `results/exp_NNN/`.

## Recent authoritative closures

### EXP-066 — TT/MPO bond-rank core

```text
144 dense projections
4,384 preregistered plans
operation p50/p90 3.8941% / 6.7788%
storage p50/p90 11.0524% / 22.9883%
```

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

After granting all preceding Transformer work, the winning output-head row, metadata, and an independently optimal reveal order for each competitor for free:

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

```text
3 models
144 dense projections
3,024 plans
checksum/reconstruction/collision/control failures 0
operation p50/p90 88.4856% / 91.4423%
query-byte and static p50/p90 111.0294% / 112.7907%
minimum joint worst-axis fraction 105.4244%
```

Decision:

```text
REJECT_EXACT_Q4_LOCAL_PATTERN_TABLE_AS_CORE
RETAIN_BLOCK_PATTERN_ANALYZER_AUXILIARY
```

Authority: `results/exp_070/summary.json`.

### EXP-071 — Exact dense-runtime lower-bound applicability audit

EXP-071 directly audited CGL15 Theorem 3 and CKL18 Theorems 1.2/1.3 before allowing an impossibility claim.

Integrity:

```text
3 theorem statements registered
1,052,740 exhaustive binary reduction cases
4,164 direct float32 replay cases
reduction mismatches 0
control failures 0
9 Llama-405B tensor families
884 tensor instances
405,849,243,648 parameters
```

Applicability:

```text
largest valid square subproblem n          16,384
CKL18 maximum registered side state        8 MiB
VORTEX hot/side-state allowance            8 GiB
ratio                                      1,024x
covered tensor families                    0 / 9
required model-wide direct sum             not established
finite Omega constants                     unavailable
```

Decision:

```text
INSUFFICIENT_LOWER_BOUND_DO_NOT_CLAIM_IMPOSSIBILITY
```

This does not establish feasibility. It prohibits dividing the shared 8 GiB state by tensor count, summing per-matrix asymptotic bounds without a direct-sum theorem, or equating cell probes with hardware transactions.

Authority:

```text
results/exp_071/summary.json
results/exp_071/raw/theorem_hypotheses.jsonl
results/exp_071/raw/tensor_rows.jsonl
results/exp_071/processed/direct_sum_audit.json
workflow 30965323458
artifact 8914506737
artifact ZIP SHA-256 bc81e90e3b5a35935f893ad7396d4b41a13de46606ce14bccc53cf79e30e8ba4
```

## Auxiliary infrastructure retained

- checksum and provenance tooling;
- exact modular-rank and witness certifiers;
- Kronecker/MPO and dyadic temporal-rank auditors;
- exact row/group/block-pattern analyzers;
- theorem-hypothesis and exact-reduction auditor;
- exact verifier and fail-closed fallback components;
- adversarial random, forced-unique, structured, recurrence, and late-flip controls.

Auxiliary classification does not mean the final runtime objective is achieved.

## Primary unresolved bottleneck

No tested mechanism supplies a universal exact way to avoid almost all dense weight information and arithmetic. At the same time, the audited lower-bound papers do not rule out a jointly preprocessed model with 8 GiB of side state.

The remaining gap is therefore constructive: find an exact smaller executable program that is more general than row/column equality, fixed tensor decompositions, contiguous block tables, or temporal replay.

## Current frontier

`EXP-072 — Exact Nonlocal Q4 Shared Arithmetic-DAG Synthesis Gate`, preregistered in `NEXT_EXPERIMENT.md`.

It tests whether a deterministic compiler can synthesize exact integer straight-line programs that share non-contiguous linear forms across output rows and across projections receiving the same activation. The Gate will:

- synthesize bounded exact arithmetic DAGs on every registered real-Q4 tile and shared-input group;
- verify every output by full symbolic coefficient-vector reconstruction;
- charge runtime arithmetic, circuit bytes, operand IDs, constants, output maps, row scales, and cross-tile accumulation;
- use structured positive controls and dense-random/forced-unique negative controls;
- stop before CUDA or model-wide transcoding unless operation, query, and storage p50/p90 Gates all pass.

Floating-point evaluation-order preservation, physical kernels, 405B execution, and target hardware remain `NOT TESTED` regardless of the EXP-072 outcome.
