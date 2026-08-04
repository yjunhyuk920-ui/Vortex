# VORTEX Research State

Last updated: 2026-08-04 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- total peak GPU VRAM <=8 GiB;
- no target retraining, distillation, fine-tuning, LoRA, or target-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 and p95 <=1.5x on the same target machine;
- independently reproducible evidence.

The objective is unchanged. A small-checkpoint success, projected storage number, or theoretical mechanism is not target completion.

## Current environment truth

Available and measured:

```text
GitHub repository
GitHub Actions CPU
Python/PyTorch
pinned downloadable small checkpoints
synthetic/reference controls
frozen machine-readable evidence
```

Unavailable and `NOT TESTED`:

```text
405B model download/storage/execution
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

The measured pinned real-Q4 population has repeatedly behaved like general dense computation under exact classical structure tests:

```text
EXP-058 ordinary exact low rank: full rank
EXP-059 shift-displacement structure: full displacement rank
EXP-060 exact-zero weight sparsity: insufficient
EXP-061 exact-zero activation skipping: no observed zeros
EXP-062 non-mask attention zeros: negligible
EXP-063 exact cached K/KV equivalence: no duplicates
EXP-064 exact output-row reuse: population failure
EXP-065 Kronecker rearrangement rank: unfavorable full-rank cuts
EXP-066 exact TT/MPO bond-rank lower bound: storage Gate failure
EXP-067 exact Q/K/V common arithmetic: zero reusable rows
EXP-068 exact absolute-unread demand certificate: output-head-only p50 Gate failure
```

These failures do not prove the final objective impossible. They close the registered mechanisms as primary cores and prevent repetition without a genuinely new fact or theorem. Detailed permanent restrictions are in `FAILED_APPROACHES.md`; machine-readable authority is under `results/exp_NNN/`.

## EXP-066 authoritative closure

```text
153 two-dimensional tensors
144 dense projections
4,384 preregistered TT/MPO plans
operation p50/p90 3.8941% / 6.7788%
storage p50/p90 11.0524% / 22.9883%
```

The favorable storage lower bound failed the 10% p50 Gate. Unresolved ranks and implementation costs can only increase it.

Decision:

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

Authority: `results/exp_066/summary.json`.

## EXP-067 authoritative closure

```text
24/24 complete Q/K/V groups
10,752 Q4 rows
exact equality/sign/integer-proportional reusable rows: 0
operation p50/p90 100% / 100%
storage p50/p90 107.4142% / 114.1204%
common-right rank fraction 100%
```

Decision:

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Authority: `results/exp_067/summary.json`.

## EXP-068 authoritative closure

EXP-068 granted the candidate an impossible favorable oracle:

```text
all preceding Transformer work free
winning LM-head row free
all bound/order metadata free
independent optimal coordinate order per competitor
```

Integrity and coverage:

```text
3 pinned models
18 model/prompt cases
6 required families
153/153 source tensor hashes matched
reference replay mismatches 0
bound violations 0
control failures 0
```

Output-head-only necessary lower bound against the whole-model dense baseline:

```text
p50 weight and operation fraction 13.7696858262%  FAIL against 10%
p90 weight and operation fraction 19.2524013315%  PASS against 25%
minimum case 10.1376199755%
maximum case 21.0055007890%
```

Decision:

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

The registered norm/absolute-unread lazy-execution family is closed. Full-network propagation, scheduling, and kernels would add cost and are not authorized.

Authority:

```text
results/exp_068/summary.json
results/exp_068/raw/case_rows.jsonl
results/exp_068/evidence_manifest.json
workflow 30918865952
artifact 8896230736
artifact ZIP SHA-256 ff0f4398c0d162142d3e71d6864a3990704a14bf59e007182c9dce72c913835f
```

## Auxiliary infrastructure retained

- exact/checksummed evidence and mmap/pointer tooling;
- modular rank and witness certifiers;
- tensor/Kronecker/MPO rank auditors;
- exact row/group analyzers;
- exact verifier and fail-closed fallback components;
- output-head absolute-unread bound auditor;
- adversarial late-flip, triangular, and first-token controls.

Auxiliary classification does not mean the final runtime objective is achieved.

## Primary unresolved bottleneck

No tested mechanism has yet supplied a universal exact way to avoid almost all dense weight work:

- static exact structure is absent or too costly;
- exact sparsity and duplicates are absent;
- future-token proposals diverge too early;
- target-only iterative generation requires too many target passes;
- exact demand bounds need too much output-head work even with impossible free grants.

A new primary candidate must introduce a new exact information source rather than another representation of one matrix or another ordering of unread tiles.

## Current frontier

`EXP-069 — Causal Exact Temporal-Span Replay Gate`, preregistered in `NEXT_EXPERIMENT.md`.

It tests whether projection inputs from earlier exact decode tokens span later inputs, allowing `W x_t` to be reconstructed from cached exact `(x_k, W x_k)` pairs without rereading `W`.

The first Gate is an exact dyadic modular-rank lower bound:

- certify how often a new projection input adds an independent temporal direction;
- count every certified-independent arrival as a mandatory full dense pass;
- credit replay only after exact coefficient reconstruction and source-input verification;
- charge coefficient discovery, cached vectors, output combinations, and cache storage;
- stop before a replay kernel unless p50/p90 traffic and operation Gates pass.

405B execution and target hardware remain `NOT TESTED` regardless of the EXP-069 outcome.
