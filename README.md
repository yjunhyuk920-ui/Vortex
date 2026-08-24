# VORTEX

VORTEX researches an executor-only path for arbitrary public, unmodified dense Hugging Face 405B-class models on one 8-GiB GPU while preserving exact output and successor-state behavior and approaching same-machine native-4B-Q4 latency (`p50<=1.2x`, `p95<=1.5x`).

## Research rules

- no retraining, fine-tuning, distillation, LoRA or semantic target-weight change;
- `REAL_EXECUTOR_ONLY` is authoritative;
- all storage, traffic, arithmetic, KV/state, metadata, verification, repair, fallback, packing and synchronization are charged;
- research and validation are local; GitHub stores the validated commit and handoff;
- three ideas are only a minimum batch. If all are `NO_GO`, invert their common failure premise and generate another batch until `GO` or `CHEAP_KILL_ONLY`;
- README freshness is part of completion.

## Workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

GitHub Actions reexecution is not required unless explicitly requested.

## Latest authoritative result — EXP-107A

```text
REJECT_CURRENT_680_EXPLICIT_FMM_CATALOG_COMPOSITIONS_AS_10X_CORE
REJECT_REGISTERED_NAIVE_RANK46_APA_INTERPOLATION_EXACTIFICATION
```

The pinned 2026 catalog contains 680 small-format schemes, including 52 with normalized exponent below Strassen. Its best displayed exponent is `2.792481250`. Even granting that exponent to every registered 405B projection shape with unit constant and zero transform/ABI/byte cost leaves `12.515087006%` multiplication at `K=16384`. The first-core requirement is `10%`, corresponding to exponent `2.7700683089152998`.

Rank-46 `4x4` APA has zero-overhead headroom, but the registered generic 22-evaluation exact interpolation plan exceeds classical work. This rejects that exactification plan, not every border-rank/direct-sum extraction.

See:

- `docs/research/EXPERIMENT_107A_RESEARCH_PRIOR_AND_CURRENT_FMM_ENVELOPE_GATE.md`
- `docs/research/EXP_107A_LATEST_RESULT.md`
- `results/exp_107a/local/result.json`

## Active frontier — EXP-108A

```text
FINITE_STRASSEN_CALCULUS_DIRECT_SUM_EXTRACTION_CHEAP_KILL_ONLY
```

The next Gate must turn the 2026 asymptotic direct-sum speedup framework into one explicit finite exact target-size schedule. Every multiplication, addition, extraction/interpolation operation, coefficient word, transform and workspace byte is charged. No kernel is built unless the complete projection inventory crosses `10%` arithmetic.

The causal accepted-token source, `N/A`, exact verification and successor state remain separately mandatory even if arithmetic survives.

Read `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, `docs/research/RESEARCH_PRIORITIZATION_CONTRACT.md`, and `docs/research/VORTEX_RESEARCH_HANDOFF.md` before continuing.

## Claim boundary

Complete 405B execution, physical complete 8-GiB allocation, target CUDA/SASS, SSD/PCIe/HBM behavior and final same-machine p50/p95 remain `NOT_TESTED`.

```text
README_CURRENT=true
README_UPDATED=EXP-107A prior batches and arithmetic envelope; EXP-108A finite direct-sum frontier
```
