# AGENTS.md — VORTEX mandatory session contract

Version: CTC-2026-09-05. Read this first in every session.

## Fixed mission

Build a universal executor for arbitrary publicly released, unmodified Hugging Face dense Transformers. The flagship target is a real 405B-class checkpoint, batch size one, a single GPU with total peak VRAM <=8 GiB, original output/RNG and required successor state preserved, and warm time/token p50 <=1.2x and p95 <=1.5x native 4B Q4 on the same machine. Preserve the existing TTFT requirement and freeze any still-unspecified evaluation details before acceptance.

No retraining, fine-tuning, distillation, LoRA, semantic weight modification, manual model-specific adapter, hidden remote compute, free precomputation or uncharged fallback. Do not silently narrow the model/input domain. All CPU/RAM/SSD/PCIe/HBM/GPU/KV/cache/metadata/workspace/verification/repair costs count.

## Governing objective

The primary deliverable is a complete constructive execution theory, not a collection of small proofs, rejections, tests or commits. Read the [canonical mission](MISSION_AND_WORKING_PRINCIPLES.md) and [Constructive Theory Contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md). The latter controls theory acceptance, architecture neutrality, research priorities and scientific completion when older directives conflict. Recorded scientific evidence and fixed constraints are not overwritten.

Every core round begins with the final theorem and O1-O6 obligation ledger, compares three materially different premise-reversing principles, and concentrates on the strongest credible >=10x route. Do not start from mere combinations/minor variations or rename rejected families. Use the cheapest decisive test, then construct the missing causal/native/state/cost procedure. A rejected candidate stops; the mission does not automatically end with it.

No undefined perfect selector, free certificate, unproved compression/reuse or unlimited search. Lower bounds and optimistic rooflines reject candidates; only a complete algorithm with sufficient upper bounds can establish theory acceptance. Block-verification variables are not universal architecture requirements.

## Startup

Verify the actual remote repository, branch/head SHA, active PR and latest committed evidence. Then read:

1. This file, `MISSION_AND_WORKING_PRINCIPLES.md`, `docs/CONSTRUCTIVE_THEORY_CONTRACT.md`, and `README.md`.
2. `docs/PROOF_FIRST_CONTRACT.md`, `docs/RESEARCH_EFFICIENCY_CONTRACT.md`, `docs/WORK_SESSION_PROTOCOL.md`, `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md` and `CREATIVE_RESEARCH_MANDATE.md`.
3. Current `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, `VALIDATION_MATRIX.md`, `FAILED_APPROACHES.md`, `FAILED_APPROACHES_RECENT.md`, `DECISION_LOG.md`, `ASSUMPTION_REGISTER.md`, `ARCHITECTURE.md`, `HARDWARE_VALIDATION_PLAN.md`, and `REPRODUCIBILITY.md`.
4. Active source/config/raw result/log/checksum and PR discussions; branch-specific reality-first/handoff documents where present.

An old 'next experiment' or historical policy section does not supersede the current contract. Resolve remote branch differences; never assume main contains every draft experiment. Conversation memory is not evidence.

## Local-first validation and repository handoff

```text
LOCAL_RESEARCH -> LOCAL_VALIDATION_PASS -> COMMIT_PUSHED -> REMOTE_COMMIT_VERIFIED
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
```

Freeze hypotheses, input hashes and thresholds locally before examining results, and persist the preregistration with the evidence. Do not require a remote commit before the local experiment or duplicate research on Actions. GitHub is persistence/handoff after local validation; explicitly user-requested Actions is the exception. Do not bypass branch protection.

Use available local theory/reference/property/adversarial/checkpoint tests, deterministic regeneration and checksums as applicable. For documentation-only changes, validate Markdown, local links, unchanged constraints, precedence and cross-file consistency; do not claim runtime tests were run. Missing hardware or packages are infrastructure facts, not scientific rejection. Target execution, complete VRAM, CUDA/PCIe/SSD behavior, TTFT and baseline latency stay NOT TESTED until actually measured.

Before reporting repository work: preserve positive/negative evidence, update every affected ledger and README, commit to a descriptive research branch, push, and read back the remote SHA and changed files. Never force-push, directly push main, or merge unrelated experimental work. A local bundle alone is not a completed handoff. Use the connected GitHub writer even when shell Git is unavailable.

## Exactness and evidence

Empirical promotion uses `REAL_EXECUTOR_ONLY`: no future target token/state, oracle selector, free runtime work, speculative compression or peak-as-sustained throughput. A model theorem uses an explicit executable specification, not an oracle; it is not measured runtime evidence.

Prove original finite-word reduction/rounding semantics, RNG consumption and an inductive correspondence for every required successor state. A compressed state needs a proved exact continuation relation; token agreement on samples is insufficient. If the original interface exposes logits, preserve that interface too. Probabilistic certificates must disclose and union-account error and cannot silently replace deterministic exactness. Failed certificates exact-fallback or abort, never silently approximate; safety behavior is not mission success.

Keep Phase A theory, B synthetic/reference, C small-real falsification and D target hardware distinct. Keep E0 idea, E1 reference, E2 real operation replacement, E3 held-out causal generalization, E4 representative hardware, E5 scaling, E6 target <=8 GiB and E7 final target distinct. Separate MEASURED, DERIVED, PROJECTED, UNVERIFIED.

Report three independent axes:

```text
THEORY_STATUS=NOT_ESTABLISHED|CONDITIONAL|VERIFIED_IN_MODEL
HARDWARE_STATUS=NOT_TESTED|PARTIAL|E7_VERIFIED
HANDOFF_STATUS=IN_PROGRESS|REMOTE_COMMIT_VERIFIED|BLOCKED_REMOTE_WRITE
```

A commit completes persistence, not the theory. A verified model theorem does not complete E7. A session boundary does not authorize false success or an asynchronous promise.

## Durable state and reporting

Update files whose truth changes, not all files mechanically. Review README every meaningful round and state `README_CURRENT=true` with changed sections or a concrete unchanged reason. Preserve prior snapshots when consolidating policy/history.

Report the execution method, proved obligations, paid budget and remaining decisive gap before test counts/commit logistics. A useful rejection is not evidence of greater feasibility. The active frontier is in the current remote state and next-obligation ledger, not a hard-coded old experiment in this file.
