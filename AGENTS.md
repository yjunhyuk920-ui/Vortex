# AGENTS.md — VORTEX mandatory session contract

This is the first file every AI or human session must read.

## Fixed mission

Build a universal runtime that executes an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the executor.

Flagship acceptance target:

- real 405B-class dense model;
- peak GPU VRAM `<= 8 GiB`;
- no retraining, distillation, fine-tuning, LoRA, semantic weight modification, or user-authored model-specific adapter;
- original-model output and required successor-state contract preserved;
- p50 warm time/token `<= 1.2x` a native 4B Q4 baseline on the same target machine;
- p95 `<= 1.5x` that baseline;
- reproducible evidence from pinned code, inputs, configs, results, and hashes.

The target may not be silently reduced.

The canonical compact answer is [`MISSION_AND_WORKING_PRINCIPLES.md`](MISSION_AND_WORKING_PRINCIPLES.md). When the user asks “우리의 목표와 작업 원칙은?”, answer from that file and current remote research state.

## Canonical answer contract

A correct answer includes:

1. arbitrary public unmodified dense 405B-class checkpoint;
2. executor replacement only;
3. single `8 GiB` GPU;
4. exact original output and required successor state;
5. same-machine native 4B Q4 targets `p50 <=1.2x`, `p95 <=1.5x`;
6. full accounting of fallback, storage, traffic, state, verification, repair, packing, metadata, and cache;
7. three materially different new principles and cheapest-kill-first;
8. local/sandbox research and validation first;
9. GitHub commit/push/read-back only after local validation;
10. mandatory README freshness.

## Current-environment truth

Local/sandbox work is authoritative for research and validation when the required experiment can actually be executed there. Missing hardware-dependent quantities stay `NOT TESTED`.

Unless actual target evidence exists, do not claim:

- complete 405B target execution;
- physical complete 8-GiB target allocation;
- target CUDA/SASS behavior;
- target SSD/PCIe/HBM/decompression throughput;
- actual 405B TTFT/tokens-per-second;
- same-machine native 4B Q4 p50/p95 acceptance.

Never infer target hardware measurements from unrelated hardware.

## Mandatory local-first, commit-only workflow

The default research loop is now:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

### `LOCAL_RESEARCH`

Use the local/sandbox environment for:

- derivations, cost equations, exact calculators;
- combinatorial/tensor/circuit/program search;
- prototype implementation and rapid repair;
- exhaustive/random/adversarial controls;
- focused unit/property/regression tests;
- available public-checkpoint experiments;
- raw evidence, logs, and checksums.

### `LOCAL_VALIDATION_PASS`

Before committing, validate every item the current environment can honestly validate:

- exactness and fail-closed behavior;
- causal information availability;
- measured candidate/commit counts and fallback;
- implemented arithmetic, traffic, cache, metadata, repair, and workspace cost;
- target-scale equations and 8-GiB accounting;
- deterministic regeneration where applicable;
- focused/full test suite as feasible.

Anything unavailable locally remains `NOT TESTED`; do not replace it with an idealized grant.

### GitHub is persistence and handoff, not a second laboratory

After local validation passes, GitHub is used to:

- commit/push source, configs, results, logs, checksums, and docs;
- preserve negative evidence and provenance;
- update README and canonical ledgers;
- let later sessions resume from an exact SHA;
- read back the remote branch head to verify persistence.

**Do not rerun the same validated research on GitHub Actions.** A separate hosted reproduction, public-checkpoint rerun, Actions approval step, or artifact reproduction is not required for normal round completion.

Default policy:

```text
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
```

Use GitHub Actions only when the user explicitly asks for it. Existing automatic CI may run, but its conclusion is not a mandatory scientific approval after `LOCAL_VALIDATION_PASS`.

## Mandatory startup order

Read before proposing or editing:

1. `AGENTS.md`
2. `MISSION_AND_WORKING_PRINCIPLES.md`
3. `README.md`
4. `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md`
5. `docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`
6. `docs/research/VORTEX_RESEARCH_HANDOFF.md`
7. `RESEARCH_STATE.md`
8. `FAILED_APPROACHES.md`
9. `FAILED_APPROACHES_RECENT.md`
10. `DECISION_LOG.md`
11. `ASSUMPTION_REGISTER.md`
12. `VALIDATION_MATRIX.md`
13. `NEXT_EXPERIMENT.md`
14. `ARCHITECTURE.md`
15. `HARDWARE_VALIDATION_PLAN.md`
16. `REPRODUCIBILITY.md`
17. `docs/PROOF_FIRST_CONTRACT.md`
18. `docs/RESEARCH_EFFICIENCY_CONTRACT.md`
19. active experiment source/config/result/checksum files and PR discussion.

Then verify repository, branch, remote head SHA, latest authoritative result, and README freshness. Conversation memory is not authoritative.

## Validation phases

### Phase A — theory and structure

Permitted claims: mathematical validity, correctness/failure conditions, causal logic, lower bounds, resource equations, counterexamples.

### Phase B — synthetic/reference

Require independent reference code, randomized/property tests, boundary cases, fault injection, deterministic replay, and scaling trends. Synthetic success is not LLM success.

### Phase C — small real-model falsification

Use unmodified public checkpoints where available. Measure held-out prompts, future-information use, forward/layer calls, token/logit/state agreement, fallback, time, RAM, and size trend.

Purpose: falsify the principle early on a real Transformer checkpoint.

### Phase D — target hardware

Requires the actual target 8-GiB GPU, target storage, target checkpoint, same-machine baseline, and hardware profilers. Only this phase can validate final hardware acceptance.

## Evidence levels and provenance

- E0: idea/equation;
- E1: synthetic/reference validation;
- E2: real small-model operation replacement;
- E3: held-out generalization with measured causal coverage;
- E4: measured improvement on accessible representative hardware;
- E5: medium/large-model scaling validation;
- E6: target model runs under 8 GiB;
- E7: 405B meets the declared 4B-class target.

Separate metrics into `MEASURED`, `DERIVED`, `PROJECTED`, `UNVERIFIED`. Never present projected or unavailable values as measured.

## Reality-first authoritative execution

Every new core Gate has one authoritative arm:

```text
REAL_EXECUTOR_ONLY
```

Future target tokens/hidden states, perfect selectors, free `N/A`, free transforms, free metadata/workspace, free repair/fallback, unmeasured compression, and peak throughput presented as sustained throughput cannot satisfy promotion.

The authoritative arm charges the finite-word causal path that actually exists: candidate generation, target positions, verification, mismatch repair, rollback, fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache, workspace, fragmentation, and measured wall time where available.

Synthetic/oracle calculations may be debugging diagnostics only.

## Core resource and candidate Gate

At minimum:

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right).
\]

A credible core route must jointly improve:

\[
A\gg1,\qquad N/A\rightarrow1,\qquad r\ll1.
\]

Before implementation, a core candidate must:

- identify the original operation/weight movement eliminated;
- show a credible route to at least 10× reduction;
- define a causal information source without future leakage;
- charge selector, metadata, intermediate, verification, correction, fallback, RAM, SSD, PCIe, HBM, VRAM;
- explain scaling behavior;
- define the cheapest decisive falsification;
- differ materially from closed families.

Accepted length alone receives no core credit when `r=1`.

## Creative-research mandate

Before choosing a core experiment, invent three materially different execution principles. Each must reverse at least one hidden premise, computation order, information flow, or verification unit. Compare them by:

- exactness equation;
- fully charged realistic resource equation;
- explicit `>=10x` route;
- strongest counterexample;
- cheapest decisive Gate.

Do not use impossible favorable grants to promote a candidate. Implement only the strongest realistic survivor.

Repeated negative evidence closes a family. Reopening requires a new information source, asymptotic mechanism, execution dependency, or measured fact—not a parameter sweep or rename.

## Mandatory proof-first loop

1. Verify remote repository, branch/head SHA, active PR, latest committed evidence.
2. Read mission, failures, decisions, assumptions, efficiency contract, and handoff.
3. Generate three materially different principles.
4. Freeze success/rejection thresholds and target-scale ceiling.
5. Derive correctness, state, memory, traffic, compute, fallback, and scaling equations.
6. Run the cheapest local falsification first.
7. Stop immediately on a decisive negative result.
8. For a survivor, implement reference/prototype and tests locally.
9. Run the required local public-checkpoint/real-model validation if available.
10. Save raw evidence, processed result, logs, checksums locally.
11. Update canonical ledgers and README.
12. Commit/push the already validated state to a research branch.
13. Read back the remote SHA.
14. Report `REMOTE_COMMIT_VERIFIED` and all `NOT TESTED` boundaries.

There is **no mandatory GitHub Actions/reproduction step between 12 and 13**.

## Safety and exactness

- Future target tokens/states are forbidden for authoritative execution.
- Failed/unavailable certificate must exact-fallback or abort, never silently approximate.
- Probabilistic certification declares and union-accounts its error budget.
- Different floating reduction order is not bitwise equality until the ABI proves it.
- Missing files/checkpoints/hardware are infrastructure facts, not scientific evidence.
- Failed hypotheses are permanent project data.
- Remote/hidden compute may not be used to claim single-machine performance.

## Required repository state

Maintain as applicable:

```text
README.md
MISSION_AND_WORKING_PRINCIPLES.md
RESEARCH_STATE.md
NEXT_EXPERIMENT.md
DECISION_LOG.md
FAILED_APPROACHES.md
FAILED_APPROACHES_RECENT.md
ARCHITECTURE.md
ASSUMPTION_REGISTER.md
VALIDATION_MATRIX.md
HARDWARE_VALIDATION_PLAN.md
REPRODUCIBILITY.md
docs/research/VORTEX_RESEARCH_HANDOFF.md
```

Update every file whose truth changed; do not mechanically touch every file.

## README freshness Gate

Review `README.md` every meaningful round. Update it in the same commit when mission, governance, active frontier, latest authoritative result, quick start, dependencies, repository map, or advertised capabilities change.

Final report must contain either:

```text
README_CURRENT=true
README_UPDATED=<sections>
```

or:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

## Repository completion Gate

Every repository-changing round must satisfy:

```text
HAS_MEANINGFUL_CHANGE=true
LOCAL_VALIDATION_RECORDED=true
HAS_COMMIT=true
REMOTE_CONTAINS_COMMIT=true
RESEARCH_STATE_CURRENT=true
NEXT_GATE_CURRENT=true
README_CURRENT=true
PROVENANCE_TRUTHFUL=true
```

A local-only result is not a completed repository handoff. Once the locally validated result is committed/pushed and the remote SHA is read back, the round is complete; a hosted rerun is not required.

## Status vocabulary

Use as applicable:

```text
LOCAL_RESEARCH
LOCAL_VALIDATION_PASS
LOCAL_VALIDATION_FAILED
COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

Use `WORKFLOW_RUNNING`/`WORKFLOW_FAILED` only in the exceptional case where the user explicitly requests GitHub Actions.

## Active frontier

Do not hard-code a long-lived frontier here. Read the latest remote branch, committed raw result/checksums, `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, and `docs/research/VORTEX_RESEARCH_HANDOFF.md`.
