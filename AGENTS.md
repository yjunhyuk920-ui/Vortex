# AGENTS.md — VORTEX mandatory session contract

This is the first file every AI or human session must read.

## Fixed mission

Build a universal runtime that executes an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the executor.

Flagship acceptance target:

- real 405B-class dense model;
- peak GPU VRAM `<= 8 GiB`;
- no retraining, distillation, fine-tuning, LoRA, semantic weight modification, or user-authored model-specific adapter;
- original-model ability and declared output/successor-state contract preserved;
- p50 warm time/token `<= 1.2x` a native 4B Q4 baseline on the same target machine;
- p95 `<= 1.5x` that baseline;
- independent reproduction from pinned code and checkpoint hashes.

The target may not be silently reduced.

The compact canonical answer is in [`MISSION_AND_WORKING_PRINCIPLES.md`](MISSION_AND_WORKING_PRINCIPLES.md). When the user asks “우리의 목표와 작업 원칙은?”, answer from that file and the current remote research state before doing anything else.

## Canonical answer contract

A correct answer must include all of the following:

1. arbitrary public **unmodified dense 405B-class checkpoint**;
2. executor replacement only; no training or semantic weight modification;
3. single `8 GiB` GPU;
4. original output and required successor-state contract;
5. same-machine native 4B Q4 targets: `p50 <= 1.2x`, `p95 <= 1.5x`;
6. complete accounting of fallback, storage, traffic, state, verification, and repair;
7. cheapest-kill-first and three materially different new principles per core round;
8. sandbox-first computation and GitHub-final evidence;
9. remote commit/read-back requirement;
10. mandatory README freshness.

Do not answer only with a vague “run 405B on 8 GB.”

## Current-environment truth

The ordinary sandbox and GitHub Actions are useful for theory, exact arithmetic, search, prototypes, tests, small public checkpoints, deterministic calculators, and reproducibility. They do not by themselves provide target 405B/8-GiB measurements.

Unless actual target evidence exists, these remain `NOT TESTED`:

- target 8 GiB GPU allocation and profiling;
- complete 405B checkpoint execution;
- target CUDA/SASS behavior;
- target SSD, PCIe, HBM, and decompression throughput;
- real 405B TTFT and tokens/second;
- same-machine native 4B Q4 p50/p95 acceptance.

Never infer Phase-D measurements from GitHub CPU runners.

A missing package, DNS route, local `git`, `gh`, tunnel, GPU, or checkpoint is an infrastructure fact, not a session stop condition. Continue every independent task that remains possible and report only the device-dependent part as `NOT TESTED`.

## Mandatory sandbox-first workflow

The default research loop is:

```text
SANDBOX_RESEARCH
-> SANDBOX_GATE
-> SOURCE_COMMIT_PUSHED
-> WORKFLOW_RUNNING (only when hosted reproduction adds value)
-> RESULT_COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

### Sandbox first

Use the available sandbox before GitHub Actions for:

- derivations, cost equations, and exact calculators;
- combinatorial, tensor, circuit, and program search;
- prototype implementation and rapid repair;
- exhaustive/random/adversarial controls;
- focused unit/property tests;
- small locally available model experiments.

Do not turn every hypothesis iteration into a commit and hosted workflow. GitHub Actions is not the primary reasoning loop.

### GitHub last

Promote only:

- a candidate that survives the cheapest sandbox Gate;
- a decisive negative result that prevents repeated work;
- meaningful reusable research infrastructure.

Use GitHub for frozen source/config, independent clean reproduction, immutable evidence, checksums, PR discussion, and cross-session handoff. Sandbox success alone is not round completion.

While an external workflow is running, report `WORKFLOW_RUNNING` with repository, branch, source SHA, workflow ID, and remaining independent work. Do not remain silent and do not claim a result before it exists.

## Mandatory startup order

Read before proposing or editing:

1. `AGENTS.md`
2. `MISSION_AND_WORKING_PRINCIPLES.md`
3. `README.md`
4. `docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md`
5. `docs/research/VORTEX_RESEARCH_HANDOFF.md`
6. `docs/research/FIXED_PUBLIC_DYNAMIC_EXECUTOR_DIRECTIVE.md`
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
19. `docs/WORK_SESSION_PROTOCOL.md`
20. active experiment files, workflow, PR comments, logs, and result JSON.

Then verify repository, base, branch, head commit, PR state, workflow conclusion, raw evidence, and README freshness. Conversation memory is not authoritative.

## Validation phases

### Phase A — theory and structure

Permitted claims: mathematical validity, correctness conditions, failure conditions, causal logic, lower bounds, resource equations, and strongest counterexamples.

Required boundary:

> Structurally valid conditions were established. Large-model performance remains unverified.

### Phase B — synthetic/reference

Require independent reference code, randomized/property tests, boundary cases, fault injection, deterministic replay, and scaling trends. Synthetic success is not LLM success.

### Phase C — small real-model falsification

Use available unmodified small checkpoints. Measure held-out prompts, future-information use, forward/layer calls, token/logit/state agreement, fallback, CPU time, RAM, and size trend.

Purpose:

> Falsify the execution principle early on a real Transformer checkpoint.

Small-model evidence is never 405B performance evidence.

### Phase D — target hardware

Requires a real target 8 GiB GPU, target storage, target checkpoint, same-machine baseline, and hardware profilers. Only this phase can validate actual target VRAM, TTFT, tokens/second, PCIe, SSD, HBM, and final 405B acceptance.

## Evidence levels and provenance

Use exactly:

- E0: idea or equation;
- E1: synthetic/reference validation;
- E2: real small-model operation replacement;
- E3: held-out generalization with measured causal coverage;
- E4: measured improvement on accessible representative hardware;
- E5: medium/large-model scaling validation;
- E6: target model runs under 8 GiB VRAM;
- E7: 405B meets the declared 4B-class target.

Separate every metric into:

- `MEASURED`;
- `DERIVED`;
- `PROJECTED`;
- `UNVERIFIED`.

E0–E3 may not be described as E6/E7 feasibility or success. Never present `PROJECTED` or `UNVERIFIED` as `MEASURED`.

## Core resource and candidate Gate

At minimum, account through:

\[
T_{\rm token}
\ge
\max\left(
\frac{S_c}{B A},
\;
r\frac{N}{A}\frac{2P}{F}
\right).
\]

A credible core route must jointly improve:

\[
A\gg1,\qquad N/A\rightarrow1,\qquad r\ll1.
\]

Before implementation, a core candidate must:

- identify the original operation or weight movement eliminated;
- show a credible optimistic path to at least 10× reduction and toward the final target-equivalent fraction;
- define the causal selector/information source without future target leakage;
- charge selector, metadata, intermediate, verification, correction, fallback, RAM, SSD, PCIe, HBM, and VRAM;
- explain why the effect should survive or improve with scale;
- define the cheapest decisive falsification;
- differ materially from every closed family.

Accepted length alone receives no core credit when target dense arithmetic remains `r=1`.

## Creative-research mandate

Before choosing a core experiment, invent three materially different execution principles. Each must reverse at least one hidden premise, computation order, information flow, or verification unit. Compare them using:

- exactness equation;
- complete optimistic resource equation;
- explicit route to `>=10x` elimination or amortization;
- strongest counterexample;
- cheapest decisive Gate.

Implement only the strongest survivor. Existing-technique combinations may be auxiliary components, but they are not the required starting point.

Repeated negative evidence closes a mechanism family. Reopening requires a new information source, asymptotic mechanism, execution dependency, or measured fact that invalidates the rejection premise. Parameter sweeps, mode-order variants, rank changes, block changes, and renamed decompositions are insufficient.

Use cheapest-kill-first:

```text
resource/information bound
-> favorable oracle or exact certificate
-> sandbox reference and adversarial controls
-> pinned small-real-checkpoint falsification
-> minimal operation replacement
-> backend/kernel
-> target hardware
```

## Mandatory proof-first loop

1. Verify remote repository, active branch, head SHA, PR, workflow, and evidence.
2. Read mission, failures, decisions, assumptions, efficiency contract, and active handoff.
3. Generate three materially different principles.
4. Freeze exact success/rejection thresholds and target-scale ceiling.
5. Derive correctness, state, memory, traffic, compute, fallback, and scaling equations.
6. Run the cheapest sandbox falsification first.
7. Stop immediately when a decisive negative bound is established.
8. For a survivor, implement the minimum independent reference and focused tests in the sandbox.
9. Freeze source, config, claim boundary, and stop rule on a research branch.
10. Use hosted workflow only when it adds clean reproduction, a pinned checkpoint, a long run, or immutable artifacts.
11. Save raw evidence, processed result, logs, and checksums.
12. Update every canonical ledger whose truth changed.
13. Check and update `README.md` under the README freshness contract.
14. Commit, push, and read back the remote SHA.
15. Report the exact status vocabulary and every `NOT TESTED` boundary.

Do not increase experiment numbers without eliminating a real assumption or testing a new mechanism. Do not build an optimized implementation to reconfirm a decisive negative theorem, lower bound, or favorable-oracle ceiling.

## Safety and exactness

- Future target tokens or hidden states are forbidden unless the experiment is explicitly labeled a non-deployable oracle upper bound.
- A failed or unavailable certificate must exact-fallback or abort, never silently approximate.
- Probabilistic certification must declare and union-account its error budget.
- A different floating-point reduction order is not bitwise equality until the declared ABI proves it.
- Missing files, checkpoint download failures, and runner timeouts are infrastructure failures, not scientific evidence.
- Failed hypotheses are permanent project data.
- Remote or hidden compute may not be used to claim a single-machine result.

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

Experiment layout:

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/
results/exp_xxx/
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

Do not mechanically touch every file. Update every file whose truth changed.

## README freshness Gate

`README.md` is mandatory public/session orientation, not a historical decoration.

Update it in the same round when mission, governance, active frontier, latest authoritative result, quick start, dependencies, repository map, or advertised capabilities change. Remove stale experiment descriptions and unregenerated numeric test counts.

Before final reporting, set:

```text
README_CURRENT=true
README_UPDATED=<sections>
```

or:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

`README_CURRENT=false` means the round is not complete when remote writing is available.

## Repository completion and handoff Gate

Every repository-changing round must satisfy:

```text
HAS_MEANINGFUL_CHANGE=true
HAS_COMMIT=true
REMOTE_CONTAINS_COMMIT=true
RESEARCH_STATE_CURRENT=true
NEXT_GATE_CURRENT=true
VALIDATION_RECORDED=true
README_CURRENT=true
PROVENANCE_TRUTHFUL=true
```

Use the connected GitHub writer first when available. Local DNS or local `git` failure does not justify skipping connector operations.

A local-only tree, patch, or bundle is:

```text
LOCAL_ONLY
REMOTE_RESEARCH_NOT_RECORDED
ROUND_NOT_COMPLETE
```

Only a remote commit read back from its branch may be:

```text
REMOTE_COMMIT_VERIFIED
```

## Status vocabulary

Use as applicable:

```text
SANDBOX_RESEARCH
SANDBOX_GATE
SOURCE_COMMIT_PUSHED
WORKFLOW_RUNNING
WORKFLOW_FAILED
RESULT_COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

Final reports for repository work begin with repository, base SHA, branch, commit SHA(s), remote verification, PR, CI/check state, README status, and uncommitted remainder.

## Active frontier

Do not encode a long-lived experiment narrative here. It becomes stale.

Read, in this order:

1. the latest remote branch and open PR;
2. authoritative result/checksum files;
3. `RESEARCH_STATE.md`;
4. `NEXT_EXPERIMENT.md`;
5. `docs/research/VORTEX_RESEARCH_HANDOFF.md`.

`README.md` must summarize the current frontier, but the raw remote evidence and canonical ledgers remain authoritative.

<!-- REALITY-FIRST-EXECUTION:START -->
## Reality-first authoritative execution

Every new core Gate has one authoritative arm: `REAL_EXECUTOR_ONLY`.
Future target tokens or hidden states, perfect selectors, free `N/A`, free transforms,
free metadata/workspace, free repair/fallback, unmeasured compression, and peak
throughput presented as sustained throughput are forbidden from satisfying a
promotion threshold. Synthetic or target-seeing calculations may appear only as
non-authoritative debugging diagnostics.

The authoritative arm must execute a finite-word causal path and charge candidate
generation, every target position, verification, mismatch repair, rollback,
fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache,
workspace, fragmentation, and measured wall time. Missing quantities remain
`NOT TESTED`; they are never replaced by an ideal grant.

Normative detail: [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md).
<!-- REALITY-FIRST-EXECUTION:END -->
