# AGENTS.md — VORTEX mandatory session contract

This is the first file every AI or human session must read.

## Fixed mission

Build a universal runtime that executes an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the executor.

Flagship acceptance target:

- real 405B-class dense model;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or model-specific user-authored adapter;
- original-model ability and declared output contract preserved;
- p50 warm time/token <=1.2x a native 4B Q4 baseline on the same target machine;
- p95 <=1.5x that baseline;
- independent reproduction from pinned code and checkpoint hashes.

The target may not be silently reduced.

## Current-environment truth

The current primary environment is GitHub plus limited GitHub Actions CPU runners.

Currently unavailable:

- target 8 GiB GPU measurement;
- 405B checkpoint download/storage/execution;
- CUDA profiling;
- PCIe and target SSD profiling;
- real 405B TTFT or tokens/second;
- real target peak VRAM.

All Phase D claims are therefore **NOT TESTED** until actual hardware evidence exists. Never imitate or infer a Phase D measurement from GitHub Actions.

## Mandatory startup order

Read before proposing or editing:

1. `AGENTS.md`
2. `RESEARCH_STATE.md`
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `ARCHITECTURE.md`
9. `HARDWARE_VALIDATION_PLAN.md`
10. `REPRODUCIBILITY.md`
11. `docs/PROOF_FIRST_CONTRACT.md`
12. `docs/RESEARCH_EFFICIENCY_CONTRACT.md`
13. `docs/WORK_SESSION_PROTOCOL.md`
14. active experiment files, workflow, PR comments, logs, and result JSON.

Then verify branch, head commit, PR state, workflow conclusion, and authoritative raw evidence. Conversation memory is not authoritative.

## Validation phases

Every experiment declares one or more phases.

### Phase A — theory and structure

Permitted claims: mathematical validity, correctness conditions, failure conditions, causal logic, lower bounds, resource equations, and strongest counterexamples.

Required wording:

> Structurally valid conditions were established. Large-model performance remains unverified.

### Phase B — synthetic/reference

Required: independent reference implementation, randomized/property tests, boundary cases, fault injection, deterministic replay, and scaling trends.

Synthetic success is not LLM success.

### Phase C — small real-model falsification

Use only available unmodified small checkpoints. Required measurements include held-out prompts, future-information audit, forward/layer calls, token/logit agreement, fallback, CPU time, RAM, and size trend.

Purpose:

> Falsify the proposed execution principle early on real Transformer checkpoints.

Small-model evidence is never 405B performance evidence.

### Phase D — target hardware

Requires a real 8 GiB GPU, target storage, 70B/405B checkpoints, baseline runtime, and hardware profilers. Only this phase can validate actual target VRAM, TTFT, tokens/second, PCIe, SSD, and original 405B quality.

Current status: **NOT TESTED**.

## Evidence levels

Use exactly:

- E0: idea or equation;
- E1: synthetic/reference validation;
- E2: real small-model operation replacement;
- E3: held-out generalization with measured causal coverage;
- E4: measured improvement on accessible representative hardware;
- E5: medium/large model scaling validation;
- E6: target model runs under 8 GiB VRAM;
- E7: 405B meets the declared 4B-class performance target.

E0–E3 may not be described as E6/E7 feasibility or success.

## Provenance labels

Every metric and claim must be separated into:

- `MEASURED` — produced by an actual run in the declared environment;
- `DERIVED` — exact formula or calculation from measured inputs;
- `PROJECTED` — extrapolation to another model or machine;
- `UNVERIFIED` — not tested in the current environment.

Never present PROJECTED or UNVERIFIED values as MEASURED.

## Core-research filter

Core research must directly answer all twelve questions in `RESEARCH_STATE.md`, including:

- the original operation skipped or replaced;
- causal selector without future tokens;
- selector cost;
- wrong-skip detection;
- exact/specified fallback;
- worst-case output contract;
- scaling trend;
- reason all weights need not be read;
- RAM/SSD/VRAM movement;
- 405B minimum bandwidth/compute;
- distance from the 4B target;
- strongest falsification.

Token-path storage, response replay, file compression, or bounded grammar memorization are auxiliary unless attached to a new causal operation-skipping principle.

## Research-efficiency and candidate-selection Gate

`docs/RESEARCH_EFFICIENCY_CONTRACT.md` is mandatory and has the same authority as the proof-first contract.

Before opening an experiment branch, a proposed core candidate must pass E0 triage:

- show a credible optimistic path to at least an order-of-magnitude reduction and toward the final target-equivalent fraction;
- identify a materially new mechanism or new evidence rather than a nearby variant of a rejected family;
- explain why the effect should survive or improve with model scale;
- include selector, metadata, intermediate, verification, correction, fallback, RAM, SSD, PCIe, and VRAM costs;
- define the cheapest decisive falsification before backend or kernel work;
- preserve the fixed arbitrary-model, unmodified-checkpoint, runtime-only, fail-closed mission.

Do not run experiments merely because a mathematical decomposition or optimization has not yet appeared in the sequence. Do not complete a taxonomy for its own sake.

An optimization whose favorable ceiling is only a few tens of percent is auxiliary unless an independently justified composition closes the remaining orders-of-magnitude gap. Population-level p50/p90 behavior controls promotion; an isolated best matrix, prompt, row, head, or synthetic fragment does not.

Repeated negative evidence closes a mechanism family. Reopening it requires a new information source, asymptotic mechanism, execution dependency, or measured fact that invalidates the prior rejection premise. Parameter sweeps, mode-order variants, rank changes, and renamed decompositions are not sufficient.

Use the cheapest-kill-first order:

```text
resource/information bound
-> exact certificate or favorable oracle upper bound
-> pinned small-real-checkpoint measurement
-> minimal operation replacement
-> backend/kernel
-> target hardware
```

No model-wide backend, physical kernel, or broad rescue search may start before the cheaper Gate survives.

Default research prioritization is approximately 70% high-upside new execution paradigms, 20% cheap falsification and certificates, and 10% auxiliary engineering. This is a prioritization rule, not fabricated time accounting.

## Mandatory proof-first loop

Before a model-wide backend is built:

1. read previous state, failures, decisions, assumptions, and the efficiency contract;
2. select one falsifiable high-upside core hypothesis that passed E0 candidate triage;
3. define success and rejection thresholds plus the optimistic target-scale ceiling;
4. derive correctness, memory, traffic, compute, and fallback equations;
5. identify all unverified assumptions;
6. implement an independent reference;
7. implement only the minimum candidate required by the cheapest decisive Gate;
8. run the strongest current-environment falsification;
9. stop immediately when a decisive rejection bound is established;
10. save raw logs, processed results, and checksums;
11. update all required root documents and experiment files;
12. commit before reporting progress.

Do not increase experiment numbers without eliminating a real assumption or testing a new mechanism. Do not build an optimized implementation to reconfirm a decisive negative theorem, lower bound, or oracle ceiling.

## Safety and correctness

- Future generated tokens are forbidden unless the experiment is explicitly labeled a non-deployable oracle upper bound.
- A failed or unavailable certificate must trigger exact fallback or abort, never silent approximation.
- Probabilistic certification must declare and union-account its error budget; it is not deterministic exactness.
- Missing files, checkpoint download failures, and runner timeouts are infrastructure failures, not scientific evidence.
- Failed hypotheses are permanent project data.

## Required repository state

Maintain on every meaningful session:

```text
RESEARCH_STATE.md
NEXT_EXPERIMENT.md
DECISION_LOG.md
FAILED_APPROACHES.md
ARCHITECTURE.md
ASSUMPTION_REGISTER.md
VALIDATION_MATRIX.md
HARDWARE_VALIDATION_PLAN.md
REPRODUCIBILITY.md
```

Experiment layout:

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/
results/exp_xxx/
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

Before a user-facing progress response after repository work, commit the current state. If writing or validation fails, say so explicitly.

## Active frontier

Read `NEXT_EXPERIMENT.md`.

EXP-066 Tensor-Train/MPO is authorized only as a bounded cheap-kill bond-rank certificate Gate. Exact MPO reconstruction, runtime integration, kernels, and broad factorization rescue searches are prohibited unless the lower-bound Gate survives.

If EXP-066 fails, close exact classical single-matrix tensor factorization as the primary direction for the measured population. The next core candidate must change execution class and pass E0 efficiency triage, prioritizing high-upside joint multi-operator arithmetic compilation or certificate-guided demand-driven execution rather than another adjacent decomposition.
