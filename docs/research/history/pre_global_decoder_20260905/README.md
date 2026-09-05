# VORTEX

VORTEX is a proof-first research runtime for executing very large, publicly released Hugging Face dense Transformers under a small GPU-memory budget **without changing the model's meaning**.

> **Current truth:** the final 405B-on-one-8-GiB-GPU-at-4B-class-latency target has not yet been achieved. The repository contains executable prototypes, falsification Gates, raw evidence, resource ledgers, and active constructive research.

## Fixed mission

Run an arbitrary public, unmodified Hugging Face dense 405B-class model by replacing only the executor, with:

- peak GPU VRAM `<= 8 GiB`;
- no retraining, fine-tuning, distillation, LoRA, semantic weight modification, or user-authored model-specific adapter;
- the declared original output and required successor-state contract preserved;
- warm p50 time/token `<= 1.2x` a native 4B Q4 baseline on the same machine;
- warm p95 `<= 1.5x` that baseline;
- reproducible evidence from pinned code, configs, inputs, results, and hashes.

The target is not “make 405B run slowly with offload.” It is:

\[
\boxed{\text{exact 405B behavior}+\text{single 8 GiB GPU}+\text{same-machine 4B-class latency}}
\]

Canonical goal and working principles: [`MISSION_AND_WORKING_PRINCIPLES.md`](MISSION_AND_WORKING_PRINCIPLES.md).

## Non-negotiable research rules

- Arbitrary public dense checkpoints, not a cherry-picked model or prompt.
- Executor-only: no training or semantic checkpoint modification.
- Exact/fail-closed output and successor-state behavior under the declared ABI.
- Full accounting of GPU/CPU/SSD/PCIe/HBM traffic, KV/state, metadata, decompression, packing, verification, repair, synchronization, candidate count, committed tokens, and fallback.
- `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED` are kept separate.
- A closed mechanism is not reopened by only changing seed, rank, threshold, tile, or block length.
- Every core round begins with three materially different execution principles and implements only the strongest realistic candidate with a credible `>=10x` path.
- `REAL_EXECUTOR_ONLY` is the authoritative arm; impossible favorable grants cannot promote a mechanism.
- Meaningful positive and negative results are preserved in a verified remote commit.
- README freshness is part of round completion.

## Reality-first authoritative execution

The authoritative result cannot be promoted using:

- future target tokens or hidden states;
- a perfect target-seeing selector or perfect accepted block;
- free `N/A`, transform, addition, packing, metadata, workspace, repair, rollback, verification, or fallback;
- unmeasured compression or ideal peak throughput represented as measurement;
- an unimplemented arithmetic instruction or hidden/remote execution omitted from the ledger.

Oracle calculations may exist only as `DIAGNOSTIC_ONLY` controls. See [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md).

## Research workflow: local validation, GitHub commit only

\[
\boxed{\textbf{Do the research and validation locally; use GitHub only to persist and hand off the validated result.}}
\]

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Use the local/sandbox environment for derivations, search, cost models, prototypes, counterexamples, unit/property/regression tests, public-checkpoint execution, raw evidence, logs, and checksums.

After `LOCAL_VALIDATION_PASS`, **do not rerun the same experiment on GitHub Actions**. Commit/push the already validated source, config, result, logs, checksums, canonical ledgers, and README; then read back the remote SHA.

Default policy:

```text
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
```

GitHub Actions is exceptional and used only when the user explicitly requests an additional hosted run. Automatic CI may still execute, but it is not a mandatory scientific approval Gate for a locally validated round.

## Resource objective

At minimum:

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right).
\]

The project must jointly obtain:

\[
A\gg1,\qquad N/A\rightarrow1,\qquad r\ll1.
\]

Large accepted blocks alone do not solve the target when the exact target still performs `r=1` dense arithmetic per committed token.

## Current research status

### Final target

```text
405B target execution:                     NOT TESTED / NOT ACHIEVED
physical complete 8-GiB allocation:        NOT TESTED
same-machine native-4B p50/p95 acceptance: NOT TESTED
```

### EXP-100A

```text
Decision: REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE
best fully charged explicit arithmetic fraction   38.251649686367%
best free-transform rank-oracle fraction           13.010262621991%
required first-core boundary                       10%
```

### EXP-101A

EXP-101A was closed as `DIAGNOSTIC_ONLY` because its multiplication-only Gate granted future blocks and zero-cost runtime components. It cannot establish deployable progress.

### Latest authoritative completed Gate — EXP-102A

```text
REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

EXP-102A used real public draft and target checkpoints and charged draft generation, tokenizer bridge, target candidate positions, measured `N/A`, mismatch repair, draft-state rebuild, cache/RSS bytes, and wall time. No compression or impossible oracle credit was used. See `docs/research/EXP_102A_LATEST_RESULT.md`.

### Active frontier

The next research round is selected from the latest committed `NEXT_EXPERIMENT.md` and `RESEARCH_STATE.md`, but it must now be executed and validated locally first. Once validated, GitHub receives only the resulting commit/handoff; no duplicate hosted research run is required.

For current truth, read:

1. [`RESEARCH_STATE.md`](RESEARCH_STATE.md)
2. [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md)
3. [`docs/research/VORTEX_RESEARCH_HANDOFF.md`](docs/research/VORTEX_RESEARCH_HANDOFF.md)
4. committed raw `results/exp_*` evidence/checksums
5. [`DECISION_LOG.md`](DECISION_LOG.md) and [`FAILED_APPROACHES.md`](FAILED_APPROACHES.md)

Conversation memory is not authoritative.

## Repository map

- `vortex_runtime/` — runtime primitives and experiment mechanisms.
- `experiments/` — frozen experiment runners/configs.
- `tests/` — focused/property/regression/control tests.
- `results/` — committed processed/raw evidence and checksums.
- `docs/research/` — preregistrations, latest results, contracts, and handoff.
- `.github/workflows/` — legacy/optional hosted workflows; not required after local validation unless explicitly requested.
- root ledgers — state, decisions, failures, assumptions, architecture, validation, hardware plan, reproducibility.

## Quick start

```bash
git clone https://github.com/yjunhyuk920-ui/Vortex.git
cd Vortex
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

Some experiments use additional pinned requirements. Follow the active experiment's config/docs and preserve dependency versions used by the locally validated result.

## Start here in a new session

1. [`AGENTS.md`](AGENTS.md)
2. [`MISSION_AND_WORKING_PRINCIPLES.md`](MISSION_AND_WORKING_PRINCIPLES.md)
3. this README
4. [`docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md`](docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md)
5. [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md)
6. [`docs/research/VORTEX_RESEARCH_HANDOFF.md`](docs/research/VORTEX_RESEARCH_HANDOFF.md)
7. [`RESEARCH_STATE.md`](RESEARCH_STATE.md)
8. [`FAILED_APPROACHES.md`](FAILED_APPROACHES.md)
9. [`DECISION_LOG.md`](DECISION_LOG.md)
10. [`ASSUMPTION_REGISTER.md`](ASSUMPTION_REGISTER.md)
11. [`VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md)
12. [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md)
13. active experiment source/config/result/checksum files

Verify the actual remote branch/head before proposing another mechanism.

## README freshness contract

Review this file during every meaningful repository round. Update it when mission, mandatory workflow, active frontier, latest authoritative result, quick start, repository map, or advertised capability changes. If no update is needed, record a specific `README_UNCHANGED_REASON`.

## Claim boundary

Only a real target-hardware Phase-D/E6/E7 run can establish actual 405B execution, physical peak VRAM, target traffic, or same-machine 4B-class p50/p95. Until then those claims remain `NOT TESTED`.

```text
README_CURRENT=true
README_UPDATED=local-validation/GitHub-commit-only workflow; current authoritative EXP-102A status
```
