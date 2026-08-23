# VORTEX

VORTEX is a proof-first research runtime for executing very large, publicly released Hugging Face dense Transformers under a small GPU-memory budget **without changing the model's meaning**.

> **Current truth:** the final 405B-on-one-8-GiB-GPU-at-4B-class-latency target has not yet been achieved. This repository contains executable runtime prototypes, exactness/resource auditors, falsification Gates, raw evidence, and an active constructive research program.

## Fixed mission

Run an arbitrary public, unmodified Hugging Face dense 405B-class model by replacing only the executor, with:

- peak GPU VRAM `<= 8 GiB`;
- no retraining, fine-tuning, distillation, LoRA, semantic weight modification, or user-authored model-specific adapter;
- the declared original output and required successor-state contract preserved;
- warm p50 time/token `<= 1.2x` a native 4B Q4 baseline on the same machine;
- warm p95 `<= 1.5x` that baseline;
- independently reproducible evidence from pinned code and checkpoint hashes.

The target is not “make 405B run slowly with offload.” It is:

\[
\boxed{
\text{exact 405B behavior}
+
\text{single 8 GiB GPU}
+
\text{same-machine 4B-class latency}
}
\]

The canonical goal and working-principles answer is in [`MISSION_AND_WORKING_PRINCIPLES.md`](MISSION_AND_WORKING_PRINCIPLES.md).

## Non-negotiable research rules

- Arbitrary public dense checkpoints, not a cherry-picked model or prompt.
- Executor-only: no training or semantic checkpoint modification.
- Exact/fail-closed output and successor-state behavior under the declared ABI.
- Full accounting of GPU/CPU/SSD/PCIe/HBM traffic, KV/state, metadata, decompression, packing, verification, repair, synchronization, candidate count, committed tokens, and fallback.
- `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED` are kept separate.
- A closed mechanism is not reopened by only changing seed, rank, threshold, tile, or block length.
- Every core round begins with three materially different execution principles and implements only the strongest candidate with a credible `>=10x` path.
- Meaningful positive and negative results are preserved in a verified remote commit.
- README freshness is part of round completion.

## Reality-first authoritative execution

Every new core Gate has exactly one authoritative arm:

```text
REAL_EXECUTOR_ONLY
```

The authoritative result cannot be promoted using:

- future target tokens or hidden states;
- a perfect target-seeing selector or perfect accepted block;
- free `N/A`, transform, addition, packing, metadata, workspace, repair, rollback, verification, or fallback;
- unmeasured compression or ideal peak throughput represented as measurement;
- an unimplemented arithmetic instruction or hidden/remote execution omitted from the ledger.

Oracle calculations remain useful as diagnostic controls, but they are `DIAGNOSTIC_ONLY` and cannot authorize a backend or count as deployable progress. See [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md).

## Research workflow: sandbox first, GitHub last

\[
\boxed{
\textbf{Fast computation and iteration in the sandbox first;}
\quad
\textbf{persistent evidence and final reproduction in GitHub last.}
}
\]

```text
SANDBOX_RESEARCH
-> SANDBOX_GATE
-> SOURCE_COMMIT_PUSHED
-> WORKFLOW_RUNNING       # only when hosted reproduction adds value
-> RESULT_COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Use the sandbox for derivations, search, cost models, prototypes, counterexamples, and focused tests. Do **not** turn each hypothesis iteration into a GitHub Actions run.

Use GitHub for frozen source/config, independent reproduction, public-checkpoint or long hosted runs, immutable artifacts, checksums, PR discussion, and cross-session handoff.

## Resource objective

At minimum, the core must confront the dual roofline:

\[
T_{\rm token}
\ge
\max\left(
\frac{S_c}{B A},
\;
r\frac{N}{A}\frac{2P}{F}
\right).
\]

The project must jointly obtain:

\[
A\gg1,\qquad N/A\rightarrow1,\qquad r\ll1.
\]

Large accepted blocks alone do not solve the target when the exact target still performs `r = 1` dense arithmetic per committed token.

## Current research status

### Final target

```text
405B target execution:                     NOT TESTED / NOT ACHIEVED
physical complete 8-GiB allocation:        NOT TESTED
same-machine native-4B p50/p95 acceptance: NOT TESTED
```

### Latest completed arithmetic Gate

EXP-100A tested catalogued small-coefficient rectangular fast-matrix-multiplication programs with fully charged transforms, moves, cold bytes, workspace, and native-order repair.

```text
Decision: REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE
best fully charged explicit arithmetic fraction   38.251649686367%
best free-transform rank-oracle fraction           13.010262621991%
required first-core boundary                       10%
```

EXP-101A was subsequently closed as `DIAGNOSTIC_ONLY`, because its multiplication-only Gate granted future blocks and zero-cost runtime components. Its artifact may inform algebraic work, but it cannot establish deployable progress.

### Active frontier — EXP-102A

```text
Branch: research/exp-102a-causal-segment-delta-reality-gate
Gate:   EXP-102A reality-first causal draft/verify
```

EXP-102A executes real public draft and target checkpoints. Its authoritative arm charges:

- draft prefill and autoregressive generation;
- cross-tokenizer decode/retokenization bridge;
- every target candidate position, argmax, and comparison;
- measured `N/A`;
- mismatch repair;
- real same-tokenizer cache crop/replay or cross-tokenizer state rebuild;
- model/cache/RSS bytes and wall time;
- exact token and complete terminal target-KV equality.

No compression credit is used. Target prompt prefill is excluded symmetrically as the declared warm-prefix boundary. Until a remote result commit exists, EXP-102A is an active Gate, not a scientific result.

For current truth, read the remote branch/PR/workflow and:

1. [`RESEARCH_STATE.md`](RESEARCH_STATE.md)
2. [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md)
3. [`docs/research/VORTEX_RESEARCH_HANDOFF.md`](docs/research/VORTEX_RESEARCH_HANDOFF.md)
4. raw `results/exp_*` evidence and checksums
5. [`DECISION_LOG.md`](DECISION_LOG.md) and [`FAILED_APPROACHES.md`](FAILED_APPROACHES.md)

Conversation memory is not authoritative.

## Repository map

- `vortex_runtime/` — runtime primitives, exactness/resource helpers, and experiment mechanisms.
- `experiments/` — frozen experiment runners and configs.
- `tests/` — focused, property, regression, and control tests.
- `results/` — committed processed/raw evidence and checksum ledgers.
- `docs/research/` — preregistrations, latest-result summaries, contracts, and handoff.
- `.github/workflows/` — clean hosted reproduction and artifact production.
- root ledgers — current state, decisions, failures, assumptions, architecture, validation, hardware plan, and reproducibility.

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

Some frozen experiments use additional pinned requirements. Follow the active experiment README/workflow rather than silently installing newer dependencies.

Inspect a local Hugging Face safetensors model:

```bash
python -m vortex_runtime.cli inspect /path/to/model
```

Run the tiny streamed-Llama demo:

```bash
python -m vortex_runtime.cli demo --tokens 8 --budget-mb 2
```

These commands exercise prototypes; they do not constitute final target validation.

## Start here in a new session

Read in this order:

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
13. the active experiment, PR, workflow, raw result, and checksum files

Then verify the actual remote branch/head before proposing another mechanism.

## README freshness contract

Review this file during every meaningful repository round. Update it when the mission, mandatory workflow, active frontier, latest authoritative result, quick start, repository map, or advertised capability changes. When no update is required, record a specific `README_UNCHANGED_REASON`.

## Claim boundary

Only a real target-hardware Phase-D/E6/E7 run can establish actual 405B execution, physical peak VRAM, target traffic, or same-machine 4B-class p50/p95. Until then, those claims remain `NOT TESTED`.

```text
README_CURRENT=true
README_UPDATED=reality-first contract, EXP-101A diagnostic closure, EXP-102A active frontier
```

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
