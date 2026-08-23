# VORTEX

VORTEX is a proof-first project for executing arbitrary public, unmodified dense Hugging Face Transformers by replacing only the executor.

## Fixed mission

- arbitrary public dense 405B-class checkpoint;
- one 8-GiB GPU;
- no retraining, fine-tuning, distillation, LoRA, or semantic target-weight modification;
- exact declared output and required successor-state contract;
- same-machine native-4B-Q4 warm latency `p50 <=1.2x`, `p95 <=1.5x`;
- full accounting of storage, traffic, arithmetic, KV/state, metadata, packing, verification, repair, fallback, and synchronization.

## Local-first workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Research, checkpoint execution, tests, measurements, and evidence generation happen locally. GitHub stores the validated source/results and handoff. Duplicate GitHub Actions execution is not required unless explicitly requested.

## Reality-first rule

`REAL_EXECUTOR_ONLY` is authoritative. Future target state, perfect selectors, free `N/A`, free transforms/workspace/repair/fallback, unmeasured compression, hidden compute, and free HBM scans cannot promote a mechanism.

## Resource objective

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right),
\qquad A\gg1,\quad N/A\to1,\quad r\ll1.
\]

## Latest authoritative results

### EXP-104A — complete-layer resident slice

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

### EXP-105A — exact progressive vocabulary tournament

```text
REJECT_ACTIVATION_ORDERED_EXACT_RESIDUAL_BOUND_HEAD_INDEX_WITH_COMPLETE_LAYER_AS_P50_CORE
NO_REGISTERED_EXP_105A_PRINCIPLE_SURVIVES_COMPLETE_EXECUTOR_GATE
```

EXP-105A implemented an exact Q4/Q8 query-dependent head tournament. It processed activation dimensions in descending magnitude and eliminated rows using exact Cauchy tail bounds. Eight focused tests, 72 random finite-word executions, positive/negative controls, a deterministic rerun, and checksums passed.

Key result:

```text
best legal target-scale late-decision head query   1,122,065,408 bytes/token
one complete realistic-Q4 target-width layer      1,693,450,240 bytes/token
combined                                           2,815,515,648 bytes/token
p50 budget                                         2,400,000,000 bytes/token
combined / budget                                  1.17313152x
```

Capacity fits (`4,330,597,376` resident bytes in the registered ledger), but per-token traffic does not. The head alone is not promoted because it lacks a concrete causal state source and measured `A>=339` continuation.

See:

- `docs/research/EXPERIMENT_105A_ACTIVATION_ORDERED_EXACT_HEAD_TOURNAMENT_GATE.md`
- `docs/research/EXP_105A_LATEST_RESULT.md`
- `results/exp_105a/local/result.json`

## Active frontier — EXP-106A

`Depth-Complete Width-Thin Checkpoint Surrogate Reality Gate`.

Instead of a few complete target-width layers, the next candidate derives a narrow training-free draft-only state through every target layer. The full bridge, narrow operators, nonlinear state, head/index, KV, workspace, and per-token traffic must fit the 8-GiB and 2.4-GB-equivalent p50 Gates before public-checkpoint execution. Actual accepted prefix, not configured width or K, is authoritative.

Read `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, and `docs/research/VORTEX_RESEARCH_HANDOFF.md` before continuing.

## Quick start

```bash
git clone https://github.com/yjunhyuk920-ui/Vortex.git
cd Vortex
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e . pytest
python -m pytest -q
```

Complete 405B execution, physical 8-GiB target allocation, target CUDA/SASS, storage/H2D throughput, and final same-machine latency remain `NOT TESTED`.

```text
README_CURRENT=true
README_UPDATED=EXP-105A authoritative result; EXP-106A active frontier; local-commit-only workflow retained
```
