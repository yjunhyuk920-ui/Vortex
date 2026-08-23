# VORTEX

VORTEX is a proof-first project for executing arbitrary public, unmodified dense Hugging Face Transformers by replacing only the executor.

## Fixed mission

- arbitrary public dense 405B-class checkpoint;
- one 8-GiB GPU;
- no retraining, fine-tuning, distillation, LoRA, or semantic target-weight modification;
- exact declared token/output and required successor-state contract;
- same-machine native-4B-Q4 warm latency `p50 <=1.2x`, `p95 <=1.5x`;
- full accounting of storage, traffic, arithmetic, KV/state, metadata, packing, verification, repair, fallback, and synchronization.

## Research workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Research, public-checkpoint execution, tests, measurements, and evidence generation happen locally. After validation, GitHub stores the source, result, logs, checksums, ledgers, and handoff. Duplicate GitHub Actions execution is not required unless explicitly requested.

## Reality-first rule

`REAL_EXECUTOR_ONLY` is authoritative. Future target states, perfect selectors, free `N/A`, free transforms/workspace/repair/fallback, unmeasured compression, hidden compute, or free HBM scans cannot promote a mechanism.

## Resource objective

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right),
\qquad A\gg1,\quad N/A\to1,\quad r\ll1.
\]

## Latest authoritative results

### EXP-102A — external causal drafts

```text
REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

### EXP-103A — native-rounding page certificates

```text
REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
```

### EXP-104A — checkpoint-native resident slice

```text
REJECT_FULL_VOCABULARY_FULL_LAYER_RESIDENT_SLICE_AS_P50_CORE
```

A realistic group-128 Q4 full vocabulary head plus one complete 405B-width target layer requires `2,809,790,464` resident weight bytes per proposed token, or `1.404895232x` the ideal native-4B-Q4 weight population. Even metadata-free Q4 requires `1.322254336x`. Both exceed the final `1.2x` p50 target before target verification or other costs.

A fully charged 8-GiB capacity ledger can hold the Q4 head and three complete layers, but each additional layer makes the per-token scan worse. Capacity is not latency.

See:

- `docs/research/EXPERIMENT_104A_CHECKPOINT_NATIVE_RESIDENT_SLICE_GATE.md`
- `docs/research/EXP_104A_LATEST_RESULT.md`
- `results/exp_104a/local/result.json`

## Active frontier — EXP-105A

The next Gate must either avoid scanning the complete 128,256-row vocabulary head with a new causal information source or produce multiple useful candidate tokens per resident weight scan. Another full-head layer slice is closed.

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

Complete 405B execution, physical 8-GiB allocation, target hardware traffic, CUDA/SASS, and final same-machine p50/p95 remain `NOT TESTED`.

```text
README_CURRENT=true
README_UPDATED=EXP-104A authoritative result; EXP-105A active frontier
```
