# VORTEX

VORTEX is a proof-first research runtime for executing arbitrary public, unmodified dense Hugging Face Transformers under a small GPU-memory budget without changing the model's meaning.

## Fixed mission

Run an arbitrary public dense 405B-class checkpoint by replacing only the executor, with:

- peak GPU VRAM `<= 8 GiB`;
- no retraining, fine-tuning, distillation, LoRA, or semantic target-weight modification;
- exact declared token/output and required successor-state contract;
- same-machine native-4B-Q4 warm latency `p50 <=1.2x`, `p95 <=1.5x`;
- reproducible pinned source, inputs, results, logs, and checksums.

## Reality-first rule

The authoritative arm is always `REAL_EXECUTOR_ONLY`. Future target states, perfect selectors, free `N/A`, free transforms/workspace/repair/fallback, unmeasured compression, hidden compute, and projected peak throughput cannot satisfy promotion.

## Local-first workflow

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Research, checkpoint execution, tests, measurements, and evidence generation happen locally. After local validation, GitHub is used only to commit/push the already validated source/result/handoff and read back the remote SHA. Duplicate GitHub Actions execution is not required unless the user explicitly asks.

## Resource objective

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right),
\qquad A\gg1,\quad N/A\to1,\quad r\ll1.
\]

Accepted length alone is not a core result when exact target arithmetic remains `r=1`.

## Latest authoritative results

### EXP-102A — external causal drafts

```text
REJECT_FROZEN_REAL_CAUSAL_EXTERNAL_DRAFTS_AS_85_TOKEN_AMORTIZATION_SOURCE
```

Real public draft/target checkpoints preserved exact token and terminal KV state, but untouched-holdout minimum committed tokens were only `1` and `2`.

### EXP-103A — native-rounding page certificates

```text
REJECT_HIERARCHICAL_NATIVE_ROUNDING_PAGE_CERTIFICATES_AS_UNIVERSAL_10X_CORE
```

The exact BF16/FP32 certificate passed 200,000 randomized soundness cases with zero mismatches. However, metadata plus the p50 traffic budget requires roughly 99% page skipping; a legal all-ones dense row and 64 Gaussian BF16 controls read 100% of pages. The mechanism is retained only as an auxiliary.

See:

- `docs/research/EXPERIMENT_103A_NATIVE_ROUNDING_PAGE_CERTIFICATE_GATE.md`
- `docs/research/EXP_103A_LATEST_RESULT.md`
- `results/exp_103a/local/result.json`

## Active frontier — EXP-104A

`Checkpoint-Native Resident Slice Transducer Reality Gate`.

The next candidate uses a fully charged <=8-GiB target-derived resident draft representation, exact embedding-row traffic, real causal generation, unchanged-target verification, and exact terminal-state checking. It must demonstrate an actual accepted segment `A >=339` for the raw-BF16 p50 floor unless it measurably reduces exact target sweep bytes.

Read `NEXT_EXPERIMENT.md`, `RESEARCH_STATE.md`, and `docs/research/VORTEX_RESEARCH_HANDOFF.md` before continuing.

## Repository map

- `vortex_runtime/` — runtime primitives;
- `experiments/` — frozen local experiment runners/configs;
- `tests/` — focused/property/regression controls;
- `results/` — committed raw/processed evidence and checksums;
- `docs/research/` — experiment contracts, latest results, and handoff;
- root ledgers — current state and next Gate.

## Quick start

```bash
git clone https://github.com/yjunhyuk920-ui/Vortex.git
cd Vortex
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e . pytest
python -m pytest -q
```

Target 405B execution, physical complete 8-GiB allocation, target CUDA/SASS, storage/H2D throughput, and same-machine 4B-Q4 acceptance remain `NOT TESTED`.

```text
README_CURRENT=true
README_UPDATED=EXP-103A authoritative result; EXP-104A active frontier; local-commit-only workflow retained
```
