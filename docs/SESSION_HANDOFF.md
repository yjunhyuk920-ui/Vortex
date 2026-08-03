# Session Handoff

Last updated: 2026-08-03 Asia/Seoul

## First action next session

Read, in order:

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
11. `docs/research/EXPERIMENT_047_CAUSAL_PROBABILISTIC_TILE_CERTIFICATE.md`
12. `results/exp_047/summary.json`
13. PR #56 and workflow `30791851508`.

The root files are now primary. This handoff remains for compatibility.

## Fixed target

Real arbitrary unmodified Hugging Face dense model, runtime replacement only, 405B flagship, <=8 GiB VRAM, original contract preserved, and 4B-class user experience.

Do not lower the target.

## Current environment truth

```text
8 GiB target GPU: unavailable
405B checkpoint storage/execution: unavailable
CUDA/PCIe/target SSD profiling: unavailable
real 405B TTFT/tokens/sec/VRAM: NOT TESTED
```

GitHub Actions CPU results are Phase A/B or limited small-model Phase C only.

## Current branch and PR

```text
branch: research/governance-exp047-cptc
PR: #56
base main at branch creation: 3f3ee348e8a72070b3e43cf7af56c078fc4d83c7
latest authoritative evidence workflow: 30791851508
source head recorded by evidence: d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b
current evidence directory: results/exp_047/
```

PR #56 remains draft until final full CI and factual PR update complete.

## Governance completed

Created and mandated:

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

Updated `AGENTS.md`, proof contract, and session protocol with:

- Phase A/B/C/D;
- E0–E7;
- MEASURED/DERIVED/PROJECTED/UNVERIFIED;
- Phase-D NOT TESTED rule;
- core unseen-prompt operation-skipping filter;
- future-information and fallback requirements.

Existing mmap/index/DAG work is preserved as auxiliary. Raw prefix memoization remains rejected as a core mechanism.

## EXP-047 mechanism

Causal randomized sampling without replacement over decision-relevant linear tiles, with alpha-spending Serfling intervals and exact fallback.

```text
delta_n = delta * 6/(pi^2 n^2)
```

No future generated tokens are used. Early commit is probabilistic under declared range assumptions; exact fallback evaluates every remaining tile.

## Authoritative result

### MEASURED

```text
phase: A/B
evidence: E1
tests: 10 passed
cases: 525
certified: 4
fallback: 521 = 99.238%
wrong accepts: 0
fallback mismatches: 0
independent-bound mismatches: 0
adversarial fallback: 15/15
Phase D: NOT TESTED
```

Largest positive control:

```text
107/1024 tiles = 10.449%
```

Broad trend:

```text
N64/N128/N256: 100% evaluated
N512: 98.519% mean evaluated
N1024: 98.294% mean evaluated
Python optimized path: roughly 8.8–9.1x simple full reference
```

### PROJECTED

```text
405B Q4 full stream: 188.593 GiB
1.2x 4B Q4 allowance: 2.235 GiB/token
required fraction before overhead: 1.185%
positive control gap: 8.817x above target fraction
```

## Decision

```text
E1 correctness primitive: ACCEPT
Global-range CPTC-v1 core performance: REVISE
Direct Phase-C operation backend: BLOCKED
Phase D: NOT TESTED
```

Do not say CPTC solved execution. The pre-registered positive control passed, but broad performance did not.

## Infrastructure history

Non-authoritative failed workflows:

- `30791055142`: eager optional `safetensors` import;
- `30791192434`: missing repo root on runner `PYTHONPATH`.

Both were infrastructure failures. Lazy imports and explicit `PYTHONPATH` fixed them.

Authoritative success: `30791851508`.

## Next decisive work

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit` in `NEXT_EXPERIMENT.md`.

Goal:

Determine whether CPTC-v1 failed because real contribution distributions are intrinsically hard or because one global range is too loose.

Compare on held-out states from available unmodified small checkpoints:

1. current global range;
2. exact per-state oracle min/max, explicitly non-deployable;
3. checkpoint-derived static stratified bounds;
4. only mathematically justified variance-adaptive finite-population bounds.

First operation: LM-head winner-versus-runner tile contribution audit. Full baseline calls and exact contributions must be charged and marked oracle/non-deployable.

Reject range-only CPTC if oracle-tight median evaluated fraction >10% or p90 >25%, or if wrong accepts occur.

Offline audit is not E2. Actual operation replacement comes only after a deployable bound shows useful held-out coverage.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
```

Do not overwrite `results/exp_047/` when starting EXP-047R; use a separate experiment/result directory.
