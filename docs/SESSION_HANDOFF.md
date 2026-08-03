# Session Handoff

Last updated: 2026-08-03 Asia/Seoul

## Mandatory startup

Read:

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
11. EXP-047 document and frozen result summary
12. PR #56.

Root files and machine-readable result JSON are authoritative; conversation memory is not.

## Fixed target

Real arbitrary unmodified Hugging Face dense model, runtime only, 405B flagship, <=8 GiB VRAM, original contract preserved, and 4B-class user experience.

## Environment truth

```text
8 GiB target GPU unavailable
405B storage/execution unavailable
CUDA/PCIe/target SSD profiling unavailable
real 405B TTFT/tokens/sec/VRAM NOT TESTED
```

## Branch and frozen evidence

```text
branch research/governance-exp047-cptc
PR #56
authoritative file results/exp_047/summary.json
frozen workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
Phase D NOT TESTED
```

`.github/workflows/exp_047_gate.yml` is now manual-only and never mutates the repository. Documentation updates cannot regenerate frozen measurements.

## Governance completed

Nine root state files plus updated `AGENTS.md`, proof contract, and session protocol enforce Phase A–D, E0–E7, provenance labels, Phase-D NOT TESTED, core operation-skipping filter, future-information audit, and fallback rules.

Existing mmap/index/DAG work is auxiliary. Raw prefix scaling remains rejected.

## EXP-047 result

Mechanism: causal random sampling without replacement over decision-relevant tiles, alpha-spending Serfling intervals, exact fallback.

### MEASURED

```text
10 tests passed
525 cases
certified 4
fallback 521 = 99.238%
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial fallback 15/15
future generated tokens false
Phase D NOT TESTED
```

```text
positive control 107/1024 = 10.449%
N64/N128/N256 100% evaluated
N512 98.519% mean evaluated
N1024 98.294% mean evaluated
Python optimized/reference about 8.6–9.1x
```

### PROJECTED

```text
405B Q4 stream 188.593 GiB
1.2x 4B allowance 2.235 GiB/token
required fraction before overhead 1.185%
positive-control gap 8.817x
```

## Decision

```text
E1 certificate/fallback correctness ACCEPT
Global-range CPTC-v1 performance REVISE
Direct real-operation backend BLOCKED
Phase D NOT TESTED
E6/E7 not achieved
```

Do not report the primitive positive-control pass as architecture success.

## Infrastructure history

Excluded from science:

- `30791055142` optional dependency import;
- `30791192434` runner path failure.

## Next work

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`.

Use held-out current-token states from available unmodified small checkpoints. Full exact tile contributions are an explicit non-deployable analysis oracle used to falsify upper bounds and catch wrong accepts.

Compare:

1. global range;
2. exact per-state oracle range;
3. deployable checkpoint-derived stratified bounds;
4. independently proven variance-adaptive finite-population bounds.

Reject range-only CPTC if oracle-tight median evaluated fraction >10%, p90 >25%, any wrong accept occurs, or overhead/fallback exceeds full reference.

Offline audit remains below E2. Actual operation replacement follows only after deployable bounds show useful held-out coverage.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
cd results/exp_047 && sha256sum -c checksums.sha256
```

Do not overwrite `results/exp_047/` when starting EXP-047R.
