# Session Handoff

Last updated: 2026-08-03 Asia/Seoul

## Mandatory next-session startup

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
11. EXP-047 document and result summary
12. PR #56 and workflow `30792813542`.

Root files are primary. Conversation memory is not authoritative.

## Fixed target

Real arbitrary unmodified Hugging Face dense model, runtime replacement only, 405B flagship, <=8 GiB VRAM, original contract preserved, and 4B-class user experience.

## Environment truth

```text
8 GiB target GPU: unavailable
405B storage/execution: unavailable
CUDA/PCIe/target SSD profiling: unavailable
real 405B TTFT/tokens/sec/VRAM: NOT TESTED
```

Do not infer Phase D from GitHub Actions CPU.

## Branch/PR

```text
branch: research/governance-exp047-cptc
PR: #56
final EXP-047 workflow: 30792813542
source implementation SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
evidence head immediately after workflow: 3359371762c004db3532ebb16872b4eee85accf6
results: results/exp_047/
```

The branch will have later documentation-only commits. They do not regenerate EXP-047 measurements.

## Governance completed

Created and mandated nine root state documents. Updated `AGENTS.md`, proof contract, and session protocol with Phase A–D, E0–E7, provenance labels, Phase-D NOT TESTED, core operation-skipping filter, future-information audit, and fallback rules.

Existing mmap/index/DAG work remains auxiliary. Raw prefix scaling remains rejected.

## EXP-047 mechanism

Causal tile sampling without replacement, alpha-spending Serfling intervals, and exact full-tile fallback.

```text
delta_n = delta * 6/(pi^2 n^2)
```

No future generated tokens.

## Final authoritative result

### MEASURED

```text
phase A/B, evidence E1
10 tests passed
525 cases
certified 4
fallback 521 = 99.238%
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial exact fallback 15/15
Phase D NOT TESTED
```

Largest positive control:

```text
107/1024 tiles = 10.449%
```

Broad scaling:

```text
N64/N128/N256: 100% evaluated
N512: 98.519% mean evaluated
N1024: 98.294% mean evaluated
Python optimized/reference: about 9.2–9.7x
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
E1 certificate/fallback correctness: ACCEPT
Global-range CPTC-v1 performance: REVISE
Direct real-operation backend: BLOCKED
Phase D: NOT TESTED
E6/E7: not achieved
```

Do not report the pre-registered positive-control Gate as architecture success.

## Infrastructure history

Excluded from scientific interpretation:

- `30791055142`: optional dependency import failure;
- `30791192434`: runner path failure.

Final authoritative success: `30792813542`.

## Next decisive work

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`.

Use held-out current-token states from available unmodified small checkpoints. Fully computed exact tile contributions are an explicit non-deployable oracle used only to falsify upper bounds and detect wrong accepts.

Compare:

1. current global range;
2. exact per-state oracle range;
3. deployable checkpoint-derived stratified bounds;
4. independently proven variance-adaptive finite-population bounds.

Reject range-only CPTC if oracle-tight median evaluated fraction >10%, p90 >25%, any wrong accept occurs, or overhead/fallback exceeds full reference.

Offline audit is below E2. Actual operation replacement comes only after a deployable bound has useful held-out coverage.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
cd results/exp_047 && sha256sum -c checksums.sha256
```

Do not overwrite `results/exp_047/` when starting EXP-047R.
