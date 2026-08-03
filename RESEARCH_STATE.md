# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- real 405B-class dense flagship;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored model-specific adapter;
- original declared ability/output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 on the same target machine;
- independently reproducible evidence.

The target is unchanged.

## Current environment

MEASURED capability:

- GitHub repository and GitHub Actions CPU runners;
- Python 3.10/3.12 CI;
- small downloadable checkpoints when time/storage permit.

Unavailable and therefore NOT TESTED:

- target 8 GiB GPU;
- 405B checkpoint storage/download/execution;
- target CUDA, PCIe, SSD, power, TTFT, tokens/second, and peak VRAM.

## Validation system

- Phase A theory/structure;
- Phase B synthetic/reference;
- Phase C small real-model falsification;
- Phase D actual target hardware, currently NOT TESTED.

Evidence: E0 idea, E1 synthetic/reference, E2 real small-model operation replacement, E3 held-out causal generalization, E4 accessible representative hardware, E5 medium/large scaling, E6 target under 8 GiB, E7 real 405B at declared 4B-class performance.

All claims separate MEASURED, DERIVED, PROJECTED, and UNVERIFIED.

## Strongest actual achievements

MEASURED:

- exact/checksummed mmap pointer VM;
- bounded unmodified TinyLlama decision-index compiler with 72/72 checked token replay inside its finite grammar;
- exact future-suffix DAG compressed 64 bounded records to 38 nodes;
- EXP-047 Phase-B certificate passed 10 tests and 525 generated cases with zero wrong accepts, zero fallback mismatches, and zero independent-bound mismatches.

These replay/index/DAG components are auxiliary, not the unseen-prompt operation-skipping principle.

## EXP-047 authoritative evidence

```text
branch: research/governance-exp047-cptc
PR: #56
workflow: 30792813542
source implementation SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
evidence commit/current branch head after bot commit: 3359371762c004db3532ebb16872b4eee85accf6
results: results/exp_047/
phase: A/B
evidence: E1
Phase D: NOT TESTED
```

### MEASURED correctness

```text
cases: 525
certified: 4
fallback: 521 = 99.238%
wrong accepts: 0
fallback/reference mismatches: 0
independent-bound mismatches: 0
adversarial exact fallback: 15/15
future generated tokens used: false
```

Largest positive cancellation control:

```text
1,024 tiles
107 sampled
10.449% evaluated before certificate
```

Broad scaling:

```text
N=64/128/256: 100% fallback and 100% tiles evaluated
N=512: mean evaluated 98.519%
N=1024: mean evaluated 98.294%
Python optimized/reference mean time: about 9.2–9.7x
```

### PROJECTED target gap

```text
405B Q4 full stream: 188.593 GiB
4B Q4 full stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required average evaluated fraction before overhead: 1.185%
positive-control fraction / target fraction: 8.817x
```

These are parameter-count projections, not target-hardware measurements.

## Scientific decision

- ACCEPT E1 correctness primitive: causal randomized sampling, alpha-spending Serfling calculation, independent interval check, deterministic replay, error rejection, and exact fallback passed.
- REVISE core architecture: one global range certified only 4/525 cases, evaluated about 98% of tiles overall, and was ~9x slower than a simple full sum in Python.
- BLOCK direct Phase-C performance backend until the bound problem is resolved.
- Phase D remains NOT TESTED; E6/E7 are not achieved.

Required wording:

> Phase B, E1: the certificate and exact fallback were correct on the committed synthetic corpus. The current global-range form did not provide a useful general skip rate; real-model and target-hardware performance remain unverified.

## Primary unresolved bottleneck

Derive causal, sound, checkpoint-generated bounds on omitted decision-relevant tile contributions that are tight enough to approach about 1.185% average evaluated target weights while charging metadata, selector, and fallback.

## Verified

- Phase D cannot run in the current environment.
- replay/index work is auxiliary;
- raw prefix memoization failed held-out routing;
- CPTC-v1 correctness/fallback passed Phase B;
- CPTC-v1 broad skip performance is insufficient.

## Refuted

- static compression alone closes target quality/traffic;
- raw prefix enumeration generalizes;
- metadata size equals traffic;
- one host probe proves latency failure;
- current global-range CPTC-v1 is a plausible primary executor;
- synthetic/small-model results prove 405B success.

## Unverified

- real Transformer tile distributions permit useful certification;
- oracle-tight or deployable stratified bounds close early;
- model-wide nonlinear propagation can be certified;
- savings persist with model size;
- any architecture reaches E6/E7.

## Current frontier

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`, defined in `NEXT_EXPERIMENT.md`.

First use held-out current-token states from available unmodified small checkpoints to compare global, non-deployable oracle-tight, and deployable stratified bounds. Offline audit remains below E2. If oracle-tight bounds still need high tile fractions, reject range-only CPTC rather than tune it.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
```

## Next-session reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. EXP-047 document and `results/exp_047/summary.json`
9. PR #56 and workflow `30792813542`
