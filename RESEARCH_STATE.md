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

MEASURED capability: GitHub repository, GitHub Actions CPU, Python 3.10/3.12, and small downloadable checkpoints when time/storage permit.

Unavailable and NOT TESTED: target 8 GiB GPU, 405B checkpoint execution, target CUDA/PCIe/SSD/power, TTFT, tokens/second, and peak VRAM.

## Validation system

- Phase A theory/structure;
- Phase B synthetic/reference;
- Phase C small real-model falsification;
- Phase D actual target hardware, currently NOT TESTED.

Evidence E0–E7 and provenance MEASURED/DERIVED/PROJECTED/UNVERIFIED are mandatory.

## Existing component classification

Auxiliary accepted:

- exact/checksummed mmap pointer VM;
- bounded TinyLlama compiler with 72/72 checked token replay in its finite grammar;
- exact future-suffix DAG with 64->38 bounded record compression.

Rejected as core:

- raw prefix scaling and future-aware routing as unseen-prompt runtime;
- prior static compression, deterministic residual, recurrent program, and repair families listed in `FAILED_APPROACHES.md`.

## EXP-047 authoritative evidence

The machine-readable source of truth is:

```text
results/exp_047/summary.json
results/exp_047/raw/cases.jsonl
results/exp_047/checksums.sha256
```

Frozen committed summary currently records:

```text
PR: #56
workflow: 30793232558
source SHA: 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
phase: A/B
evidence: E1
Phase D: NOT TESTED
```

### MEASURED correctness

```text
10 tests passed
525 cases
certified 4
fallback 521 = 99.238%
wrong accepts 0
fallback/reference mismatches 0
independent-bound mismatches 0
adversarial exact fallback 15/15
future generated tokens used false
```

Largest positive cancellation control:

```text
1,024 tiles
107 sampled
10.449% evaluated before certificate
```

Broad scaling:

```text
N=64/128/256: 100% tiles evaluated
N=512: mean evaluated 98.519%
N=1024: mean evaluated 98.294%
Python optimized/reference mean time: about 8.6–9.1x
```

### PROJECTED target gap

```text
405B Q4 full stream: 188.593 GiB
4B Q4 full stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required evaluated fraction before overhead: 1.185%
positive-control fraction / target fraction: 8.817x
```

These are parameter-count projections, not target-hardware measurements.

## Scientific decision

- ACCEPT E1 correctness primitive: causal randomized sampling, alpha-spending Serfling calculation, independent interval check, deterministic replay, error rejection, and exact fallback passed.
- REVISE architecture: one global range certified only 4/525 cases, evaluated about 98% of tiles overall, and was much slower than full Python summation.
- BLOCK direct Phase-C performance backend until the bound problem is resolved.
- Phase D remains NOT TESTED; E6/E7 are not achieved.

Required wording:

> Phase B, E1: the certificate and exact fallback were correct on the committed synthetic corpus. The current global-range form did not provide a useful general skip rate; real-model and target-hardware performance remain unverified.

## Primary unresolved bottleneck

Derive causal, sound, checkpoint-generated bounds on omitted decision-relevant tile contributions tight enough to approach about 1.185% average evaluated target weights while charging metadata, selector, and fallback.

## Current frontier

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`, defined in `NEXT_EXPERIMENT.md`.

Use held-out current-token states from available unmodified small checkpoints to compare global, non-deployable oracle-tight, and deployable stratified bounds. Offline analysis remains below E2. If oracle-tight bounds still need high tile fractions, reject range-only CPTC rather than tune it.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
cd results/exp_047 && sha256sum -c checksums.sha256
```

## Next-session reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. EXP-047 document and frozen result summary
9. PR #56
