# VORTEX Research State

Last updated: 2026-08-03 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- flagship: real 405B-class dense model;
- peak GPU VRAM <=8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or user-authored model-specific adapter;
- original-model ability and declared output contract preserved;
- p50 warm time/token <=1.2x native 4B Q4 on the same target machine;
- independent reproduction from pinned code and checkpoint hashes.

The target has not been reduced.

## Current environment

MEASURED capability:

- GitHub repository and GitHub Actions CPU runners;
- Python 3.10/3.12 CI;
- small downloadable Hugging Face checkpoints when storage/time permit.

Unavailable:

- target 8 GiB GPU;
- 405B checkpoint storage/download/execution;
- target CUDA, PCIe, SSD, power, TTFT, tokens/second, and peak-VRAM profiling.

Phase D status: **NOT TESTED**.

## Validation and evidence

- Phase A: theory/structure.
- Phase B: synthetic/reference.
- Phase C: small real-model falsification.
- Phase D: actual target hardware, currently NOT TESTED.

Evidence scale: E0 idea, E1 synthetic/reference, E2 real small-model operation replacement, E3 held-out causal generalization, E4 accessible representative-hardware improvement, E5 medium/large scaling, E6 target model under 8 GiB, E7 405B at declared 4B-class performance.

Every result separates MEASURED, DERIVED, PROJECTED, and UNVERIFIED.

## Strongest actual achievements

MEASURED:

- exact/checksummed mmap pointer VM;
- bounded unmodified TinyLlama decision-index compiler with 72/72 checked token replay inside its declared grammar;
- exact future-suffix DAG compressed 64 bounded trace records to 38 nodes;
- EXP-047 Phase-B certificate passed 10 tests and 525 generated cases with zero wrong accepts, zero fallback mismatches, and zero independent-bound mismatches.

DERIVED:

- constructed Llama-style complete decision metadata can exceed 8 GiB;
- explicit pointer families can force serial host probes without forcing large logical bytes/token;
- simple Q4 traffic comparison requires a 405B evaluated-weight fraction near 1.185% before selector/fallback overhead to fit 1.2x a 4B Q4 stream.

The mmap/index/DAG achievements remain auxiliary. They do not solve unseen-prompt Transformer operation skipping.

## EXP-047 authoritative result

Source branch: `research/governance-exp047-cptc`

PR: `#56`

Authoritative workflow: `30791851508`

Authoritative source head recorded by the run: `d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b`

Raw evidence is committed under `results/exp_047/`.

### MEASURED

```text
phase: A/B
evidence: E1
cases: 525
certified: 4
fallback: 521 = 99.238%
wrong certified accepts: 0
fallback/reference mismatches: 0
independent-bound mismatches: 0
adversarial cases: 15/15 exact fallback
Phase D: NOT TESTED
```

Largest positive cancellation control:

```text
population tiles: 1,024
sampled before certificate: 107
sample fraction: 10.449%
decision agreement: pass
```

General synthetic scaling:

```text
N=64/128/256: 100% fallback, 100% tiles evaluated
N=512: mean evaluated fraction 98.519%
N=1024: mean evaluated fraction 98.294%
```

Python optimized-path time was roughly 8.8–9.1x the simple full-sum reference in the measured buckets. This is implementation/CPU evidence only, not accelerator projection.

### PROJECTED

```text
405B Q4 full stream: 188.593 GiB
4B Q4 full stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required target fraction before overhead: 1.185%
positive-control fraction / target fraction: 8.817x
```

### Scientific decision

- **ACCEPT E1 correctness primitive:** the alpha-spending Serfling implementation, independent check, causal sampling, and exact fallback passed the committed Phase-B Gate.
- **REVISE core architecture:** global-range CPTC-v1 certified only 4/525 cases and evaluated about 98% of tiles overall. It is not approved for Phase-C performance promotion.
- **NOT TESTED:** sound bounds derived from real checkpoint weights/activations, real operation replacement, held-out prompts, 70B/405B trend, GPU selector cost, 8 GiB execution, PCIe/SSD, TTFT, or tokens/second.

Required wording:

> Phase B, E1: the finite-population certificate and exact fallback were correct on the committed synthetic corpus. The current global-range form did not provide a useful general skip rate; real-model and target-hardware performance remain unverified.

## Primary unresolved bottleneck

Find a causal and verifiable way to bound decision-relevant omitted weight-tile contributions tightly enough to approach an average evaluated fraction near 1.185%, while charging selector metadata and exact fallback.

## Verified claims

- Current environment cannot execute Phase D.
- Existing replay/index work is auxiliary.
- Raw prefix memoization did not generalize on measured TinyLlama grammar.
- CPTC-v1 correctness/fallback implementation passed Phase B.
- CPTC-v1 broad synthetic skip rate is currently insufficient.

## Refuted claims

- Static compression/dictionary families alone close quality and traffic.
- Raw prefix enumeration is a broad unseen-prompt runtime.
- Total metadata equals per-token traffic.
- One host probe/token alone proves latency failure.
- The global `[-1,1]` range Serfling certificate is presently a useful primary executor.
- Tiny/synthetic evidence proves 405B success.

## Unverified claims

- Real Transformer tile distributions permit useful causal certificates.
- Checkpoint-derived stratified/tile-norm bounds are tight enough.
- A probabilistic union budget remains usable model-wide.
- Skip fractions persist or improve with model size.
- Any current architecture reaches E6 or E7.

## Current branch state

- branch: `research/governance-exp047-cptc`
- PR: `#56` draft until final documentation and CI complete
- active research: EXP-047 revision Gate
- Phase D: NOT TESTED

## Reproduction

```bash
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/reproduce.sh
```

## Next-session mandatory reading

1. `AGENTS.md`
2. this file
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `docs/research/EXPERIMENT_047_CAUSAL_PROBABILISTIC_TILE_CERTIFICATE.md`
9. `results/exp_047/summary.json`
10. PR #56 and workflow `30791851508` logs
