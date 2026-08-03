# VORTEX Research State

Last updated: 2026-08-03 15:15 Asia/Seoul

## Fixed final objective

Execute an arbitrary publicly released, unmodified Hugging Face dense Transformer by replacing only the runtime:

- flagship: 405B-class dense model;
- peak GPU VRAM: 8 GiB;
- no retraining, distillation, fine-tuning, LoRA, or model-specific user-authored adapter;
- original-model ability and declared output contract preserved;
- final user-perceived performance near a native 4B-class model on the same target machine;
- independently reproducible evidence.

The target is not reduced by current hardware limitations or failed experiments.

## Current execution environment

MEASURED environment capability:

- GitHub repository and GitHub Actions CPU runners;
- no dedicated 8 GiB VRAM target GPU;
- no local storage or access suitable for a 405B checkpoint;
- no real 405B execution;
- no target CUDA, PCIe, SSD-bandwidth, power, or tokens/second profiling;
- small downloadable Hugging Face checkpoints may be used for Phase C falsification only.

## Validation phases

- **Phase A — theory and structure:** equations, correctness contracts, causal conditions, lower bounds, fallback safety, and 405B resource models.
- **Phase B — synthetic/reference:** independent reference implementations, randomized/property tests, fault injection, deterministic replay, and scaling trends.
- **Phase C — small real-model falsification:** unmodified small checkpoints, held-out prompts, real forward counts, logits/tokens, fallback, CPU time, and RAM.
- **Phase D — target hardware:** real 8 GiB GPU plus 70B/405B checkpoints and hardware profilers. Current status: **NOT TESTED**.

## Evidence scale

- E0: idea or equation.
- E1: synthetic/reference validation.
- E2: real small-model operation replacement.
- E3: held-out generalization with measured causal coverage.
- E4: measured improvement on currently accessible representative hardware.
- E5: medium/large model scaling validation.
- E6: target model runs under 8 GiB VRAM.
- E7: 405B reaches the declared 4B-class user-perceived target.

Current project ceiling reached: **E2 for bounded auxiliary decision-index compilation; E1/E2 for several component experiments. E6/E7 are NOT TESTED.**

## Strongest actual achievements

MEASURED:

- A portable mmap exact pointer VM with checksums, atomic build, deterministic replay, and bounded cache.
- A bounded unmodified TinyLlama decision-index compiler replayed 72/72 checked tokens for its declared finite grammar.
- An exact finite-horizon future-suffix DAG reduced 64 raw records to 38 nodes on the compiled traces.

DERIVED:

- Constructed end-to-end Llama-style decision families require more than 8 GiB of complete resident decision metadata.
- An explicit pointer-table family can force near-one serial host probe per token while requiring only a few logical bytes per probe.

These achievements are **auxiliary**. They do not solve unseen-prompt Transformer operation skipping.

## Primary unresolved bottleneck

Find a causal, verifiable execution principle that avoids reading or applying most 405B weights on an unseen prompt while preserving the original model's decision contract and providing a safe exact fallback.

Every core experiment must answer:

1. which original operations are skipped or replaced;
2. how the skip is decided without future tokens;
3. decision cost;
4. wrong-skip detection;
5. fallback path;
6. worst-case correctness;
7. scaling behavior;
8. reason a next token can be decided without reading all weights;
9. RAM/SSD/VRAM movement;
10. 405B minimum bandwidth and compute;
11. distance from the 4B target;
12. strongest falsification test.

## Current core hypothesis

**EXP-047 — Causal Probabilistic Tile Certificate (CPTC).**

Partition a linear operator into input-dimension tiles. Evaluate tiles in a causal randomized order. Use finite-population/martingale confidence bounds to certify that omitted tile contributions cannot change a declared decision. If the certificate fails, evaluate all remaining tiles and return the exact baseline result.

The first Gate is Phase A/B only. It must determine whether statistically exploiting cancellation can beat previously failed worst-case residual bounds without ever silently returning an uncertified result.

## Verified claims

- Current environment cannot perform Phase D measurements.
- Existing mmap/DAG mechanisms are auxiliary replay/index structures, not a universal runtime.
- Raw exact-prefix memoization showed linear node growth and zero held-out start coverage on the measured TinyLlama grammar.
- Exact future-suffix sharing exists on bounded traces but requires a separate causal start mechanism.

## Refuted claims

- Static low-rank/dictionary/factorized representations alone preserve useful autonomous model behavior under the target traffic envelope.
- Raw prefix enumeration provides broad unseen-prompt generalization.
- One small host probe per token alone proves the target impossible.
- Synthetic or TinyLlama success constitutes 405B/8 GiB performance evidence.

## Unverified claims

- CPTC can certify useful tile omission rates in real Transformer layers.
- Probabilistic certification overhead remains below the savings.
- Small-model certified skip rates persist or improve with model size.
- Any current architecture can run a real 405B model in 8 GiB VRAM.
- 4B-class TTFT or tokens/second can be achieved.

## Active repository state

- branch: `research/governance-exp047-cptc`
- latest base main commit: `3f3ee348e8a72070b3e43cf7af56c078fc4d83c7`
- active experiment: `EXP-047`
- Phase D: `NOT TESTED`

This file must be updated again before the branch is merged, with the actual head commit, PR, workflows, results, and exact reproduction command.

## Reproduction entry points

```bash
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/run_current_env.sh
```

## Next session mandatory reading

1. `AGENTS.md`
2. `RESEARCH_STATE.md`
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `docs/research/EXPERIMENT_047_CAUSAL_PROBABILISTIC_TILE_CERTIFICATE.md`
9. latest workflow logs and `results/exp_047/summary.json`
