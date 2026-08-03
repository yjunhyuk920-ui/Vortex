# Reproducibility

## Objective

An independent researcher or new session must determine exactly what ran, what did not run, and how every result was produced from repository state and pinned external checkpoints.

Every summary separates `MEASURED / DERIVED / PROJECTED / UNVERIFIED`. Unavailable fields remain `NOT TESTED`.

## Required experiment layout

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/{README.md,config.json,run_current_env.sh,reproduce.sh,future_gpu_run.sh}
results/exp_xxx/{raw,processed,summary.json,logs,artifacts,checksums.sha256}
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

## Earlier frozen authorities

```text
EXP-047  results/exp_047/summary.json   workflow 30793232558
EXP-047R results/exp_047r/summary.json  workflow 30795946233
EXP-048  results/exp_048/summary.json   workflow 30798936320
```

EXP-048 artifact authority:

```text
artifact 8850040445
artifact ZIP SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
```

## EXP-049 frozen authority

```text
results/exp_049/summary.json
workflow 30803672059
source head SHA 91d0caa86d784c663bc520d36d9b512f0cc526e9
workflow merge SHA 173dd3477e2a6f5ecb0d55b58375ec18dfe774dd
artifact ID 8851957250
artifact name exp-049-candidate-30803672059
artifact size 105493 bytes
artifact ZIP SHA-256 4cd6c8c4afb833562438a97f052d45d331f3691362472fb08e594bd0c5585b9e
phase A/B/C-observation
evidence E1
```

Scientific decision:

```text
REJECT_TARGET_ONLY_CONTINUOUS_FIXED_POINT_CORE_RETAIN_SOLVER_AND_VERIFIER_AUXILIARY
```

## EXP-049 pinned external state

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact file hashes are in `results/exp_049/raw/checkpoint_manifest.json`.

## EXP-049 committed evidence

```text
results/exp_049/summary.json
results/exp_049/raw/artifact_provenance.json
results/exp_049/raw/workflow_summary.json
results/exp_049/raw/checkpoint_manifest.json
results/exp_049/raw/triangular_audit.json
results/exp_049/raw/cases.jsonl
results/exp_049/processed/aggregate.json
results/exp_049/logs/run.log
results/exp_049/artifacts/contract.txt
results/exp_049/artifacts/environment.json
results/exp_049/artifacts/workflow_checksums.sha256
results/exp_049/checksums.sha256
```

Raw `cases.jsonl` is 1,009,278 bytes and contains 18 case rows with 1,458 recorded fixed solver trajectories. It was copied byte-identically from the successful workflow artifact by the one-shot evidence-freeze workflow.

Internal authoritative artifact hashes:

```text
raw/cases.jsonl
  0b9c0ca21ae5211fe19f6ce56dc40c474597880e4c9e83575249938c44d22f72
raw/checkpoint_manifest.json
  62a2713d5033fa3b00e03a49cf22c93e3e957280233198070f97510d0ce556f8
raw/triangular_audit.json
  59979799af69f14a0e604addaf8256280e9f9fce48da623c698c2690bcb1d6a8
raw/workflow_summary.json
  156aac891836e69d00d5eb77fd2208f02096512f881216238fc05185c64de40f
processed/aggregate.json
  5834ff4cf30d55c98126b47dce1077ff3e192f6ab54dadbe2d9f99a029fbf458
```

The current authoritative `summary.json` adds artifact provenance and raw-evidence pointers, so its checksum intentionally differs from the original workflow summary preserved under `raw/workflow_summary.json`.

Verify committed evidence:

```bash
cd results/exp_049
sha256sum -c checksums.sha256
```

## Reproduce EXP-049

```bash
git checkout research/exp-049-anderson-continuous-fixed-point
python -m pytest -q tests/exp_049
python scripts/run_validation.py
bash experiments/exp_049/reproduce.sh
```

The reproduction wrapper writes to `results/exp_049_reproduction` unless overridden and must not overwrite frozen `results/exp_049`.

Expected invariant fields:

- exact checkpoint/tokenizer revisions and prompt hashes;
- 18 cases and no silent context exclusions;
- fixed variants, blocks 64/128/256, pass checkpoints 1/2/4;
- zero selected exact mismatch and target-future use;
- triangular transcript/barrier booleans;
- deterministic proposal tokens under pinned CPU environment;
- decision logic.

CPU elapsed time and RSS may vary by runner.

## EXP-049 independent validation

Required and passed within E1 scope:

- 9 model-independent Picard/Anderson/lower-bound tests;
- linear-contraction Anderson positive control;
- shape/NaN/Inf and ill-conditioned solve fault tests;
- coefficient clipping/regularization/condition limit;
- fail-closed Picard fallback;
- exact future-state S3 alignment;
- exact hardening and retained verifier integration;
- no target future tokens in S1/S2;
- hidden triangular one-position-per-round construction;
- hidden-suffix transcript indistinguishability;
- pinned checkpoint manifests;
- raw logs and checksums.

These checks establish implementation and adversarial scope, not useful target performance.

## Workflow rule

After authoritative evidence is frozen, the scientific Gate workflow becomes manual `workflow_dispatch` only. Reproduction writes isolated candidate evidence and never mutates the repository.

The one-shot `freeze_exp_049_evidence.yml` is path-triggered only by modification of itself. Its successful bot commit installed the authoritative artifact. It does not rerun on result/document updates.

## Determinism

EXP-049 seed: `202608033`; thread count, revisions, prompts, block sizes, pass checkpoints, top-k, damping, initializations, Anderson histories, regularization, clipping, condition limit, and adversarial chains are frozen.

## EXP-050 reproduction boundary

EXP-050 must additionally freeze:

- every target/draft pair;
- draft greedy tokens and forward count;
- draft/target parameter counts and bytes;
- exact-prefix distribution for each block size;
- target-independent first-token counterexample;
- exact-reference pool-selection label;
- actual small-model and projected 4B/405B traffic equations;
- target/draft KV and RSS accounting where measurable.

A reference-selected best draft is a favorable falsification oracle, not a deployable selector.

## Infrastructure failure procedure

1. classify infrastructure versus science;
2. preserve logs/artifacts;
3. do not interpret dependency, download, timeout, or storage errors as hypothesis evidence;
4. update durable state;
5. rerun only after correction.
