# Reproducibility

## Objective

An independent researcher or new session must determine what ran, what did not run, and how every result was produced from repository state and pinned external checkpoints.

## Provenance

Every summary separates MEASURED, DERIVED, PROJECTED, and UNVERIFIED. Unavailable fields are null or `NOT TESTED`.

## Required experiment layout

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/{README.md,config.json,run_current_env.sh,reproduce.sh,future_gpu_run.sh}
results/exp_xxx/{raw,processed,summary.json,logs,artifacts,checksums.sha256}
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

## EXP-047 frozen authority

```text
results/exp_047/summary.json
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
phase A/B, evidence E1
```

## EXP-047R frozen authority

```text
results/exp_047r/summary.json
workflow 30795946233
source head SHA 0beb068e9679c9f4d51d1b210b0eee7fbc325214
workflow merge SHA 213e69a54c4d2b5c2d4102f8651cab847ade312f
artifact ID 8848886335
artifact name exp-047r-candidate-30795946233
artifact ZIP SHA-256 6c9a4fdca80d29964eca02d16f8b36f5ca8e211653f6fb9ddfe548a729c6e12d
phase A/B/C-observation, evidence E1
```

Scientific decision:

```text
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
```

## EXP-047R pinned external state

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact downloaded file hashes are in `results/exp_047r/raw/checkpoint_manifest.json`. The config now pins these immutable revisions rather than moving `main`.

## EXP-047R committed evidence

```text
results/exp_047r/summary.json
results/exp_047r/raw/artifact_provenance.json
results/exp_047r/raw/checkpoint_manifest.json
results/exp_047r/raw/cases_part_01.jsonl
results/exp_047r/raw/cases_part_02.jsonl
results/exp_047r/raw/cases_part_03.jsonl
results/exp_047r/processed/aggregate.json
results/exp_047r/logs/run.log
results/exp_047r/artifacts/contract.txt
results/exp_047r/artifacts/pip_freeze.txt
results/exp_047r/artifacts/workflow_checksums.sha256
results/exp_047r/checksums.sha256
```

The original 18-row `cases.jsonl` is reconstructed by concatenating case parts in numeric order without modification. The exact original workflow archive is protected by the artifact ZIP SHA-256 above. `workflow_checksums.sha256` preserves the workflow's internal file checksums.

## Reproduce EXP-047R

```bash
git checkout research/exp-047r-oracle-stratified-audit
python -m pytest -q tests/exp_047r
python scripts/run_validation.py
bash experiments/exp_047r/reproduce.sh
```

The reproduction wrapper writes to `results/exp_047r_reproduction` unless overridden. It must not overwrite frozen `results/exp_047r`.

Verify durable byte-identical files:

```bash
cd results/exp_047r
sha256sum -c checksums.sha256
```

The root checksum file covers files committed byte-for-byte from the artifact or exact case reconstruction parts. Derived concise summary/manifest formatting is linked to the exact archive digest and original internal checksums.

## Workflow rule

After authoritative evidence is frozen, the experiment workflow is manual `workflow_dispatch` only. Reproduction uploads isolated candidate evidence and does not mutate the repository. Documentation changes must not regenerate or replace completed measurements.

## Determinism

EXP-047R seed: `202608031`; prompt texts/hashes, checkpoint/tokenizer revisions, tile size, stratum count, thread count, and dependency versions are frozen. Sampling order is deterministic. CPU timing remains environment-dependent.

## Independent validation

EXP-047R requires:

- exact LM-head reconstruction from the returned final hidden state;
- exact pair-margin reconstruction from tile contributions;
- bound validation against all materialized contributions;
- property/adversarial/fault-injection tests;
- deterministic replay;
- zero wrong accepts in the committed corpus;
- fail-closed behavior on malformed bounds or reconstruction mismatch.

These passed within E1 scope. This does not establish useful savings or real operation replacement.

## External checkpoints

Real-model work pins model ID, exact revision, file manifest/hashes, tokenizer revision, download command, dependency environment, prompt hash, and decoding parameters. Moving `main` is never authoritative.

## Infrastructure failure procedure

1. classify infrastructure versus science;
2. preserve logs/artifacts;
3. do not interpret missing checkpoints, timeouts, or dependency errors as hypothesis evidence;
4. update durable state;
5. rerun only after correction.

Prior corrected infrastructure failures `30791055142` and `30791192434` remain non-scientific evidence.
