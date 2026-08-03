# Reproducibility

## Objective

An independent researcher or new session must determine exactly what ran, what did not run, and how every result was produced from repository state and pinned external checkpoints.

Every summary separates `MEASURED / DERIVED / PROJECTED / UNVERIFIED`. Missing target-hardware fields remain `NOT TESTED`.

## Required layout

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/{README.md,config.json,run_current_env.sh,reproduce.sh,future_gpu_run.sh}
results/exp_xxx/{raw,processed,summary.json,logs,artifacts,checksums.sha256}
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

## Frozen authorities

```text
EXP-047  results/exp_047/summary.json   workflow 30793232558
EXP-047R results/exp_047r/summary.json  workflow 30795946233
EXP-048  results/exp_048/summary.json   workflow 30798936320
EXP-049  results/exp_049/summary.json   workflow 30803672059
```

## EXP-050 frozen authority

```text
results/exp_050/summary.json
workflow 30806015309
source head SHA 1388c780abea11067c66cd666ed0a313ec2f682c
workflow merge SHA 6bdd0a20334e394ec5252a6c0e676c1f62b608d0
artifact ID 8852817664
artifact name exp-050-candidate-30806015309
artifact size 34225 bytes
artifact ZIP SHA-256 a32ffe8dbfc201c6d70ca8dac660164d8400691ad4d8fe3593d688e7754f6159
phase A/B/C-observation
evidence E1
```

Scientific decision:

```text
REJECT_TARGET_INDEPENDENT_EXTERNAL_DRAFT_AS_UNIVERSAL_CORE
```

The tested TinyStories fixed draft pool is also rejected as a restricted practical core.

## EXP-050 pinned state

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact files are in `results/exp_050/raw/checkpoint_manifest.json`.

## EXP-050 committed evidence

```text
results/exp_050/summary.json
results/exp_050/raw/artifact_provenance.json
results/exp_050/raw/workflow_summary.json
results/exp_050/raw/pair_rows.jsonl
results/exp_050/raw/case_rows.jsonl
results/exp_050/raw/checkpoint_manifest.json
results/exp_050/raw/generation_records.json
results/exp_050/raw/universal_counterexample.json
results/exp_050/raw/E3_oracle_rows.json
results/exp_050/processed/aggregate.json
results/exp_050/logs/run.log
results/exp_050/artifacts/contract.txt
results/exp_050/artifacts/environment.json
results/exp_050/artifacts/workflow_checksums.sha256
results/exp_050/checksums.sha256
```

Original workflow hashes:

```text
summary.json
  64dadbe5eb69f1cc048bab0f3ab0939ace22952d8027710a1469d8d8ec2e2935
raw/pair_rows.jsonl
  d2c42fd1dae4f6e7863575bd0e08f839d8ce6147e5247f50617882257d90e116
raw/case_rows.jsonl
  4a72391862ed02f13e131112b2dece7951214b7fe3768b0f9df5aee3484d6a30
raw/checkpoint_manifest.json
  32f11e310c5dadc534975563551f05cea35510c90baeea4e040f767fac7c08f4
raw/universal_counterexample.json
  c6a6acc875ff2511ab76484084e753b6ae0d25469a2b3c43f6dbf3977624792e
raw/generation_records.json
  df84c4f908aa22660096883bf5e797dd5605198c57ea24b857f4d2f102ae4c22
raw/E3_oracle_rows.json
  4e4949619a8f03168ab94dc54826bde1580ce234198f0d8bdb76831aee6e8fae
processed/aggregate.json
  e0ec27876b2bfbf66e90e8628b8b9d3cb3636747955a2983ac4e8db9cdb59968
```

The provenance-enriched authoritative summary differs intentionally from `raw/workflow_summary.json`.

Verify:

```bash
cd results/exp_050
sha256sum -c checksums.sha256
```

## Reproduce EXP-050

```bash
git checkout research/exp-050-external-draft-advice
python -m pytest -q tests/exp_050
python scripts/run_validation.py
bash experiments/exp_050/reproduce.sh
```

The reproduction output defaults to `results/exp_050_reproduction` and must not overwrite frozen evidence.

Expected invariants:

- pinned revisions and prompt hashes;
- 18 target/prompt cases, 36 target/draft/prompt pairs, 108 K rows;
- zero exclusions, exact mismatch, and target-future use;
- one 256-token cached generation per model/prompt;
- one 256-token target verification per pair/prompt;
- K=64/128/256 rows derived causally from the same pass;
- universal counterexample matching prefix zero;
- exact-reference favorable selector label;
- decision logic.

CPU timing, RSS, and cache paths may vary.

## Independent validation

EXP-050 passed:

- 9 accounting/counterexample tests;
- self-draft and undercharged-forward failure tests;
- 507-token dynamic requirement test;
- exact prefix/correction equality;
- reference-selector ordering tests;
- revision/file manifests;
- exact target future oracle alignment;
- raw logs and checksums.

This establishes implementation scope, not useful proposal quality.

## Workflow rule

After evidence freezing, `.github/workflows/exp_050_gate.yml` becomes manual-only and writes isolated reproduction output. The path-triggered one-shot freeze workflow installs the authoritative artifact only when its own file is introduced/modified.

## EXP-051 evidence requirements

EXP-051 must freeze:

- every target token state and every intermediate block-depth token/margin;
- exact hidden-state alignment contract;
- first-match and suffix-stable depths;
- transient flips after first match;
- actual block, final norm, LM-head, and embedding-row logical bytes;
- favorable oracle byte/depth fractions;
- fixed-depth accuracy by target/family;
- late-decision residual-chain adversary;
- exact-reference depth-selector label;
- excluded states and context limits;
- raw CPU/RSS/environment data.

A suffix-stable oracle depth is non-deployable and below E2. Actual operation replacement requires a sound causal tail certificate and real omitted-layer execution counts.

## Infrastructure failure procedure

1. classify infrastructure versus science;
2. preserve logs/artifacts;
3. do not interpret dependency/download/timeout/storage errors as hypothesis evidence;
4. update durable state;
5. rerun only after correction.
