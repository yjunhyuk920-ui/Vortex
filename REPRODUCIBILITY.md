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

Exact downloaded file hashes are in `results/exp_047r/raw/checkpoint_manifest.json`. The config pins immutable revisions rather than moving `main`.

## EXP-047R committed evidence and reproduction

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

The original 18-row `cases.jsonl` is reconstructed by concatenating case parts in numeric order without modification.

```bash
git checkout research/exp-047r-oracle-stratified-audit
python -m pytest -q tests/exp_047r
python scripts/run_validation.py
bash experiments/exp_047r/reproduce.sh
cd results/exp_047r && sha256sum -c checksums.sha256
```

## EXP-048 frozen authority

```text
results/exp_048/summary.json
workflow 30798936320
source head SHA 484a1f0f313d88733d2f7210f2a24d3904bf1373
workflow merge SHA d60e392d66d694fc020f2cfe2435e47e5f5a22ca
artifact ID 8850040445
artifact name exp-048-candidate-30798936320
artifact size 17689 bytes
artifact ZIP SHA-256 67c1e6d8965f7535020ecd4c02bb8a2af1156a234564f3cdf74d10c882fd7eb9
phase A/B/C-observation, evidence E1
```

Scientific decision:

```text
REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER
```

## EXP-048 pinned external state

```text
EleutherAI/gpt-neo-125M tokenizer @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Exact model/tokenizer file hashes are in `results/exp_048/raw/checkpoint_manifest.json`. The workflow downloaded only pinned model/config and tokenizer files required by the run.

## EXP-048 committed evidence

```text
results/exp_048/summary.json
results/exp_048/raw/artifact_provenance.json
results/exp_048/raw/checkpoint_manifest.json
results/exp_048/raw/cases.jsonl.gz.b64
results/exp_048/processed/aggregate.json
results/exp_048/logs/run.log
results/exp_048/artifacts/contract.txt
results/exp_048/artifacts/pip_freeze.txt
results/exp_048/artifacts/workflow_checksums.sha256
results/exp_048/checksums.sha256
```

The exact original workflow artifact archive is protected by the ZIP digest above. `artifacts/workflow_checksums.sha256` preserves the internal hashes emitted by the authoritative run:

```text
raw/cases.jsonl SHA-256 b70d56f3e13ab1f39dd8947be468e663d6b5691fb20236b990f20a343bcbe4d2
raw/checkpoint_manifest.json SHA-256 8dd7be9a710a7ddd805e0bfebcbbd3f9b8e32410093ae72693722d805cf103b7
summary.json SHA-256 46d38301f9fecc21b31e0f8e987dc6a25d42451dc516519ca73cd2011866ea54
```

The 18-row raw JSONL is committed losslessly as deterministic gzip plus base64 text. Restore and verify it:

```bash
base64 -d results/exp_048/raw/cases.jsonl.gz.b64 | gunzip > /tmp/exp_048_cases.jsonl
sha256sum /tmp/exp_048_cases.jsonl
# b70d56f3e13ab1f39dd8947be468e663d6b5691fb20236b990f20a343bcbe4d2
```

Verify durable byte-identical files:

```bash
cd results/exp_048
sha256sum -c checksums.sha256
```

## Reproduce EXP-048

```bash
git checkout research/exp-048-causal-block-amortization
python -m pytest -q tests/exp_048
python scripts/run_validation.py
bash experiments/exp_048/reproduce.sh
```

The reproduction wrapper writes to `results/exp_048_reproduction` unless overridden and must not overwrite frozen `results/exp_048`.

Expected scientific classification:

```text
B1 perfect future oracle: exact non-deployable upper bound
B2 hard Jacobi: rejected as core
B3 partial-layer self-draft: rejected as core
exact block verifier: retained auxiliary
```

CPU timing may vary across machines. Exact tokens, pass counts, causal/future labels, fixed model revisions, prompt hashes, and decision thresholds are the primary reproduction fields.

## Workflow rule

After authoritative evidence is frozen, an experiment workflow becomes manual `workflow_dispatch` only. Reproduction uploads isolated candidate evidence and does not mutate the repository. Documentation changes must not regenerate or replace completed measurements.

## Determinism

EXP-047R seed: `202608031`.

EXP-048 seed: `202608032`; exact prompt texts/families, revisions, block sizes, Jacobi iterations/fill token, partial-layer counts, thread count, and dependency versions are frozen.

B0 greedy tokens and B1 proposal are deterministic under the pinned CPU environment. B2/B3 use greedy argmax and fixed control parameters. CPU elapsed time remains environment-dependent.

## Independent validation

EXP-048 requires:

- model-independent proof tests for matching-prefix commit and first-mismatch correction;
- no commitment of target predictions after first mismatch;
- exact B0/B1/B2/B3 output agreement wherever committed;
- explicit future-information label on B1 and zero future use in B3;
- every B2 target iteration charged;
- every B3 draft layer/head/embedding-equivalent stream, target pass, rejected position, and correction charged;
- exact revisions and file manifests;
- fail-closed malformed proposal and accounting validation.

These passed within E1 scope. They do not establish a useful deployable proposal source or physical block weight reuse.

## External checkpoints

Real-model work pins model ID, exact revision, file manifest/hashes, tokenizer revision, download command, dependency environment, prompt hash, and decoding parameters. Moving `main` is never authoritative.

## Infrastructure failure procedure

1. classify infrastructure versus science;
2. preserve logs/artifacts;
3. do not interpret missing checkpoints, timeouts, or dependency errors as hypothesis evidence;
4. update durable state;
5. rerun only after correction.

Prior corrected infrastructure failures `30791055142` and `30791192434` remain non-scientific evidence.
