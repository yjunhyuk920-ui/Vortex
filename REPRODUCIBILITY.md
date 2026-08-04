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

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## EXP-052 frozen authority

Authority: `results/exp_052/summary.json`; workflow `30811429049`; source head `d4c2328027a5377b997e9ee1d8df0f55190fb652`; artifact `8854946309`; ZIP SHA-256 `1beb137e1ee14fe80ded0a3309c4ed297035d552a46bf901b2e4233ab95549ca`.

Verify with `cd results/exp_052 && sha256sum -c checksums.sha256`. Original workflow hashes are preserved in `results/exp_052/artifacts/workflow_checksums.sha256`; the original workflow summary is `results/exp_052/raw/workflow_summary.json`.

<!-- EXP-053-AUTHORITATIVE-FINAL -->
## EXP-053 frozen authority

Authority: `results/exp_053/summary.json`; workflow `30814648709`; source head `325cc694d4b2e88e34dba5ba8e980e3970c34c66`; workflow merge `4ecca6405f549fc9a05d7ad17cfe1d7c3a9c3398`; artifact `8856213147`; ZIP SHA-256 `eb7ecf8f284cc974d62e03bee767892666160abfae79a70bb32446f0dfe95178`.

Verify with `cd results/exp_053 && sha256sum -c checksums.sha256`. Original workflow hashes are preserved in `results/exp_053/artifacts/workflow_checksums.sha256`; original summary is `results/exp_053/raw/workflow_summary.json`; all 24 binary AIGs are under `results/exp_053/raw/circuits/`.

<!-- EXP-054-AUTHORITATIVE-FINAL -->
## EXP-054 frozen authority

Authority: `results/exp_054/summary.json`; workflow `30816333096`; source head `2c63da85050afcedad6a00698a6f8fddd3bc99d2`; artifact `8856906303`; ZIP SHA-256 `0dc642f306cea99ce01095758a5f49151092d530efb94d36985553e408596edf`.

Verify with `cd results/exp_054 && sha256sum -c checksums.sha256`. Original workflow hashes are preserved under `results/exp_054/artifacts/`; all completed binary diagrams are under `results/exp_054/raw/diagrams/`.

<!-- EXP-058-AUTHORITATIVE-FINAL -->
## EXP-058 authority

Workflow `30826618962`; source head `8ae03de4cc34317b5536aed42b9b8c22f98c88ea`; workflow merge `3730d6ce8ca89df347079c366a91bcad4d904a85`; artifact `8861905858` (29349 bytes); artifact ZIP SHA-256 `851582a616412e7e078b7c05ddb64883b972cec895847b6df3d7d75dc615bfae`; config SHA-256 `18356731d606c819da29807a98de600c8d4d515ff16b5d06c0b90613ee431906`. Reproduce with `experiments/exp_058/reproduce.sh` and verify `results/exp_058/checksums.sha256`.

<!-- EXP-059-AUTHORITATIVE-FINAL -->
## EXP-059 authority

Workflow `30840432745`; source head `cdae6160cd87b537e2f318c16430619736c7c9d9`; workflow merge `82979e393a87845c4c757ce5dfd3fadc4e701d92`; artifact `8866573958` (68652 bytes); ZIP SHA-256 `61d0c24ccacd310d7d0e7600cc926a882c74281827d524c4880c6715fad8800d`; config SHA-256 `3e318ff909597e8b9ceca9b39b2a02caacc1427ce2b34132baa6ab7456003e62`. Reproduce with `experiments/exp_059/reproduce.sh` and verify `results/exp_059/checksums.sha256`.

<!-- EXP-060-AUTHORITATIVE-FINAL -->
## EXP-060 authority

Workflow `30841671707`; source head `bf89d087343a4790202126c34562ca0344ebe452`; workflow merge `5f2af394180beaf3e5b5b8c7386d2becdf7eb8e7`; artifact `8867145590` (58039 bytes); ZIP SHA-256 `5e5255dbedd779b734876faa027cd2bf5e4a1b00ece7f28cbf35f428fb9a0b05`; config SHA-256 `82254fd1177bcce6b788199ed92bbc122d97f04783f0bc02d056c090ba043a29`. Reproduce with `experiments/exp_060/reproduce.sh` and verify `results/exp_060/checksums.sha256`.

<!-- EXP-061-AUTHORITATIVE-FINAL -->
## EXP-061 authority

Workflow `30843404056`; source head `15097a9b0323aa992679214173aaac0e7a98821c`; workflow merge `44c3d6691d78714dc975e46e19bb8fdfe97a22cf`; artifact `8867731496` (662994 bytes); ZIP SHA-256 `a01d31b012badd7d06087df576279b852db07813a0c7fb50d65c3a7283e9ca65`; config SHA-256 `b5635e3cd57dae39bc66c7939ef75ea7c79d6dab2a22d634c1441f0a9d930e82`. Reproduce with `experiments/exp_061/reproduce.sh` and verify `results/exp_061/checksums.sha256`.

<!-- EXP-062-AUTHORITATIVE-FINAL -->
## EXP-062 authority

Workflow `30844873182`; source head `c38baa187e41760ef07676326c6a14f08635acc3`; workflow merge `891868c186eb22869925ad20cba43ef32d371589`; artifact `8868287407` (523940 bytes); ZIP SHA-256 `497816dcca7e6b8c40e9222ed8511efa266fe2358aab847a93795d7c04637390`; config SHA-256 `c987fc4ab548d08036e7db534b473aa13addc50398cc3492c22222b0fb21d98f`. Reproduce with `experiments/exp_062/reproduce.sh` and verify `results/exp_062/checksums.sha256`.

<!-- EXP-063-AUTHORITATIVE-FINAL -->
## EXP-063 authority

Workflow `30846082964`; source `979bde3a23b76270f740740fbf511c7f90900a7c`; merge `488fa0e3785885bbcea25681aae55bb361fa0f84`; artifact `8868770832` (2371412 bytes); ZIP SHA-256 `b900a7019d8527d6f67d0eb412bb2fb7a0331188d84cd74444ca10762a105a14`; config SHA-256 `69ebb4868b3707bbdf42d07a9f7f75458c147eb9c43dfe7e92e93843a5ffc32b`. Reproduce with `experiments/exp_063/reproduce.sh`; the full group table is frozen losslessly as `raw/group_rows.jsonl.gz`; verify `results/exp_063/checksums.sha256`.
