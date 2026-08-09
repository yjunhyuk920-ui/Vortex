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

<!-- EXP-064-AUTHORITATIVE-FINAL -->
## EXP-064 authority

Workflow `30869720552`; source `a6371c39d85dc39669b98eac6125d9c3bbf4a5dc`; merge `3716584078a91ae307b11b4bf1b2662e1511e9c9`; artifact `8877450455` (102883 bytes); ZIP SHA-256 `99c634bd4fb3903d32a1ed45fada7853ea4e1d199b375c129d1d4b8da4f39cb8`; config SHA-256 `d80c0eb37968f6cfecfbfe781aef406b30b536be873b052c69734aa9add68343`. Reproduce with `experiments/exp_064/reproduce.sh` and verify `results/exp_064/checksums.sha256`.

<!-- EXP-065-AUTHORITATIVE-FINAL -->
## EXP-065 authority

Workflow `30870558294`; source `22fd41697979f0e5aeb570880714a47958270d7f`; merge `2e512e91b5bfcd5e30a19ef163a6438221a134dc`; artifact `8878551394` (244495 bytes); ZIP SHA-256 `cf5bfcc53bda4117430c0856b6989704e79bb34fb52c9a4f81869bf20233155d`; config SHA-256 `6dd637104c6edfdaaf424d22790e1f521dc9fa59f9a10f59552a6dfeaec18666`. Reproduce with `experiments/exp_065/reproduce.sh` and verify `results/exp_065/checksums.sha256`.

<!-- EXP-066-072A-AUTHORITATIVE-CATCHUP -->
## EXP-066 through EXP-071 authority

Machine-readable frozen authorities are `results/exp_066/summary.json` through `results/exp_071/summary.json`. Their workflow/artifact identities remain authoritative in each result bundle and `RESEARCH_STATE.md`; EXP-071 workflow is `30965323458`, artifact `8914506737`, ZIP SHA-256 `bc81e90e3b5a35935f893ad7396d4b41a13de46606ce14bccc53cf79e30e8ba4`.

## EXP-072A local authority

```text
results/exp_072a/summary.json
source commit       468f297925e10bdc541fe48f19c2f72a1e3f5e14
evidence commit     f9ac26befb01fd9a71c7c6e1efed4c4b4df31389
config SHA-256      089875c12bb2d0dd15bda6fb8e584bf8835053f40862bec31865493216483643
core SHA-256        112f0490e9e21efc8df3da04f71d92adacba4c493d2eae07d94dbf0960ac8c66
workflow/artifact   NOT RUN
```

Local authority environment: Windows 11, Python 3.12.13. Nine experiment-specific tests and the 330-test repository suite passed after the logging regression was fixed; `scripts/run_validation.py` also completed. The canonical run, an isolated reproduction, and the standard `run_current_env.sh` path returned the same decision and deterministic core hash. `results/exp_072a/checksums.sha256` contains seven entries and verified with zero mismatch.

Reproduce without overwriting authority:

```bash
bash experiments/exp_072a/reproduce.sh
```

Expected invariants: two finite domains, 272 matrices, 272 unique basis signatures, zero signature collision/control failure, Q4 information `188.98828125 GiB`, hot fraction `0.042330666997375005`, and the authoritative rejection decision.

The absence of a GitHub workflow run is explicit. Do not present the local result as Linux CI or Phase D evidence.

## EXP-073 Stage 1 target-inventory authority

```text
results/exp_073/summary.json
source commit       d3b1d2e4dd08e73781c969814cb4d181377a054d
evidence commit     4e35afb9648dc0c513c3a90c32d604f4c0b0fd21
config SHA-256      d7867c68a135bd69e5cfc8b733b62f7c5c4ea50b9190b231d60c0653f3d4f0d0
core SHA-256        aa9cae0457a6b92fcb75da35fedc1a2a2a9f3da115808d341a5a498ca4722da2
workflow/artifact   NOT RUN
```

The collector sent an LF-only read-only shell program over an existing SSH configuration and retained no connection alias, hostname, address, username, key, device name, serial, UUID, or mount path. The checkout-stable evidence commit pins the experiment definition and frozen result bundle to LF and corrects the log checksum to the committed LF bytes; measured inventory, summary, and evidence-core hash are unchanged. The result bundle contains four checksummed files plus the checksum manifest; all four verified on fresh Windows checkout. The forbidden-identifier scan and evidence-core recomputation passed.

Reproduce only with an authorized runtime-only SSH alias:

```bash
EXP073_HOST=<runtime-only-alias> \
EXP073_OUTPUT_DIR=results/exp_073_reproduction \
bash experiments/exp_073/reproduce.sh
```

The alias is not serialized. Reproduction reruns 14 experiment tests before collection and refuses to overwrite a nonempty result directory. It performs no remote write, package install, model enumeration, inference, service mutation, or benchmark. Snapshot-dependent free RAM/VRAM may change; hardware identity and static capacity fields should remain stable.

Closure verification passed: 14/14 EXP-073 tests, 344/344 repository tests, `scripts/run_validation.py`, and the Git Bash offline `run_current_env.sh` fixture Gate. The offline fixture remains synthetic and is not target evidence.

Expected decision: `COMPLETE_SANITIZED_READ_ONLY_TARGET_INVENTORY`. This is target inventory evidence only. GitHub Actions runs a synthetic fixture and must never be cited as target evidence. Stage 2 and Phase-D runtime validation remain not run.

## EXP-074 local authority

```text
results/exp_074/summary.json
source commit       8abc06e73c884b839927cf41d5f4fa6cbb8fc051
evidence commit     c1d778af011672ec7fadfa66935ba2548de8e115
config SHA-256      f86c32cbaecce73a27f77cbb1fd46f19d5096fcce425d7f4301372e64aed4e97
core SHA-256        404b43088448eaafc3f3d9631cdc3271dc9ebc16b635e27c9211cf9aa0459a65
workflow/artifact   NOT RUN
```

Local authority environment: Windows 11, Python 3.12.13. Nine EXP-074 tests
passed. The canonical reference run emitted 108 scenario rows, 18 minimum rows,
seven controls with zero failure, and eight checksummed payloads plus the
checksum manifest. It did
not use checkpoint weights, contact the target server, or execute inference.

Reproduce without overwriting authority:

```bash
bash experiments/exp_074/reproduce.sh
```

Expected decision:

```text
REVISE_MTP1_AND_EXPERT_PAGING_INSUFFICIENT_REQUIRE_LONG_CAUSAL_PROPOSAL_AND_ROUTING_LOCALITY_GATES
```

Expected invariants: MTP-1 optimistic fraction `10.0`, fixed-route/free-draft
p50 minimum `9`, dense 405B zero-draft minimum `85`, dense 405B/4B-draft
minimum `507`, direct-pull remainder `22.1811537743 GiB`, and deterministic core
hash above. Physical timing and model behavior remain unverified.

## EXP-075 local authority

```text
results/exp_075/summary.json
source commit       f85ac583a129070247992987d1b3c63634e6447f
evidence commit     2fcf7315cf9da491a5ca361536eb0f07e325e74c
config SHA-256      8f7824e18f9819f5bac1ed18dd1e0dbab6544883935f1775bce13627b8f2a637
core SHA-256        2515bb53a0e2967cfd23e15d18720142337c7d25dc658db057e86e1aa45b5674
workflow/artifact   NOT RUN
```

The canonical Windows/Python 3.12.13 audit fetched six revision-pinned public
UTF-8 files totaling 255,779 bytes and wrote 13 checksummed payloads plus the
checksum manifest. The source directory contains config, weight-index, model
card, and three vLLM source files; it contains no safetensors payload.

Eight experiment tests passed. Offline replay using the frozen source directory
through `run_current_env.sh` returned the same decision and deterministic core
hash. Verify authority with:

```bash
cd results/exp_075
sha256sum -c checksums.sha256
```

Reproduce against the immutable network revisions without overwriting authority:

```bash
bash experiments/exp_075/reproduce.sh
```

Offline replay:

```bash
EXP075_SOURCE_DIR=results/exp_075/raw/sources \
EXP075_OUTPUT_DIR=results/exp_075_offline_reproduction \
bash experiments/exp_075/run_current_env.sh
```

Expected decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

Expected invariants: 15 exact MTP tensor keys, one MTP layer, three passing
runtime source surfaces, five controls with zero failure, declared checkpoint
size 1,746,882,752 bytes, and the deterministic core hash above. Network timing
and local paths may vary. Model execution and E2-E7 remain unverified.

## EXP-076 local authority

```text
results/exp_076/summary.json
source commit       5e331137f8e03250cc74aa796abbf69f49ef87a5
evidence commit     55b79937c1f21887ae76b7e56ad61ba7dde8322a
config SHA-256      9c00c81d6b91f1da7f499f3ce4da2b477a12afeee0685888615e181aa22b438b
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
core SHA-256        199db6f8fc0dedd32d7b38be8ff1d05c29aced3388a87ddecccd1235038bd22d
workflow/artifact   NOT RUN
```

The canonical Windows/Python 3.12.13 CPU run used Torch 2.6.0+cpu,
Transformers 5.12.0, BF16 eager attention, and eight threads. The model payload
is intentionally ignored by Git and must be obtained from the pinned manifest.
The result bundle contains eleven checksummed payloads plus the checksum
manifest; all checksums verified after the run.

Reproduce without overwriting authority:

```bash
EXP076_PYTHON='C:/dincAI/Vortex/.deps/exp076-venv/Scripts/python.exe' \
EXP076_MODEL_DIR='C:/dincAI/Vortex/.deps/exp076-model' \
EXP076_OUTPUT_DIR='C:/dincAI/Vortex/results/exp_076_reproduction' \
bash experiments/exp_076/run_current_env.sh
```

Expected decision:

```text
REJECT_NATIVE_MTP_LONG_BLOCK_AS_SURROGATE_CORE
```

Expected deterministic invariants: 24 cases, 144 K rows, selected `K=4`,
accepted-prefix p05/p50/p95 `0/4/4`, two zero-accept evaluation cases, required
p05/p50 minima `9/11`, and zero integrity-control failures. Wall/CPU time may
vary. The result is E1 CPU observation only; vLLM equivalence, physical traffic,
target hardware, and E2-E7 remain unverified.

## EXP-077A local authority

```text
results/exp_077a/summary.json
source commit       33fed17ed6abed7c8c14eec543efc620e4fe537d
evidence commit     0970c6626ff848c5026b684b3e2d1bb479e96603
config SHA-256      fce6edd0ea572ce3b6b50ad97500822d24fc1207b8ac026866753cff83459837
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
trace SHA-256       1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
core SHA-256        e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411
workflow/artifact   NOT RUN
```

The Windows/Python 3.12.13 CPU run used Torch 2.6.0+cpu, Transformers 5.12.0,
BF16 eager attention, and eight threads. It replayed 24 frozen causal cases
through equal-length, unpadded two-stage cache batches. The bundle contains ten
checksummed payloads plus the checksum manifest; all checksums verified.

Reproduce without overwriting authority:

```bash
EXP077A_PYTHON='C:/dincAI/Vortex/.deps/exp076-venv/Scripts/python.exe' \
EXP077A_MODEL_DIR='C:/dincAI/Vortex/.deps/exp076-model' \
EXP077A_OUTPUT_DIR='C:/dincAI/Vortex/results/exp_077a_reproduction' \
bash experiments/exp_077a/run_current_env.sh
```

Expected decision: `REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH`. Expected
invariants are 0/192 baseline mismatch, 72 case rows, realized fraction
`0.09988839285714286`, held-out 10% top-1 `0.7152777777777778`, mean KL
`0.8841612071313042`, p95 KL `3.080751657485962`, and the core hash above.
Wall time may vary. Physical performance, large-model scaling, and E2-E7 remain
unverified.

## EXP-078A preregistered local command

Interactive construction-amortization prototype:

```powershell
.deps\exp076-venv\Scripts\python.exe -m vortex_runtime.tangent_macroblock_prototype
```

Pinned unchanged-checkpoint favorable lifetime run:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_078a\run_experiment.py --model-dir .deps\exp076-model --output-dir results\exp_078a
```

The command inherits the exact EXP-076 dependency and checkpoint manifests. It
must start from an empty output directory and performs no network or Ubuntu
server operation. No expected scientific decision is recorded before the source
commit and run complete.

## EXP-078A local authority

```text
results/exp_078a/summary.json
source commit       cc368190031d87db92f7a476dd741c18224239c1
evidence commit     fe6081917c65b2392f8760d72a2b0e17a4461982
config SHA-256      4ffc97eb9b3ab17354df2adbfedaf77b3eafcbbf80e1573e855a0c6a8370046a
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
trace SHA-256       1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
core SHA-256        c743ae14748effaad3a034def7d8d92e67fa09abf65eeb931bef3e8a26368667
workflow/artifact   NOT RUN
```

The Windows CPU run used the inherited pinned EXP-076 environment and took
`1,184.806152` seconds. The nine payload checksums verified independently.
Expected decision is `REJECT_FROZEN_TANGENT_MACROBLOCK_REUSE_PATH`; expected
invariants are 0/192 baseline mismatch, 24 candidate cases, 126 held-out reuse
tokens, 0 top-1 matches, valid-prefix p05/p50/p95 0/0/0, mean/p95 KL
`14.89842255626406/25.335416793823242`, and the core hash above.

## EXP-079A preregistered local commands

Interactive fail-closed state prototype:

```powershell
.deps\exp076-venv\Scripts\python.exe -m vortex_runtime.causal_proof_state_prototype
```

Pinned unchanged-checkpoint favorable Gate:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_079a\run_experiment.py --model-dir .deps\exp076-model --output-dir results\exp_079a
```

The command inherits the exact EXP-076 checkpoint and dependency manifests,
requires an empty output directory, performs no network or Ubuntu-server
operation, and writes checksummed raw/processed/artifact/log evidence. No
scientific result or expected decision is recorded before the source commit.

## EXP-079A local authority

```text
results/exp_079a/summary.json
source commit       e0c661eb5fe39a6835262567d75d388c4fc66c43
evidence commit     38bfd6c6277426bdf8f8e43593482125a0f776bd
config SHA-256      fc6add8c0e230bd542f02a06d750acac10402a703d317ab65a3977305b7d3996
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
trace SHA-256       1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
core SHA-256        c8d794bea8f17b31e40b9f667836d2198ce80137e23080260f81ea6866112eeb
workflow/artifact   NOT RUN
```

The Windows/Python 3.12.13 CPU run used Torch 2.6.0+cpu, Transformers
5.12.0, BF16 eager attention, and eight threads. It completed in
`1,628.6480521` seconds. The bundle has eleven payload checksums plus the
manifest; independent verification found zero mismatch. The seven
experiment-specific tests passed, the repository suite passed `392/392`, and
`scripts/run_validation.py` completed.

Expected decision is `REJECT_DCT_BLOCK_ZONOTOPE_CAUSAL_PROOF_PATH`. Expected
invariants are 0/192 baseline mismatch, p50 charged traffic
`0.011630347067120069`, 23/256 selected blocks, held-out top-1 `6/144`, mean/p95
KL `7.544862263732487/12.616226196289062`, minimum sound-radius p50/p95
`48.66391755845644/57.77874760553659`, and the core hash above. Physical speed,
large-model scaling, and E2-E7 remain unverified.

## EXP-080A preregistered inputs

```text
shape population     results/exp_071/raw/tensor_rows.jsonl
shape SHA-256        5009a4ed7234bd24eb1488b0f96a9c847158c2a614a6c90519a16cf4212ba82a
EXP-048 summary      18252019a35ecb23fe2c8525e13a93ec73a5b69eea00330e965932f4148ef718
EXP-049 summary      f0683a7852f1efe930b22c1feaf4863b68bf9d4351217461080ee784e6488a57
EXP-050 summary      00f76310a45187c820a3b76dc70abf673ebd3c1c74bb90d88536c2138e685a2a
```

The result must be reproducible without network, model payload, or Ubuntu host.
It must freeze exact integer-control rows, per-block constructive and exponent-
oracle counts, aggregate Gate rows, environment data, logs, and checksums.
Canonical commands and authority hashes will be recorded only after the source
commit and run complete.

## EXP-080A local authority

```text
results/exp_080a/summary.json
source commit       7f1c66125a844f5366e2b88f0aaa07b089dc0378
evidence commit     98e90899a0adf3614d8dbe17691167ce426bae3b
config SHA-256      97f02f4f16cf5befb498cc7d34ed1c837b3ad10b2fee9a4fa408936a4cb4f985
core SHA-256        7578c4c9f463da8135f3c320df9d7fb920ffc172d31fdd2f60b30af9778280ce
summary SHA-256     0debbb96f0a31b1ef2ffc1e662109687aed3b2cedaaf89ac1c013bf9a94c1d83
workflow/artifact   NOT RUN
```

Canonical local command:

```powershell
C:\Users\dinc2\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe experiments\exp_080a\run_experiment.py --output-dir results\exp_080a
```

The Windows/Python 3.12.13 standard-library run passed 80/80 exact signed-
integer controls. Its bundle contains ten checksummed payloads plus the
manifest; independent SHA-256 verification found zero mismatch. A second run in
an empty temporary directory reproduced the deterministic core hash, all 80
controls with zero mismatch, and the authoritative decision.

The repository suite passed `392/392` with the repository root explicitly on
`PYTHONPATH`, and `scripts/run_validation.py` completed. An initial unqualified
suite invocation produced one EXP-072A subprocess import failure because that
child did not inherit the repository module path; rerunning with the documented
root path removed the environment-only failure.

Expected decision is
`REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY`.
Expected best constructive invariants at `K=16,384` are traffic
`0.00006103515625`, arithmetic `0.3611157967135951`, favorable workspace
`8,095,006,720` bytes, zero constructive joint passes, first unit-constant
omega pass `K=512`, and first 4B-draft omega pass `K=8,192`. No checkpoint,
network, Ubuntu host, physical kernel, or E2-E7 evidence is involved.

## EXP-081A preregistered inputs

```text
shape rows SHA-256  5009a4ed7234bd24eb1488b0f96a9c847158c2a614a6c90519a16cf4212ba82a
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
field prime         2147483647
recovery rank       8
verification rows   6
```

The result must freeze control rows, target-shape accounting, build/evaluation
leakage audit, per-projection and family coverage, error spectra, raw logs,
environment inventory, and checksums. Source/evidence hashes and the canonical
command will be recorded only after implementation and execution.

## EXP-081A local authority

```text
results/exp_081a/summary.json
source commit       1d3e91fea8a8bbb68613c1afa2a56213bcd5fe7e
evidence commit     ee9573d7760ae5adea47290c3b7b9f89af7cecfa
config SHA-256      e139f81f01327e60028b6d28201db8fe57b2de000941642b56d35f9d7faff44b
shape SHA-256       5009a4ed7234bd24eb1488b0f96a9c847158c2a614a6c90519a16cf4212ba82a
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
summary SHA-256     52790207b424519c1abb512dc37a875f57f47fcea41c7a46214bae57a225ffa9
core SHA-256        8621f6357536b6fc3396872668484c52103e28d2af8291b96575bc4d007c2ccc
workflow/artifact   NOT RUN
```

Canonical command:

```powershell
.deps\exp076-venv\Scripts\python.exe experiments\exp_081a\run_experiment.py --output-dir results\exp_081a
```

The registered layer `0` did not expose `q_proj`; before any prompt forward
pass, the data-independent topology repair changed the tuple to the earliest
valid full-attention layer `3` plus `11,23` and committed the amendment. A later
metric-complete attempt stopped only because `logs/` had not been created. Its
seven scientific payload hashes all matched the canonical run after the
one-line packaging repair.

The canonical Windows/Python 3.12.13 CPU run completed in `310.8` seconds. It
passed 321/321 exact/fault controls, wrote ten checksummed payloads plus the
manifest, and independent checksum verification found zero mismatch. A second
run from evidence commit `ee9573d` in an empty output directory completed in
`322.7` seconds and reproduced the deterministic core plus all seven selected
scientific payload hashes byte for byte.

The repository suite passed `392/392` with the root on `PYTHONPATH`, and
`scripts/run_validation.py` completed. Expected decision is
`REJECT_SYNDROME_RECOVERED_LOOKUP_RESIDUAL_CODE_PATH`; expected weighted exact
coverage is `0.08681672025723475`, corrected relative-L2 p50/p95 is
`0.3512685298919678/1.2138284623622893`, and authoritative fast-path logical
traffic is `0.009266579409111565`. No network, target-Ubuntu command, physical
kernel, 122B/405B, or E2-E7 evidence is involved.

## E0 proof-carrying trace audit

The throwaway calculator reads the frozen EXP-080A shape rows and applies the
closed random-linear verification equations. Authoritative numbers are copied
to `docs/research/E0_DECISION_AND_PROOF_TRACE_TRIAGE.md`; the prototype is
deliberately ignored under `.deps/proof_trace_e0_prototype/` and is not a
production or experiment artifact.

Expected six-challenge invariants are 883 matrix instances,
`403,747,897,344` dense coefficients, `234,659,328` verifier operations,
`513.724 MiB` verifier traffic, and `0.427185 GiB` sidecar. All are derived;
no model was run.

## EXP-082A preregistered inputs

```text
contract             docs/research/EXPERIMENT_082A_DIFFERENTIAL_SPANNING_TREE_GATE.md
config               experiments/exp_082a/config.json
model/revision       Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256       04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
layers               3, 11, 23
families             q/k/v/o/gate/up/down projections
block size           32 coefficients
target fraction      0.011851851851851851
```

No result path, source/evidence commit, or canonical result command exists in
the preregistration commit. The runner must freeze exact block IDs, both
orientation bounds, per-matrix/family aggregates, controls, environment, logs,
checksums, and an infrastructure/scientific decision separation before any
result is interpreted.

## EXP-082A authoritative reproduction

```text
source commit        2caaba054d53ab81d8bdc2fc83aaa7f8241b0e4c
evidence commit      5a7c7c1518333fa75c0934f0e3303a20dca72d17
config SHA-256       34a0b9ba00d5b5ac4f305c0aea1bbaf1f04ad28fabb0d1a43089fdd352514cbe
weight SHA-256       04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
summary SHA-256      649f04b7ac5eb3f762c9fd50fd42efc02dd09a544b6345020957f41aace0ea27
deterministic core   8bf5d1f28ddf0fb53d07f3bdbc9e46e958b0375f4a53318b3c19b87e438a7a87
decision             REJECT_DIFFERENTIAL_SPANNING_TREE_FROM_CERTIFIED_LOWER_BOUND
```

Canonical command from the source commit:

```powershell
$env:PYTHONPATH = "repo;.deps"
.deps\exp076-venv\Scripts\python.exe experiments\exp_082a\run_experiment.py --output-dir results\exp_082a
```

An independent output directory under ignored `.deps` reproduced the decision
and deterministic core. `control_rows.jsonl`, `matrix_rows.jsonl`, and
`aggregate.json` were byte-identical to committed evidence with SHA-256 values
`af8cba996ddb3c06b0229a366223496fd52c4edf63a8472c5e9b9b8e4615e3c6`,
`219c317e6cb64b3559d33975e2181ea9ad186f7593ad78692f2e358a97dc3553`, and
`d382b59daed0ae5e1aa15e33e91614232bb56e014c1584818bf9ffcdc0bd4d4b`.
The run used the existing local pinned payload only; no server or download was
used.

## E0 synthetic-intermediate circuit audit authority

Authority document:
`docs/research/E0_SYNTHETIC_INTERMEDIATE_CIRCUIT_AUDIT.md`.

The audit is reproducible by checking these symbolic steps against the frozen
ledgers:

```text
synthetic edge                 z_v = z_u + (v-u).x
class inclusion               synthetic tree subset static linear DAG subset EXP-072B
metric inequality             SMT(S) >= MST(S) / 2
EXP-082A consequence          1.562367394% / 2 = 0.781183697%
registered allowance         1.185185185%
EXP-072A information floor   1,623,396,974,592 bits = 188.98828125 GiB
target root free capacity    97.6183 GiB
```

External theorem references are the linked Jiang--Miller--Pritikin hypercube
Steiner paper and Boyar--Find linear-circuit paper in the authority document.
Their results are used only as asymptotic/distributional E0 counterevidence.
No source code, model weights, private server, hardware, or benchmark was run;
there is no result artifact or experiment command.

## E0 query-adaptive cold-backed equation reproduction

Authority document:
`docs/research/E0_QUERY_ADAPTIVE_COLD_EQUATION.md`.

The deterministic calculator uses only registered integer shapes and decimal
resource fractions:

```powershell
python scripts/derive_query_adaptive_cold_equation.py
python -m pytest tests/test_query_adaptive_cold_equation.py -q
```

Expected invariants include `201,873,948,672` non-embedding Q4 bytes,
`98.814814815%` zero-cost minimum coverage, `99.081653739%` coverage after the
known verifier, 4-KiB page limits `584,126/452,612`, and the independent
EXP-081A frontier `99.741472756%`. Nine focused tests cover the branch equation,
miss/build charging, page/index state, compile amortization, and favorable PCIe
floor. No model, target server, hardware, or network access is required.

## Causal Residual Atlas source reproduction

Authority document:
`docs/research/E0_CAUSAL_RESIDUAL_ATLAS_SOURCE.md`.

```powershell
$env:PYTHONPATH = "repo;.deps"
python scripts/derive_causal_residual_atlas.py
python -m pytest tests/test_causal_residual_atlas.py -q
```

Expected registered rank-16/requested-0.2%/64-column/64-token invariants are
`526,667,776` capsule elements, `0.980995178 GiB` capsule state, `298,624`
metadata blocks, `1,009` selected pages/token, `1,466,736,640` capsule read
bytes/token, `0.349720584%` actual cold coefficient fraction,
`1.085025716%` amortized traffic, `0.557958575%` amortized operations, and
`99.899840530%` minimum traffic-governed coverage.

Ten focused tests cover pair-only `WQ` construction, exact committed-prefix
reconstruction, dependent and late-independent prefix handling, certified unread residuals, all-page exact
completion, strict top-1 certification, page rounding, minimum build
amortization, corrupt/non-finite state, and randomized no-false-bound cases.
No checkpoint, network, private server, or hardware is required. Real causal
coverage and E2-E7 remain absent.

## Causal Residual Atlas cheapest Gate preregistration

Authority document:
`docs/research/CAUSAL_RESIDUAL_ATLAS_CHEAPEST_GATE.md`.

The preregistration arithmetic and favorable page-choice rule reproduce
without a checkpoint:

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_residual_atlas_gate.py
python -m pytest tests/test_causal_residual_atlas_gate.py -q
```

Expected invariants are rank 16, teacher index 1, layer 11, page width 64,
18 evaluation token states, 36 q/down projection branches, 16/56 pages,
1,296 enumerated candidates, minimum coverage
`0.998998405303215166463`, required token successes 18, maximum failures zero,
and required family successes three. Seven focused tests cover the arithmetic,
ceil-to-success rule, top-1-first/minimum-KL page selection, deterministic tie
breaking, no-match diagnostics, and malformed/non-finite rejection.

Pinned execution inputs, when the next ticket runs the Gate:

```text
model/revision       Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256       04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
prompts SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
trace SHA-256        1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
```

No result path, experiment number, source/evidence authority, or expected
scientific decision exists in this commit. The next runner must start from an
empty output directory and freeze config, input audit, prompt/basis ranks,
per-page or at least selected-plus-failure rows, per-branch/token/family
aggregates, controls, environment, logs, and checksums. It must distinguish a
scientific Gate failure from missing payload, timeout, memory, or dependency
failure.

## EXP-083A authoritative execution and replay

Source commit:
`1c7dd78097beaa7bc159a8f3459451b876a1338e`.

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083a\run_experiment.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_083a

.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_083a\verify_results.py `
  --output-dir results\exp_083a `
  --write-report
```

The complete primary run returns
`PROMOTE_CAUSAL_RESIDUAL_ATLAS_TO_CAUSAL_PAIR_AND_BOUND_GATE`, 18/18 token
successes, 36/36 branch successes, mean/p95 KL
`0.007225545020063708/0.037660752986209814`, and deterministic-core SHA-256
`ab96e6114f44a1c02a01d080848c4ec643a747b63ebfee30c77fda45f3954845`.

The model run was repeated from the same source and registered inputs into an
empty `results/exp_083a_reproduction` directory, then independently verified.
Its decision, every deterministic scientific row, and core SHA-256 are
identical. Timing and external process-memory telemetry are intentionally
outside the deterministic core.

The original run source returned a null Windows internal RSS value because its
ctypes function signatures were implicit. `peak_rss_bytes()` now uses explicit
64-bit `K32GetProcessMemoryInfo` signatures; stored external OS high-water
telemetry preserves the observed primary/reproduction CPU working sets. This
instrumentation repair does not alter either scientific core.

The post-hoc selector audit in
`results/exp_083a/processed/exploratory_selector_audit.json` is explicitly not
part of the frozen Gate. It derives page availability and the failed
minimum-radius selector from committed rows only.

## Legal pair/outward Gate preregistration

Authority document:
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`.

Frozen untouched population:

```text
path    docs/research/inputs/causal_residual_atlas_legal_gate_prompts.json
SHA-256 67bd16d4f1e63a6f4e4a9131119335aff8d4d676549ce55287bf69b8fb879fb5
rows    24, four per family
```

No checkpoint output for those rows, result directory, experiment number,
source/evidence authority, or expected scientific result exists in this
preregistration. The prior 18 EXP-083A evaluation rows are provenance only and
may not affect the next decision.

Pure reproduction:

```powershell
$env:PYTHONPATH = ".;.deps"
python scripts/derive_causal_residual_atlas_legal_gate.py
python -m pytest tests/test_causal_residual_atlas_legal_gate.py -q
```

Expected invariants include 24/24 required tokens, layer-23 `down_proj`, rank
16, 64-column pages, proof metadata `1,055,320` bytes, charged logical
traffic/operations `0.010937062717971382/0.009287466203790127`, controlling
coverage `0.999085210866119530`, and registered spectral-compile work
`28,052,251,108,674,513` scalar operations. Ten focused tests pass.

The future runner must start from an empty output directory, pin the authority
and prompt hashes, and freeze pair rows, selected page, every outward term,
strict margin, native comparison, dense completion, leakage/taint controls,
resource counts, environment, logs, and checksums. An independent verifier
must rebuild the selector, radii, aggregate decision, and deterministic core.
