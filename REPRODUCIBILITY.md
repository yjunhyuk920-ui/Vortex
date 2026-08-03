# Reproducibility

## Objective

An independent researcher or new session must determine exactly what ran, what did not run, and how every result was produced using only the repository and pinned external checkpoints.

## Provenance

Every result separates MEASURED, DERIVED, PROJECTED, and UNVERIFIED. Unavailable fields are null or `NOT TESTED`, never invented.

## Required experiment layout

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/{README.md,config.json,run_current_env.sh,reproduce.sh,future_gpu_run.sh,environment.md}
results/exp_xxx/{raw,processed,summary.json,logs,artifacts,checksums.sha256}
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

## EXP-047 final authoritative run

```text
PR: #56
workflow: 30792813542
source implementation SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
bot evidence commit/current head immediately after run: 3359371762c004db3532ebb16872b4eee85accf6
workflow conclusion: success
phase: A/B
evidence: E1
Phase D: NOT TESTED
```

The workflow:

1. checked out the source research branch;
2. removed stale `results/exp_047`;
3. ran 10 EXP-047 tests and the measurement runner;
4. normalized source provenance;
5. regenerated SHA-256 checksums;
6. validated required provenance and raw files;
7. uploaded an artifact;
8. committed the complete result directory to the branch.

Committed evidence:

```text
results/exp_047/raw/cases.jsonl
results/exp_047/processed/scaling.json
results/exp_047/summary.json
results/exp_047/logs/run.log
results/exp_047/logs/workflow_stdout.log
results/exp_047/artifacts/certificate_contract.txt
results/exp_047/checksums.sha256
```

## Reproduce EXP-047

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q tests/exp_047
bash experiments/exp_047/reproduce.sh
```

Verify the committed authoritative evidence itself:

```bash
cd results/exp_047
sha256sum -c checksums.sha256
```

A fresh rerun regenerates environment-dependent timing, so its result checksum need not equal the authoritative runner. Logical decisions, counts, seeds, and contracts must match; timing is compared as a distribution.

## Determinism

EXP-047 config seed: `20260803`.

Per-case tile permutations are deterministic. Fixed-seed equality is tested. Timing is not deterministic.

Future real-model experiments must additionally pin prompt hashes, model/tokenizer revisions, decoding parameters, thread counts, and deterministic framework flags.

## Independent validation

Phase B requires a slow reference, optimized candidate, independently calculated critical bound, cross-checks, and fault injection.

EXP-047 independently recomputed accepted interval endpoints. Final authoritative mismatches: zero.

## Checksum generation

```bash
find results/exp_xxx -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed 's#  results/exp_xxx/#  #' \
  > results/exp_xxx/checksums.sha256
```

## External checkpoints

Real-model experiments pin model ID, exact revision, file manifest/hashes, tokenizer revision, download command, cache path, license/access state, and prompt corpus hash. Moving `main` is not authoritative.

## Workflow measurement boundary

EXP-047 reruns only when implementation, config, tests, or its workflow changes. Result interpretation/root-document commits do not regenerate measurements. This prevents raw timing identity from changing during documentation-only work.

## Infrastructure failures before authoritative run

- `30791055142`: eager optional `safetensors` import;
- `30791192434`: missing repository root on `PYTHONPATH`.

They are infrastructure failures, not hypothesis evidence. Lazy imports and explicit `PYTHONPATH` corrected them.

## Reproduction failure procedure

1. classify infrastructure versus scientific failure;
2. preserve logs;
3. do not interpret missing checkpoints/timeouts as scientific evidence;
4. update durable state;
5. rerun only after correction.
