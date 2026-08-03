# Reproducibility

## Objective

An independent researcher or new session must determine exactly what ran, what did not run, and how every result was produced using only the repository and declared external checkpoints.

## Provenance labels

Every result separates:

- MEASURED — actual declared-environment run;
- DERIVED — formula/calculation from measured or exact inputs;
- PROJECTED — extrapolation to another model/machine;
- UNVERIFIED — not tested.

## Required experiment layout

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/README.md
experiments/exp_xxx/config.json
experiments/exp_xxx/run_current_env.sh
experiments/exp_xxx/reproduce.sh
experiments/exp_xxx/future_gpu_run.sh
experiments/exp_xxx/environment.md
results/exp_xxx/raw/
results/exp_xxx/processed/
results/exp_xxx/summary.json
results/exp_xxx/logs/
results/exp_xxx/artifacts/
results/exp_xxx/checksums.sha256
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

## Required summary identity

```json
{
  "experiment": "EXP-XXX",
  "phase": ["A", "B"],
  "evidence_level": "E1",
  "git_commit": "...",
  "workflow_run": "...",
  "environment": {},
  "config_sha256": "...",
  "checkpoint": null,
  "future_information_used": false,
  "phase_d_status": "NOT TESTED",
  "MEASURED": {},
  "DERIVED": {},
  "PROJECTED": {},
  "UNVERIFIED": []
}
```

Unavailable fields are null or `NOT TESTED`, never invented.

## EXP-047 authoritative run

```text
PR: #56
workflow: 30791851508
source head: d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b
workflow conclusion: success
full tests in experiment workflow: 10 passed
phase: A/B
evidence: E1
Phase D: NOT TESTED
```

The workflow checked out the source research branch, removed stale `results/exp_047`, reran tests and measurement, normalized source provenance, regenerated checksums, uploaded an artifact, and committed the complete result directory back to the branch.

Committed authoritative evidence:

```text
results/exp_047/raw/cases.jsonl
results/exp_047/processed/scaling.json
results/exp_047/summary.json
results/exp_047/logs/run.log
results/exp_047/logs/workflow_stdout.log
results/exp_047/artifacts/certificate_contract.txt
results/exp_047/checksums.sha256
```

## EXP-047 reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q tests/exp_047
bash experiments/exp_047/reproduce.sh
sha256sum -c results/exp_047/checksums.sha256
```

Note: `reproduce.sh` regenerates timing fields, so checksums match only the authoritative committed run, not a new machine's rerun. Logical decisions, counts, and provenance contracts must match; timing is compared as a distribution, not bitwise.

## Determinism

Record:

- Python/framework seeds;
- tile permutation seed;
- threads;
- deterministic flags;
- decode parameters;
- prompt/corpus hashes;
- model/tokenizer revision.

EXP-047 uses committed config seed `20260803` and deterministic per-case permutation seeds. Fixed-seed logical results are tested.

## Independent implementations

Phase B requires:

- slow reference;
- optimized candidate;
- independently computed critical formula/bound;
- cross-checks;
- malformed-state fault injection.

EXP-047 independently recomputes accepted interval endpoints and records mismatches; authoritative mismatches were zero.

## Logs and checksums

Workflow saves stdout/stderr, environment, raw measurements, summary, artifacts, and SHA-256 list.

Canonical generation:

```bash
find results/exp_xxx -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed 's#  results/exp_xxx/#  #' \
  > results/exp_xxx/checksums.sha256
```

## External checkpoints

Real-model experiments pin model ID, revision, file list/hashes, tokenizer revision, download command, cache policy, and access/license state. Moving `main` is not authoritative.

## Workflow rules

Each experiment workflow:

- verifies paths;
- pins dependencies;
- runs experiment tests first;
- fails on NaN, missing metrics, wrong accepts, future leakage, or provenance violations;
- uploads artifacts;
- commits evidence only after the Gate passes;
- avoids recursive runs from result-only commits;
- never labels CPU runs as Phase D.

## Infrastructure failures

EXP-047 recorded two corrected infrastructure failures before the authoritative run:

1. eager package import required optional `safetensors` for a standard-library primitive;
2. script execution lacked repository root on `PYTHONPATH`.

Neither was scientific evidence. Lazy imports and explicit `PYTHONPATH` fixed them. Only workflow `30791851508` is authoritative.

## Reproduction failure procedure

1. classify infrastructure versus scientific failure;
2. preserve logs;
3. do not interpret missing checkpoints/timeouts as hypothesis evidence;
4. update durable state;
5. rerun only after correction.
