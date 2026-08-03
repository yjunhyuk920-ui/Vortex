# Reproducibility

## Reproducibility objective

A new session or independent researcher must be able to determine exactly what was run, what was not run, and how every result was produced using only the repository and declared external checkpoints.

## Required provenance labels

Every summary must separate:

- **MEASURED:** produced by an actual command in the declared environment;
- **DERIVED:** calculated from measured values or exact formulas;
- **PROJECTED:** extrapolated to another model/hardware scale;
- **UNVERIFIED:** not tested in the current environment.

## Per-experiment required files

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

Git does not preserve empty directories. Use a scoped README or generated file only when necessary; do not fabricate raw results before a run.

## Run identity

Every `summary.json` must contain:

```json
{
  "experiment": "EXP-XXX",
  "phase": ["A", "B"],
  "evidence_level": "E1",
  "git_commit": "...",
  "workflow_run": null,
  "environment": "...",
  "config_sha256": "...",
  "checkpoint": null,
  "checkpoint_revision": null,
  "checkpoint_sha256": null,
  "future_information_used": false,
  "phase_d_status": "NOT TESTED",
  "MEASURED": {},
  "DERIVED": {},
  "PROJECTED": {},
  "UNVERIFIED": []
}
```

Fields unavailable before a run must be `null` or explicitly `NOT TESTED`, never invented.

## Determinism

Where feasible, record:

- Python, NumPy, and framework seeds;
- tile permutation seed;
- thread counts;
- deterministic-algorithm flags;
- decoding parameters;
- input corpus hash;
- model revision and tokenizer revision.

A fixed seed must reproduce the same logical decisions and summary fields. Timing may vary and must be reported as a distribution.

## Independent implementations

Phase B requires:

- a slow, clear reference implementation;
- an optimized implementation;
- cross-checks on the same generated cases;
- fault injection proving the verifier rejects corruption or invalid state.

Shared helper code must not make both implementations repeat the same critical formula without an independent test oracle.

## Logs and checksums

The workflow must save:

- stdout/stderr;
- test report;
- environment inventory;
- raw measurements;
- processed summary;
- artifact manifest;
- SHA-256 checksum list.

Example:

```bash
find results/exp_xxx -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > results/exp_xxx/checksums.sha256
```

## Current-environment commands

```bash
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_047/run_current_env.sh
bash experiments/exp_047/reproduce.sh
```

## External checkpoints

A real-model experiment must pin:

- repository/model ID;
- exact revision/commit;
- file list;
- file hashes where license and tooling allow;
- tokenizer revision;
- download command;
- cache path policy;
- license/access requirements.

Do not use a moving `main` revision as authoritative evidence.

## Workflow rules

Each experiment workflow must:

- verify all referenced files exist;
- install pinned dependencies;
- run experiment-specific tests before the experiment;
- fail on NaN, missing metrics, wrong accepts, or provenance violations;
- upload artifacts;
- commit raw evidence only after all gates pass;
- use branch-specific concurrency;
- never report a synthetic/CPU run as Phase D.

## Reproduction failure

If reproduction fails:

1. classify infrastructure failure separately from hypothesis failure;
2. preserve logs;
3. do not interpret missing checkpoints or runner timeouts as scientific evidence;
4. update `RESEARCH_STATE.md` and the experiment document;
5. rerun only after the failure mode is corrected.
