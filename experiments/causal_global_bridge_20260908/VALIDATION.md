# Local validation record — 2026-09-08

This file records what actually ran after the final causal/global source and
ledger edits. It is not target-hardware evidence.

## Focused causal/global suite

From `experiments/causal_global_bridge_20260908`:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '..\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest `
  test_causal_kv_exposure `
  test_basis_column_producer `
  test_causal_rank_one_trace -v
```

Observed:

```text
Ran 14 tests
OK
```

## Related nonlinear/geometry regression

From repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest `
  tests.test_nonlinear_router_cover_gate `
  tests.test_adaptive_nonlinear_probe_degree_gate `
  tests.test_joint_batch_coset_geometry -v
```

Observed:

```text
Ran 28 tests
OK
```

## Standard repository validation

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  '.\scripts\run_validation.py'
```

Observed exit code: `0`.

The script rewrote only the volatile `elapsed_seconds` field in the tracked
`validation_results.json`; that timing-only working-tree diff was reverted and
is not part of this round.

## JSON and checksum checks

Eight causal/global JSON files parsed successfully with PowerShell
`ConvertFrom-Json`. All six frozen result hashes in
`results/e1_causal_global_bridge_20260908/checksums.sha256` matched their files.

## Whole tests directory attempt

A whole-directory unittest discovery was attempted with the known native-global
venv:

```text
Ran 220 tests in 58.152s
FAILED (errors=15)
```

All 15 discovery errors were import failures because that venv does not contain
`pytest`; no causal/global assertion failure was reported.

The system Python has `pytest 9.0.3` but not torch/transformers. Injecting its
site-packages into the native-global venv allowed pytest collection to start,
but collection still stopped on pre-existing repository/environment issues:

- duplicate test module basenames under different test directories (`test_audit`,
  `test_validation`) under pytest's default import mode;
- missing `scipy` for `exp_096a`;
- POSIX-only `resource` imports in `exp_100a` and fixed-public-dynamic code on
  Windows;
- unavailable Unix `libm` lookup in `tests/native_transfer` on Windows.

Therefore the complete repository suite is **BLOCKED BY CURRENT TEST ENVIRONMENT**,
not recorded as PASS. Focused/related suites and the standard validation above
are the applicable passing local gates for this round.

The old native-global 3,510-file replay was intentionally not rerun because no
native-global artifact changed.

## Post-persistence Boolean lift gate

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest tests.test_boolean_oracle_parity_lift_gate -v
```

Final observed after the feature-lift extension:

```text
Ran 11 tests
OK
```

The exhaustive controls include all 37,067 distinct `<n` query sets for the
full-support deletion adversary at `n<=5` and every nonempty left subset of the
complete inner-product matrix at `d<=4` for the maximum-one-rectangle check.

Canonical Boolean lift result:

```text
results/e0_boolean_oracle_parity_lift_gate_v2/summary.json
SHA-256 2d8a91800d574d32ec7611da6af6046d37cd4230a8c897f5c33ff268238695f6
```

## Not tested

```text
405B checkpoint/execution
CUDA
single GPU <=8 GiB
PCIe/SSD/HBM physical schedule
same-machine native 4B Q4 p50/p95
TTFT
area-5400 25x216 32-query native trace
```
