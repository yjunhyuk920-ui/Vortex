# Validation — direct global producer frontier

## Executed

Known local interpreter:

```text
experiments/native_global_transition_20260908/.venv/Scripts/python.exe
Python 3.12.10
```

Focused new suite:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest tests.test_direct_global_producer_gate -v
```

Final observed result:

```text
10 tests
OK
```

The suite covers:

- P1 binary/Q4/BF16 information fractions;
- exact zero-sum FP32 rounding gadgets;
- exhaustive row/query gadget composition through width 5;
- registered 883-block vector-route dimensions;
- proof-safe route-cover boundary;
- tiny exhaustive validation of the greedy route-capacity optimizer;
- arbitrary rectangular/square GF(2) Gauss-Jordan producer equality;
- the exact 16,384-width cost formula for the declared elimination producer;
- O1/O5 and hardware claim boundaries.

Canonical generator:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  scripts\derive_direct_global_producer_gate.py `
  --output-dir results\e0_direct_global_producer_gate_v5
```

Canonical summary SHA-256:

```text
a907cb1dbd40b9a7bf71b88a159cf0358c4d3b0656470e71e26f37e2c9185ffe
```

Earlier output directories v1--v4 were intentionally preserved after proof and
cost refinements.

## Deliberately not rerun

Per continuation instruction, this round did not rerun:

```text
prior nonlinear 15/15 focused suite
prior nonlinear/geometry 28/28 regression
prior causal-global 14/14 suite
prior Boolean-lift exhaustive suite
native-global 3510-file replay
```

Those source areas were not modified by this round.

## Not tested

```text
actual 405B checkpoint/execution
CUDA
single GPU <=8 GiB
PCIe/SSD/HBM physical schedule
same-machine native 4B Q4 p50/p95
TTFT
actual CUDA/HF reduction equivalence of the balanced FP32 rounding gadget
complete legal-causal reachability of the simultaneous 883-block query service
```

No PASS claim is made for any item above.
