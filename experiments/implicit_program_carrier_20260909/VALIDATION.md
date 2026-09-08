# Validation — implicit program-carrier frontier

## Executed

Interpreter:

```text
experiments/native_global_transition_20260908/.venv/Scripts/python.exe
Python 3.12.10
```

Focused new suite:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest tests.test_implicit_program_carrier_gate -v
```

Observed current-source result:

```text
11 tests
OK
```

Coverage:

- exhaustive exact alias-router MatVec on a small arbitrary matrix;
- fixed logical alias-address shape;
- square alias arithmetic and metadata accounting;
- registered 883-matrix descriptor accounting;
- exhaustive exact Patricia MatVec controls including duplicate patterns;
- Patricia arithmetic/event and edge-label traffic accounting;
- arbitrary rectangular rank-normal compiler controls;
- encoded rank-normal query equality;
- exhaustive transformed-Hadamard exactness on a 4-bit invertible control;
- frozen micrograph cost showing restored dense work.

Generator:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  scripts\derive_implicit_program_carrier_gate.py `
  --output-dir results\e0_implicit_program_carrier_gate
```

Canonical SHA-256:

```text
f6f911fce9383ab6b82a1ac4ad5ef79f354a8ca58d79d21ff71193000c8d5070
```

## Deliberately not rerun

```text
implicit-direct-query 15/15
direct-global 10/10
prior nonlinear 15/15
prior nonlinear/geometry 28/28
causal-global 14/14
Boolean exhaustive controls
native-global 3510-file replay
```

## Not tested

```text
405B checkpoint/execution
CUDA
single GPU <=8 GiB
PCIe/SSD/HBM target schedule
same-machine native4BQ4 p50/p95
TTFT
native target finite-word reduction ABI
complete arbitrary-checkpoint causal executor
```

No PASS claim is made for the not-tested items.
