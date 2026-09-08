# Validation — implicit nonlinear direct-query frontier

## Executed

Known local interpreter:

```text
experiments/native_global_transition_20260908/.venv/Scripts/python.exe
Python 3.12.10
```

Focused current suite:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest tests.test_implicit_direct_query_gate -v
```

Final observed result after the P2 dynamic-state and P3 isotopy strengthening:

```text
15 tests
OK
```

The suite covers:

- exact arbitrary-GF2 query-side image-frame controls;
- checkpoint-independent query-derived address behavior;
- registered 10/11-bit storage/query accounting;
- the equal-block `b=85` target tradeoff;
- the one-sided linear complete-image-frame sparse-subset lower bound;
- exact ordered-leaf value-class execution;
- best-case and high-distinctness value-class operation bounds;
- exact dynamic `(accumulator,weight)` coalescing;
- the FP32-exact-integer all-distinct accumulator adversary;
- linear Hadamard automorphisms;
- arbitrary Boolean-cube meet-preserving bijections;
- independent linear Hadamard isotopies;
- dense-support invariance under safe coordinate gauges.

Canonical current result:

```text
results/e0_implicit_direct_query_gate_v3/summary.json
SHA-256 78644fd5ef4b3e0131bf78867e8986718ed02faeaf37b4228fd1c37909a7af29
```

`checksums.sha256` contains the same digest and an independent `Get-FileHash`
readback matched it.

Earlier outputs are intentionally preserved:

```text
results/e0_implicit_direct_query_gate/
results/e0_implicit_direct_query_gate_v2/
```

They are history, not the current-source authority.

## Deliberately not rerun

Per the continuation instruction, this round did not rerun:

```text
direct-global focused 10/10
prior nonlinear focused 15/15
prior nonlinear/geometry 28/28
prior causal-global 14/14
prior Boolean exhaustive controls
native-global 3510-file replay
```

## Not tested

```text
actual 405B checkpoint/execution
CUDA
single GPU <=8 GiB
PCIe/SSD/HBM physical schedule
same-machine native 4B Q4 p50/p95
TTFT
arbitrary target CUDA reduction ABI
complete arbitrary-native causal producer
```

No PASS claim is made for any item above.
