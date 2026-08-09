# EXP-084A — Causal Bilinear Rank Gate

This is the one-shot E1 execution of the frozen contract in
`docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md`. It measures exact
dyadic rank and exact build-ledger membership of last-layer `down_proj`
queries from the unchanged pinned Qwen3.5-0.8B checkpoint. It is not a runtime
or a 405B performance experiment.

The protected runner implementation is frozen at
`e1902949524274968d899b987747189f792fe967`. No prompt forward is valid if a
protected source path differs from that commit.

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_084a\run_experiment.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_084a
```

Independent verification performs no Transformer forward:

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  experiments\exp_084a\verify_results.py `
  --model-dir .deps\exp076-model `
  --output-dir results\exp_084a `
  --write-report
```
