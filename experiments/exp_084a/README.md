# EXP-084A — Causal Bilinear Rank Gate

This is the one-shot E1 execution of the frozen contract in
`docs/research/E0_CAUSAL_BILINEAR_QUERY_RESTRICTION.md`. It measures exact
dyadic rank and exact build-ledger membership of last-layer `down_proj`
queries from the unchanged pinned Qwen3.5-0.8B checkpoint. It is not a runtime
or a 405B performance experiment.

The protected sequential runner implementation is frozen at
`6214700d6a5b83097c043b660e4c84e7f83feae0`. No prompt forward is valid if a
protected source path differs from that commit.

Attempt 01 correctly failed closed before producing a query row because a
batched prompt position did not bit-match its causal-prefix replay. The next
source freeze uses only sequential single-token committed-prefix prompt states;
it does not weaken rank, population, hit, or stop thresholds.

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

## Result

The authoritative sequential run passed 114 controls, built 24 independent
queries and a dimension-23 exact ledger, then stopped after the first five
held-out rows were all exact misses. The fifth miss exceeded the registered
four-fallback allowance. Decision:

```text
REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE
```

The held-out rank was five when the miss stop fired; rank 28 is not claimed.
Independent verification performed zero model forwards and rebuilt core SHA
`1e79550fb66fe050338b2eedaf069728fd959583052dee2032fdd57f5cd0a7c4`.
See `docs/research/EXPERIMENT_084A_CAUSAL_BILINEAR_RANK_GATE.md` and
`results/exp_084a`.
