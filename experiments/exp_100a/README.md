# EXP-100A — Explicit Rectangular FMM Gate

EXP-100A continues from EXP-099A's positive dyadic reassociation signal. It
loads the commit-pinned public AlphaTensor standard-arithmetic factorization
catalog, verifies eligible integral small-coefficient tensor decompositions
exactly, and searches bounded mixed recursive compositions specialized to the
registered Llama-3.1-405B projection shapes.

The calculator charges:

- all leaf multiplications and accumulation additions;
- staged activation, weight, and output transform additions/scales/writes;
- checkpoint-static transformed-weight byte expansion at every cut depth;
- compressed cold traffic per committed token;
- original BF16 row side traffic for native-order repair when transformed
  checkpoints no longer contain those rows;
- a favorable no-reread transform-workspace lower bound;
- EXP-099A role-p95 native-order repair operations;
- `N/A=1` only as an explicitly non-deployable perfect-future-block grant.

```bash
pytest -q tests/exp_100a
python experiments/exp_100a/run_experiment.py \
  --config experiments/exp_100a/config.json \
  --output-dir results/exp_100a/local
```

A promotion authorizes only a finite-word transformed-weight/kernel and causal
block Gate. It is not a 405B execution or latency claim.
