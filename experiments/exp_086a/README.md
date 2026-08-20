# EXP-086A

Exact state-axis lifting favorable-oracle Gate.

```bash
python -m pytest -q tests/exp_086a
python experiments/exp_086a/run_experiment.py \
  --config experiments/exp_086a/config.json \
  --output-dir results/exp_086a/local
```

The run loads the frozen official SmolLM2 checkpoint and consumes future exact
activations. It is an E1 falsification Gate, not a deployable executor.
