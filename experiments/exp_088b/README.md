# EXP-088B

This directory implements the frozen oracle cross-layer program-sharing Gate described in `docs/research/EXPERIMENT_088B_ORACLE_PROGRAM_SHARING_GATE.md`.

```bash
pytest -q tests/exp_088b
python experiments/exp_088b/run_experiment.py \
  --config experiments/exp_088b/config.json \
  --output-dir results/exp_088b/local
```

The real-checkpoint run requires the pinned dependencies in `requirements/fixed-public-dynamic-executor.txt` and network access to the frozen public checkpoint revision. The dense evaluator is an offline oracle and must not be reported as a fast executor.
