# EXP-093A

Official public-checkpoint one-sweep true-token rank and static branch-source Gate.

```bash
python -m pip install --disable-pip-version-check -r requirements/fixed-public-dynamic-executor.txt
python -m pytest -q tests/exp_093a
python experiments/exp_093a/run_experiment.py \
  --config experiments/exp_093a/config.json \
  --output-dir results/exp_093a/local
```

The guessed-context sweep must finish before the official incremental target starts. Target tokens are evaluation-only oracle data. A positive rank result does not construct branch-dependent successor states or reduce dense arithmetic.
