# EXP-102A

Reality-first causal draft/verify Gate.

```bash
python -m pytest -q tests/exp_102a
python experiments/exp_102a/run_experiment.py \
  --config experiments/exp_102a/config.json \
  --output-dir results/exp_102a/local
```

The hosted run downloads the pinned public target and draft checkpoints. The authoritative arm executes every causal draft, target verification position, repair, state rebuild, and incremental exactness control. No compression or zero-cost oracle credit is used.
