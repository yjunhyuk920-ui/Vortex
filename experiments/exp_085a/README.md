# EXP-085A

Preregistered reference implementation for a page-separable fused SwiGLU exact-rounding compiler.

```bash
python -m pytest -q tests/test_joint_exact_swiglu_compiler.py
python experiments/exp_085a/run_experiment.py \
  --config experiments/exp_085a/config.json \
  --output-dir results/exp_085a/local
```

The real-checkpoint run is pinned to the official SmolLM2-135M revision in `config.json`. A scientific rejection is an expected successful workflow outcome and is committed as evidence.
