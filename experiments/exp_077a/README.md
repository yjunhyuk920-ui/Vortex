# EXP-077A runner

This experiment tests whether an activation-informed, non-deployable oracle can
retain the pinned Qwen3.5-0.8B target logit distribution while executing only a
registered fraction of each SwiGLU MLP's intermediate channels.

It reuses the exact checkpoint, prompt manifest, dependency environment, and
target token trajectories frozen by EXP-076. It downloads nothing and never
contacts the private Ubuntu target.

Canonical local run from Git Bash:

```bash
EXP077A_PYTHON='C:/dincAI/Vortex/.deps/exp076-venv/Scripts/python.exe' \
EXP077A_MODEL_DIR='C:/dincAI/Vortex/.deps/exp076-model' \
EXP077A_OUTPUT_DIR='C:/dincAI/Vortex/results/exp_077a' \
bash experiments/exp_077a/run_current_env.sh
```

The full post-SiLU activation and down-column norms are visible to the oracle
and charged as free. Therefore even a passing result is only a favorable upper
bound that authorizes a later causal-selector cost Gate. CPU time is an
environment observation, not a sparse-kernel or hardware speed measurement.
