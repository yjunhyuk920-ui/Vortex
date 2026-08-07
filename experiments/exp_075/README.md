# EXP-075 runner

Run the pinned public-metadata audit:

```bash
bash experiments/exp_075/run_current_env.sh
```

Use `EXP075_SOURCE_DIR` to replay from six already captured source files without
network access. No safetensors payload or model execution is permitted.

Reproduction output is isolated by default:

```bash
bash experiments/exp_075/reproduce.sh
```
