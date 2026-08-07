# EXP-074 runner

Run the no-download deterministic budget Gate:

```bash
bash experiments/exp_074/run_current_env.sh
```

Reproduction output is isolated by default:

```bash
bash experiments/exp_074/reproduce.sh
```

The experiment does not contact the target server, download checkpoint files,
or run inference. `future_gpu_run.sh` fails closed because no runtime candidate
is promoted.
