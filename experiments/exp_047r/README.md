# EXP-047R runner

This directory contains the pre-registered oracle-tight and checkpoint-span stratified range audit.

```bash
python -m pytest -q tests/exp_047r
bash experiments/exp_047r/run_current_env.sh
```

Required Python packages for the real-checkpoint run are pinned in `.github/workflows/exp_047r_gate.yml`.

The runner resolves a Hugging Face revision SHA before each download and execution, saves model/tokenizer file hashes, and writes candidate evidence under `results/exp_047r_candidate/` by default.

The run is offline full-contribution observation. It does not replace a Transformer operation and cannot earn E2. Phase D remains `NOT TESTED`.
