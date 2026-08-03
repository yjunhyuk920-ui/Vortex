# EXP-050 — External Draft Advice Gate

Model-independent tests:

```bash
python -m pytest -q tests/exp_050
```

Pinned cross-checkpoint CPU audit:

```bash
bash experiments/exp_050/run_current_env.sh
```

Isolated reproduction:

```bash
bash experiments/exp_050/reproduce.sh
```

The runner generates one 256-token cached continuation per model/prompt, then cross-verifies every other model's proposal with one exact target block pass. K=64/128/256 rows are causal prefixes of the same pass.

A reference-selected best draft/K is recorded only as a favorable non-deployable upper bound. E0 first-token adversarial target and E3 exact future-target oracle are separate controls.

405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, combined hot-state fit, and a deployable selector are NOT TESTED.
