# EXP-049 — Continuous Fixed-Point Gate

Model-independent tests:

```bash
python -m pytest -q tests/exp_049
```

Pinned small-checkpoint CPU audit:

```bash
bash experiments/exp_049/run_current_env.sh
```

Isolated reproduction:

```bash
bash experiments/exp_049/reproduce.sh
```

Conditions:

- S0: hard synchronous Jacobi control;
- S1: fixed damped Picard variants;
- S2: bounded Anderson histories 2/4/8;
- S3: exact future-state oracle, non-deployable;
- S4: hidden triangular causal lower-bound family.

The runner records all fixed trajectories and uses a reference-selected oracle-best S1/S2 row only as a favorable falsification upper bound. It does not implement a deployable runtime selector or complete multi-cycle generation replacement.

Evidence ceiling is E1. 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, and physical target-weight reuse are NOT TESTED.
