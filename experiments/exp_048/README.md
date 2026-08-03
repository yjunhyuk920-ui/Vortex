# EXP-048 — Causal Block Verification Amortization Gate

Run the model-independent exactness tests:

```bash
python -m pytest -q tests/exp_048
```

Run the pinned small-checkpoint CPU Gate:

```bash
bash experiments/exp_048/run_current_env.sh
```

Reproduce into an isolated directory and verify checksums:

```bash
bash experiments/exp_048/reproduce.sh
```

Conditions:

- B0: exact cached greedy; one logical full target stream per token;
- B1: perfect future-token proposal; non-deployable upper bound only;
- B2: Jacobi control; every target iteration charged;
- B3: causal training-free partial-layer self-draft using the same unmodified checkpoint, followed by one exact target block verification pass.

B3 currently audits one pre-registered 32-token block from each held-out prompt using 1, 2, and 4 early layers where available. It verifies the safe longest-prefix-plus-correction commit against a 96-token exact greedy reference.

The runner measures CPU reference behavior on pinned TinyStories checkpoints. It does not measure 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, or tokens/second. Evidence ceiling remains E1 until a complete deployable multi-cycle generation path replaces sequential decoding across held-out runs.
