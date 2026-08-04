# EXP-066 — Exact Tensor-Train / MPO Bond-Rank Gate

This experiment is a bounded cheap-kill screen, not an MPO runtime implementation.

It reuses the unchanged revision-pinned TinyStories-1M/3M/8M checkpoints and the frozen EXP-057 deterministic Q4 integer checksums. For every two-dimensional tensor it evaluates only the preregistered radix family in `config.json` and the five deterministic mode-order variants implemented in `vortex_runtime/tensor_train_rank.py`.

For every internal cut it forms the exact prefix/suffix unfolding and certifies rank independently under primes 251 and 257. The maximum certified modular rank is a rigorous lower bound on the integer/rational MPO bond dimension for that cut.

The experiment charges favorable classical dense-core slot counts, 4-bit core storage, per-row scales, biases, metadata, input/output traffic, and a deliberately optimistic intermediate allowance. It does not reconstruct MPO cores.

Run:

```bash
bash experiments/exp_066/reproduce.sh
```

Default candidate evidence is written to `results/exp_066_candidate/`.

Promotion requires population p50/p90 operation and storage fractions of at most 10%/25%. Failure closes bounded exact classical single-matrix TT/MPO as a primary core direction for the measured population. It does not prove that every conceivable tensor representation is impossible.
