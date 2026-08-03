# EXP-056 — Exact Prototype Plus Sparse-Residual Dictionary Gate

This E1 experiment tests whether exact dense weight columns can be represented as shared prototypes plus fully charged sparse residual corrections.

## Compiler search

Every case compiles both deterministic `frequency` and `greedy` prototype selection for requested prototype counts 1, 2, 4, and 8. All eight compile costs are charged before selecting the lowest fully accounted runtime plan.

## Registered controls

- exact repeated columns;
- repeated prototypes with sparse exact perturbations;
- positive/negative prototype clusters;
- sparse columns;
- low-rank columns;
- dense-random columns;
- forced-unique columns.

## Run

```bash
bash experiments/exp_056/reproduce.sh
```

The artifact contains every binary plan, validation row, selected case accounting, environment metadata, summaries, and checksums.

## Claim boundary

This is synthetic signed modular binary-linear evidence only. Real Transformer operation replacement, real checkpoint dictionary sparsity, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec are not tested.
