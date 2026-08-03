# EXP-055 — Exact Column-Signature Popcount Aggregation Gate

This E1 experiment tests whether exact repeated or sign-related weight columns can replace per-column dense conditional additions with grouped activation popcounts and scaled score-vector additions.

## Registered families

- repeated columns
- exact sign-related columns
- sparse columns
- low-rank columns
- dense-random columns
- forced-unique columns

Both identical-only and exact sign-canonical plans are compiled for every case. The lower fully-accounted logical operation/byte plan is selected only after both compile costs are charged.

## Run

```bash
bash experiments/exp_055/reproduce.sh
```

The candidate artifact contains binary plans, per-plan validation rows, per-case accounting, environment metadata, checksums, and the Gate decision.

## Claim boundary

This is synthetic signed modular binary-linear validation only. Real Transformer operation replacement, 405B execution, 8 GiB VRAM closure, CUDA, PCIe, SSD, TTFT, and tokens/sec are not tested.
