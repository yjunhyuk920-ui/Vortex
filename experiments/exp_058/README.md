# EXP-058 — Pinned Real-Q4 Exact Algebraic-Rank Certificate Gate

This Phase C observation asks whether deterministic row-symmetric Q4 dense-projection matrices from the pinned TinyStories checkpoints can admit a useful **exact** low-rank factorization.

## Exact certificate

Rank is computed over registered prime fields `251`, `257`, and `263`. A nonzero full-size minor modulo any prime proves full rank over the integers/rationals. This avoids approximate floating-point SVD thresholds.

For certified rank lower bound `r` and matrix shape `m × n`, the conventional exact two-factor path is charged at least:

```text
r*n + m*r scalar terms
r*(m+n) factor scalars
```

These are conservative lower bounds; factor coefficient bitwidth and metadata can only increase physical cost.

## Run

```bash
bash experiments/exp_058/reproduce.sh
```

The artifact includes every matrix certificate, certified minor pivots/determinants, factorization lower bounds, EXP-057 Q4 checksum comparisons, permutation controls, aggregate summaries, environment metadata, and checksums.

## Claim boundary

Q4 model-output preservation, constructive factor arithmetic, factor-kernel execution, real Transformer operation replacement, 405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/sec are not tested.
