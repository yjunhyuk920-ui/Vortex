# EXP-060

Exact zero-sparsity streaming measurement for pinned real-Q4 matrices.

Registered representations:

- dense Q4
- scalar CSR
- row-wise nonzero runs
- BSR 1x4, 1x8, 4x4, 8x8, and 16x16

Only exact zero Q4 scalars are skipped. Values, indexes, row pointers, run metadata, nonzero-block internal slots, and padded edge slots are charged.

```bash
bash experiments/exp_060/reproduce.sh
```

Evidence ceiling: Phase C observation. Q4 output preservation, physical sparse kernels, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, and target hardware remain NOT TESTED.
