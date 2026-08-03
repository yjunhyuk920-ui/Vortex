# EXP-061

Pinned causal exact activation-sparsity measurement.

The runner registers every unique `torch.nn.Linear` projection, retains shared aliases, and observes projection inputs without modifying them. It separates prompt prefill, first KV-cached decode, and warm decode tokens 2..64.

Only exact positive or negative zero counts. Causal-mask and padding zeros are not measured because attention masking is already standard structural sparsity.

```bash
bash experiments/exp_061/reproduce.sh
```

Evidence ceiling: Phase C observation. Physical activation-sparse kernels, 405B activation statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, and target hardware remain NOT TESTED.
