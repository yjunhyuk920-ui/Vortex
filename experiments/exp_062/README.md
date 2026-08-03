# EXP-062

Pinned causal exact non-mask attention-probability sparsity measurement.

The runner requests eager attention probabilities and separates global versus local structural masks. Causal, padding, and local-window mask zeros are excluded. Only exact post-softmax zeros at eligible positions may skip Value accumulation.

Whole-model logical accounting includes unchanged Linear work, QK scores, softmax, Value accumulation, probability scanning, indexes, and logical Q4 Linear bytes.

```bash
bash experiments/exp_062/reproduce.sh
```

Evidence ceiling: Phase C observation. Physical attention-sparse kernels, 405B attention statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, and target hardware remain NOT TESTED.
