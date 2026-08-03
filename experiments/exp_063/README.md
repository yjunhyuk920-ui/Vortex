# EXP-063

Pinned causal exact cached-KV equivalence measurement.

For every layer/head and causal warm-decode step, the runner groups bit-identical cached Key vectors and bit-identical Key-Value pairs. Exact K duplicates may reuse QK scores; exact KV duplicates may additionally reuse probability-times-Value products while retaining source-order Value additions.

Incremental new-vector hash scanning, score copies, group metadata, representative cache reads, unchanged softmax/additions, and all unchanged Linear work/bytes are charged.

```bash
bash experiments/exp_063/reproduce.sh
```

Evidence ceiling: Phase C observation. Physical grouped-attention kernels, 405B KV equivalence statistics, actual Transformer operation replacement, 405B execution, 8 GiB VRAM, and target hardware remain NOT TESTED.
