# Next Experiment — EXP-105A

## Sub-Full-Head or Multi-Token-Per-Scan Transducer Gate

### Why this is next

EXP-104A proves that a full-vocabulary Q4 head plus even one complete 405B-width target layer already scans more resident weight bytes per proposed token than the final p50 budget allows. Longer accepted blocks do not amortize an autoregressive draft pass performed once per token.

The next mechanism must therefore change the draft execution unit, not merely select different target layers.

### Three new principles to compare locally

1. **Causal sub-full-head decision index.** Use current causal state to select a small exact set of LM-head rows to score. Charge the selector, index, candidate misses, fallback, row traffic, and verification. Reopening is allowed only if the information source differs materially from the rejected static top-k rank family.
2. **Single-scan multi-token resident transducer.** Read resident weights once and produce several causally ordered candidate tokens from an executable finite-word block state. This must not assume future hidden states or a free Jacobi fixed point.
3. **Cross-matrix decision-bit program.** Compute the next-token winner bits through a checkpoint-static program that shares work across layer projections and the vocabulary head, with all program bytes and online operations charged.

### Cheapest first Gate

For each principle freeze:

```text
online resident/cold bytes per committed token
actual causal information available
state and metadata bytes
fallback and miss rate
exact target verification path
explicit route below 2.4 GB/token p50-equivalent bytes
```

Reject before checkpoint execution unless one principle has an executable path below the p50 byte floor and does not belong to a closed mechanism family.

### Prohibited rescue attempts

```text
another full-head complete-layer slice
static top-k rows without a new causal source
future target hidden states
free HBM residency scans
configured K reported as accepted A
unmeasured sub-4-bit quality
Jacobi/Parareal retuning
```

After local validation, commit the result and verify the remote SHA. Do not rerun it in GitHub Actions unless explicitly requested.
