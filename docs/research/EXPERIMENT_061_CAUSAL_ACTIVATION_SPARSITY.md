# EXP-061 — Pinned Causal Exact Activation-Sparsity Gate

Authority: workflow `30843404056`, source `15097a9b0323aa992679214173aaac0e7a98821c`, merge `44c3d6691d78714dc975e46e19bb8fdfe97a22cf`, artifact `8867731496`, ZIP SHA-256 `a01d31b012badd7d06087df576279b852db07813a0c7fb50d65c3a7283e9ca65`.

MEASURED: 3 models; 18 cases; 1,152 generated tokens; 147 registered projections; 56,448 calls; output/hook/control mismatches 0; exact activation zeros 0; warm p50/p90 fully accounted work 100.002%/100.391%; warm query bytes 100.004%/101.566%; peak RSS 761,248 KiB.

Decision:

```text
REJECT_CAUSAL_EXACT_ACTIVATION_SPARSITY_AS_CORE_RETAIN_RUNTIME_SPARSE_AUXILIARY
```

Exact activation-zero skipping is rejected for this measured population. Physical kernels, 405B, 8 GiB, and target hardware were not tested.
