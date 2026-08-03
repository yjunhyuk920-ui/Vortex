# EXP-062 — Pinned Causal Exact Non-Mask Attention-Probability Sparsity Gate

Authority: workflow `30844873182`, source `c38baa187e41760ef07676326c6a14f08635acc3`, merge `891868c186eb22869925ad20cba43ef32d371589`, artifact `8868287407`, ZIP SHA-256 `497816dcca7e6b8c40e9222ed8511efa266fe2358aab847a93795d7c04637390`.

MEASURED: 3 models; 18 cases; 1,152 forwards; 9,216 attention rows; mismatches 0; warm eligible probabilities 8,404,224; exact non-mask zeros 2,564; p50/p90 exact-zero fractions 0%/0.0753%; whole-model work 100.048%/100.154%; bytes 100.093%/100.303%; peak RSS 742,548 KiB.

Decision:

```text
REJECT_CAUSAL_EXACT_ATTENTION_PROBABILITY_SPARSITY_AS_CORE_RETAIN_ATTENTION_AUXILIARY
```

Exact non-mask attention-zero skipping is rejected for this measured population. Physical kernels, 405B, 8 GiB, and target hardware were not tested.
