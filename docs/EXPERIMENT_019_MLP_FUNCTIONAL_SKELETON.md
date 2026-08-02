# Experiment 019 — SwiGLU functional skeleton

Evidence level: **E2 real-model MLP replacement**

## Why hard neuron assignment may be too weak

Experiments 017 and 018 assign every original SwiGLU neuron to one prototype.
Even after exact up/down gauge normalization, one centroid must represent a
large set of nonlinear functions.

This experiment keeps the same runtime storage but replaces hard assignment
with an interpolative functional skeleton.

## Compile algorithm

For each layer:

1. apply the exact up/down gauge normalization;
2. generate deterministic synthetic RMS-one Rademacher probes;
3. evaluate every original neuron function
   `SiLU(g_j x) * (u_j x)` on those probes;
4. greedily select actual original neuron response columns by residual pivoting;
5. solve a regularized linear interpolation of every neuron response from the
   selected responses;
6. absorb the interpolation matrix into the original down projection.

Runtime stores only:

```text
selected gate rows
selected normalized up rows
aggregated down columns
```

The element count is identical to the hard dictionary:
`3 * prototypes * hidden` per layer.

No user prompts, datasets, labels, gradients, fine-tuning, distillation or
learned adapter are used. Synthetic probe compilation is deterministic linear
algebra over the checkpoint function.

## Points

```text
Tiny 16 selected functions/layer -> target 128 functions/layer
Tiny 32 selected functions/layer -> target 256 functions/layer
```

Each layer uses 256 compile probes and 128 disjoint synthetic held-out probes.
Function-space activation and final MLP-output reconstruction errors are
reported separately from token-level model behavior.

## Model protocol

- replace all 22 TinyLlama MLP modules;
- keep attention, embeddings, norms and LM head exact;
- measure teacher-forced exact-token ranks;
- measure autonomous greedy exact prefix;
- build a causal top-32 tree.

## Promotion rule

A point advances only when the partial MLP hardware Gate passes and:

```text
teacher-forced exact-token top-32 coverage >= 95%
causal tree preserves at least the first exact target token
```

If function-space held-out error is low but token behavior fails, the issue is
layerwise error accumulation and the next Gate adds exact per-layer residual
certificates. If held-out function error itself remains high, the required
prototype count is incompatible with the fixed traffic envelope and this MLP
family is rejected.
