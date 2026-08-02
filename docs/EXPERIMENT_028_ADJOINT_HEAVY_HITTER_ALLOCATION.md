# Experiment 028 — Adjoint Heavy-Hitter Allocation

Evidence level: **E2 disjoint-prompt optimistic oracle**

## Motivation

Uniform exact-neuron fractions failed. At the traffic-compatible 0.25% point,
the perfect token-dependent oracle preserved only 43.75% top-32 and no
autonomous prefix. However, equal allocation assumes every Transformer layer has
the same effect on the final token decision.

This gate keeps the same total exact-neuron bytes and changes only where those
neurons are spent.

## Calibration utility

On a calibration prompt, freeze every checkpoint parameter and retain gradients
of each MLP down-projection output. For the exact continuation, define the sum of
exact top-one versus runner-up logit margins. For neuron `i` in layer `l`:

```text
a_li = SiLU(g_li x) * (u_li x)
q_li = d_li^T (partial margin / partial MLP_output_l)
utility_li = sum_positions |a_li * q_li|
```

Each original neuron has the same exact row/column byte cost. After reserving one
neuron per layer, selecting the globally largest utilities is therefore the
unit-cost knapsack optimum for this measured first-order margin objective.

## Disjoint evaluation

Calibration and evaluation prompts are different. Only per-layer neuron counts
are transferred from calibration. On the evaluation prompt:

- a uniform allocation and the adjoint allocation receive exactly the same total
  number of selected neurons;
- both use the same optimistic exact-activation heavy-hitter oracle inside each
  layer;
- teacher-forced exact-token ranks and autonomous exact prefix are compared.

The test does not learn weights, use labels or modify the checkpoint. It is still
an oracle because full gate/up activations select neurons during evaluation.

## Sweep

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- total fractions: 0.10%, 0.25%, 0.50%
- calibration continuation: 4 tokens
- disjoint evaluation continuation: 16 tokens
- minimum allocation: one exact neuron per layer
- exact total bytes identical between uniform and adjoint allocations

## Promotion rule

A point can promote only when:

```text
projected 405B MLP traffic <= 1.6 GiB/token
disjoint teacher top-32 >= 95%
disjoint autonomous exact prefix >= 4
```

A positive but insufficient improvement motivates a more exact nonlinear damage
allocator. No improvement rejects layer nonuniformity as the missing factor and
closes the exact-neuron heavy-hitter family.
