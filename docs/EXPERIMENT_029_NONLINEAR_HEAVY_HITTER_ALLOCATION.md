# Experiment 029 — Nonlinear Heavy-Hitter Damage Allocation

Evidence level: **E2 disjoint-prompt optimistic oracle**

## Why first-order allocation failed

Exact top-two margin adjoints revealed strong layer nonuniformity but did not
preserve nonlinear multi-layer behavior. At the 0.25% total fraction, top-32
improved from 43.75% to 56.25%, yet top-1 fell to zero and autonomous exact
prefix remained zero. At 0.50%, the adjoint allocation became worse than uniform.

The missing quantity is not a local gradient. It is the actual final-logit damage
caused by replacing a layer.

## Measured nonlinear damage curves

On a calibration prompt, keep every layer exact except layer `l`. Replace that
single MLP with the exact-activation original-neuron oracle at counts:

```text
1, 4, 8, 16, 32, 64 neurons
```

For each point, run the complete pretrained model and measure:

- exact-token cross entropy at the final logits;
- exact-token top-1/top-32 retention;
- sparse MLP output error and contribution coverage.

Damage is the nonnegative increase in final exact-token cross entropy over the
fully exact checkpoint.

## Discrete allocation

Each layer chooses one measured count. A dynamic program minimizes the sum of
measured nonlinear damages under a total exact-neuron budget. It may reuse a
cheaper point when a more expensive measured point is worse, but it may never
borrow the quality of a count it did not pay for.

## Disjoint validation

The damage curves are measured on an English database-concurrency prompt. The
chosen allocations are validated on a Korean sorting-algorithm prompt.

For each total fraction 0.10%, 0.25% and 0.50%:

- nonlinear and uniform allocations use exactly the same total selected-neuron
  count;
- all 22 MLPs are replaced simultaneously;
- teacher-forced exact-token rank and autonomous exact prefix are measured.

The per-token selector remains the optimistic full-activation oracle. Thus the
result is still an upper bound for a deployable runtime.

## Promotion rule

```text
projected 405B exact selected-neuron traffic <= 1.6 GiB/token
disjoint teacher top-32 >= 95%
disjoint autonomous exact prefix >= 4
```

A failure with no meaningful improvement closes the exact-neuron heavy-hitter
family. A substantial but insufficient improvement would justify a direct
interaction-aware allocation search, but not causal selector engineering yet.
