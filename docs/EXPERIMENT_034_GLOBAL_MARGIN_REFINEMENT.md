# Experiment 034 — Global Margin Refinement

## Failure addressed

Experiments 031–033 assigned each layer an equal absolute-error share:

```text
tau_layer = tau_total / layers
```

This is sufficient but unnecessarily restrictive. A difficult layer could consume more budget while an easy layer leaves slack unused.

This experiment performs one cross-layer exact-refinement allocation.

## Two-sided uncertainty

For each unrefined neuron interval `[L_i,U_i]` with approximate center `a_i`:

```text
lower uncertainty l_i = a_i - L_i
upper uncertainty u_i = U_i - a_i
```

Exact refinement replaces all three values by the exact scalar, removing both `l_i` and `u_i`.

After selecting exact set `S`, the global fixed-dual error target is satisfied when:

```text
sum_{i not in S} l_i <= tau
sum_{i not in S} u_i <= tau
```

where:

```text
tau = 0.5 * exact top-two margin
```

This is a unit-cost two-constraint cover problem.

## Dual-price sweep

For dual price `lambda in [0,1]`, order neurons by:

```text
score_i(lambda) = lambda l_i + (1-lambda) u_i
```

Take the shortest prefix satisfying both constraints. Sweep 41 prices including `lambda=0.5`, which reproduces ordinary interval-width ordering, and retain the smallest feasible prefix.

Therefore the dual-price result is never worse than the previous global width ordering. It is also compared with the equal-per-layer allocation.

## Residual representation

The experiment uses the strongest static code from Experiment 033:

```text
8-bit hot weights
block size 1024
rank 2 signed residual basis
float32 coefficient/remainder/basis metadata
metadata = 3.6299 GiB projected at 405B
```

Build prompts and evaluation prompts remain disjoint.

## Promotion conditions

For every evaluation prompt:

```text
dual-price certificate contains the exact scalar
unsafe certificates = 0
global target error is met
metadata <= 6 GiB
projected exact refinement <= 1.6 GiB/token
```

This remains an optimistic fixed-dual linearized E2 Gate. The sum of local exact-dual contributions is not yet a complete nonlinear token proof.

## Decision rule

- A pass advances global refinement to causal dual transport.
- A large improvement that still misses the traffic Gate justifies semantic-state-keyed or online codebooks.
- Little improvement closes static signed residual codes and global allocation together.
