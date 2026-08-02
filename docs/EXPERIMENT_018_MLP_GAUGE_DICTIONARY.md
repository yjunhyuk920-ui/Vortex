# Experiment 018 — exact-gauge-normalized SwiGLU neuron dictionary

Evidence level: **E2 real-model MLP replacement**

## Exact additional symmetry

For one SwiGLU neuron:

```text
d_j * SiLU(g_j x) * (u_j x)
```

is exactly unchanged by any positive scalar `s`:

```text
u_j -> u_j / s
d_j -> s * d_j
```

The baseline neuron dictionary clusters raw gate/up rows and therefore treats
up vectors with identical direction but different norm as different functions.
This experiment first chooses `s = ||u_j||`, absorbs the norm into the matching
down column, and clusters only the normalized up direction together with the
unaltered gate row.

Gate rows are not normalized because SiLU is not homogeneous.

## Contract

The gauge transform is numerically checked before clustering and uses only the
unchanged checkpoint weights. No prompt activations, labels, gradients,
fine-tuning, distillation or learned adapter are used.

All other protocol and 405B partial MLP budgets are identical to Experiment 017:

- Tiny 16 prototypes/layer -> target 128 prototypes/layer;
- Tiny 32 prototypes/layer -> target 256 prototypes/layer;
- attention, embeddings, norms and LM head remain exact;
- teacher-forced ranks, autonomous prefix and causal top-32 tree are measured.

## Decision

The gauge-normalized point must improve causal or teacher-forced preservation
relative to the baseline hard dictionary at the same prototype count. If the
exact symmetry does not produce a meaningful slope, hard prototype collapse is
rejected and the next candidate uses sparse prototype mixtures rather than one
assignment per neuron.
