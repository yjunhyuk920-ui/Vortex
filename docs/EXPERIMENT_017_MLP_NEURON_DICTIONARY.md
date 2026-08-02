# Experiment 017 — permutation-invariant SwiGLU neuron dictionary

Evidence level: **E2 real-model MLP replacement**

## Structural symmetry

SwiGLU neurons inside one decoder layer may be permuted exactly when the same
permutation is applied to:

- gate projection rows;
- up projection rows;
- down projection columns.

The previous matrix factorizations ignored this exact symmetry. This experiment
removes neuron ordering before approximation.

## Compiled nonlinear program

Gate/up row pairs define each neuron's input function. They are clustered within
each layer into a small prototype set. For every cluster:

```text
prototype gate row = mean assigned gate rows
prototype up row   = mean assigned up rows
prototype down col = sum original assigned down columns
```

At runtime:

```text
g = G_proto @ x
u = U_proto @ x
y = D_aggregated @ (SiLU(g) * u)
```

If assigned gate/up rows are identical, the compilation is exactly equivalent
regardless of neuron ordering. Down columns are never individually approximated;
they are summed exactly after assignment.

No prompt activation, token label, gradient, fine-tuning or learned adapter is
used. Clustering is deterministic checkpoint compilation.

## 405B partial budget

Two target points are tested:

```text
128 prototypes/layer: about 0.738 GiB MLP factor traffic/token
256 prototypes/layer: about 1.477 GiB MLP factor traffic/token
```

The native-4B traffic envelope is about 2.835 GiB/token, leaving respectively
about 2.10 GiB or 1.36 GiB for attention, embeddings, LM head, KV and runtime
metadata. This is intentionally a partial-family Gate; it does not claim a
complete architecture.

## TinyLlama mapping

```text
Tiny 16 prototypes/layer -> target 128 prototypes/layer
Tiny 32 prototypes/layer -> target 256 prototypes/layer
```

TinyLlama receives a slightly more favorable neurons-per-prototype ratio than
the projected target.

## Protocol

1. Generate exact TinyLlama reference logits and tokens.
2. Compile every one of the 22 SwiGLU MLPs into a neuron dictionary.
3. Replace the original MLP modules in the active model graph.
4. Leave attention, embeddings, normalization and LM head exact.
5. Measure teacher-forced token ranks, autonomous greedy prefix and a causal
   top-32 tree.

## Promotion rule

A point advances only when:

```text
partial MLP memory/traffic/latency Gate passes
teacher-forced exact-token top-32 coverage >= 95%
causal tree preserves at least the first exact target token
```

Because the rest of the model remains exact, failure directly rejects the tested
neuron-collapse ratio. Passing only establishes MLP viability; attention and
LM-head representations remain separate mandatory Gates.
