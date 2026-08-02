# Experiment 027 — Certified Heavy-Hitter Dense MLP Oracle

Evidence level: **E2 pretrained optimistic oracle**

## Why this family is different

Previous structural compilers replaced every token's dense MLP with the same
small approximation. They lost model behavior. This candidate keeps original
checkpoint neurons and chooses a different exact subset for every activation.

For SwiGLU neuron `i`:

```text
a_i = SiLU(g_i x) * (u_i x)
y   = sum_i d_i a_i
```

The oracle contribution score is:

```text
score_i = |a_i| * ||d_i||_2
```

The top-scoring original neurons are retained exactly and the rest are omitted.
No centroid, factor or synthetic neuron replaces them.

## Important oracle boundary

The first gate intentionally computes full exact gate/up activations before
selection. Therefore it is not fast. It answers a decisive upper-bound question:

> Even with perfect token-dependent knowledge, can the exact decision be
> preserved by a small enough original-neuron subset?

A causal compact selector can never outperform this oracle at the same selected
fraction.

## 405B partial MLP traffic

One exact selected neuron requires:

```text
one gate row + one up row + one down column
= 3 * hidden_size source values
```

At FP16 for the 405B-class dimensions:

- 0.10% of neurons: roughly 0.62 GiB/token;
- 0.25%: roughly 1.55 GiB/token;
- 0.50%: roughly 3.1 GiB/token.

The partial MLP traffic gate is 1.6 GiB/token, so fractions above about 0.25%
are quality diagnostics rather than viable complete-runtime points. Attention,
KV, embeddings, selector metadata and proof costs remain separate.

## Pretrained protocol

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- replace all 22 MLPs with the oracle module;
- selected fractions: 0.10%, 0.25%, 0.50%, 1.0%, 2.0%;
- exact continuation: 16 greedy tokens;
- measure teacher-forced exact-token rank and autonomous exact prefix;
- record selected score coverage, exact MLP output error and unique neuron reuse.

The oracle computes both sparse and full down outputs for diagnostics, but only
the sparse output is returned to the Transformer.

## Promotion rule

Advance to a real runtime selector only when:

```text
projected exact selected-neuron traffic <= 1.6 GiB/token
teacher-forced top-32 >= 95%
autonomous exact prefix >= 4 tokens
```

A passing point still does not solve the model. It promotes two mandatory next
gates:

1. predict the oracle subset from compact resident signatures before loading
   exact rows;
2. certify that the omitted neuron tail cannot flip the final token decision.

If the perfect oracle needs more than the traffic-compatible fraction, the
heavy-hitter family is rejected before selector engineering.
