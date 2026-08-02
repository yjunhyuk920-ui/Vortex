# Experiment 024 — Causal Activation Proof Atlas

Evidence level: **E1 sound primitive / E2 pretrained causal LM-head gate**

## Candidate

A prompt-prefill session provides causally available hidden-state vectors. Build
an orthonormal basis `U` from only those prompt activations and compile exact
residual images for the Q4 LM head:

```text
C = R U
```

For a continuation activation `x`:

```text
x_parallel = U(U^T x)
x_perp     = x - x_parallel
z_refined  = z_Q4 + C(U^T x)
```

Residual correction is exact on the prompt span. The unread remainder is
certified with:

```text
|E[row] x| <= ||E[row]||_2 ||x_perp||_2
```

No continuation token or future hidden state participates in basis construction.

## Why this differs from the rejected global sketch

The global residual SVD retained directions with large residual energy but rank
64 removed only 2.7% of total residual energy and certified zero tokens. This
atlas instead optimizes the query geometry: even a high-energy residual is
harmless when continuation activations remain close to the prompt span.

## Sweep

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- long technical prompt with at least 16 prompt positions
- basis ranks: 4, 8, 16
- basis source: last 32 prompt-prefill hidden states only
- evaluation: 32 exact continuation hidden states
- hot LM head: Q4

## Promotion rule

```text
unsafe certificates == 0
certificate rate >= 50%
projected 405B LM-head atlas metadata < 0.05 GiB
```

A passing point promotes online proof atlases to internal projections. A failing
point records continuation span drift and rejects the tested prompt-only rank,
not the exactness contract.
