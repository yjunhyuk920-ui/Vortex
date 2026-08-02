# Experiment 025 — Proof-Triggered Online Atlas Expansion

Evidence level: **E1 sound online primitive / E2 pretrained LM-head amortization gate**

## Why this is the final Atlas-family gate

A static prompt-only atlas failed because continuation hidden states were almost
orthogonal to the prompt span. Rank 16 still left a mean perpendicular ratio of
0.93395 and certified zero decisions.

The remaining Atlas hypothesis is online reuse:

```text
proof succeeds
  -> commit exact decision without reading residual values

proof fails
  -> append the current causal activation direction
  -> stream the omitted residual once to compute R u
  -> update exact residual images and remainder norms
  -> prove and commit the current decision
  -> reuse that image on later tokens
```

This gate measures whether one exact residual stream serves many future tokens.
It does not hide expansion cost behind storage or preprocessing.

## Exact update

Let the current atlas basis be `U`, and let the current activation's orthogonal
component be:

```text
p = x - U(U^T x)
u = p / ||p||
```

One expansion stores the exact residual image:

```text
c = R u
```

Because `u` is orthogonal to the previous basis, the unread remainder row norm
updates exactly by Pythagoras:

```text
new_norm[row]^2 = old_norm[row]^2 - c[row]^2
```

The current activation is then in the expanded span, so its residual correction
is exact up to floating-point error. No unproven token is committed.

## 405B LM-head expansion cost

For vocabulary 128,256, hidden size 16,384, FP16 source and Q4 hot weights, one
complete LM-head residual image requires reading:

```text
128,256 × 16,384 × 12 / 8 ≈ 2.94 GiB
```

Across a 32-token window:

```text
0 expansions = 0 GiB/token
1 expansion  ≈ 0.092 GiB/token
2 expansions ≈ 0.184 GiB/token
32 expansions ≈ 2.94 GiB/token
```

This is only the LM-head component. Promotion therefore requires no more than
one expansion per 32 tokens under the fixed 0.1 GiB/token partial budget.

## Pretrained protocol

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- prompt-only initial atlas ranks: 4, 8, 16
- prompt build samples: 32
- exact continuation: 32 greedy tokens
- hot LM head: Q4
- on each proof miss: append the current exact hidden direction
- record pre-expansion proof rate, post-expansion proof rate, expansion count,
  rank growth, reuse tokens per expansion and projected 405B residual traffic

## Promotion rule

```text
unsafe certificates == 0
post-expansion certificate rate == 100%
amortized 405B LM-head residual traffic <= 0.1 GiB/token
```

If expansions occur nearly once per token, the online Atlas family is rejected:
it would reproduce full residual streaming rather than amortize it. A pass would
promote proof-triggered expansion to selected internal Transformer projections,
where every exact image must be charged against the complete model traffic gate.
