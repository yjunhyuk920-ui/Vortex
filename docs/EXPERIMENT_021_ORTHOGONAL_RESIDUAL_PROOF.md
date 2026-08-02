# Experiment 021 — Orthogonal Residual Proof Sketch

Evidence level: **E1 proof primitive / E2 pretrained LM-head gate**

## Why unsigned norms failed

Experiment 020 preserved all 32 exact tokens inside the Q4 LM-head top-32 and
matched top-1 on 30/32 positions, but certified zero decisions. Row/block L2
norms discarded the residual direction, so the worst-case interval was wider
than the observed logit margin at every position.

The conclusion is not that the Q4 proposal lacks information. It is that an
unsigned residual envelope is too loose.

## Candidate

Decompose the omitted residual as

```text
R = C U^T + E
```

where `U` is a small shared orthonormal input basis and `C = R U`. Runtime
computes the retained directional correction exactly:

```text
z_refined = z_hot + C (U^T x)
```

Because `E U = 0`, only the activation component perpendicular to the retained
basis affects the unread remainder:

```text
x_perp = x - U(U^T x)
|E[r] x| = |E[r] x_perp| <= ||E[r]||_2 ||x_perp||_2
```

The exact residual values in `E` remain unread. The runtime stores only:

- shared basis `U`;
- output coefficients `C`;
- one remainder norm per output row.

## Exact decision contract

For refined candidate `c`:

```text
lower(c) = z_refined[c] - ||E[c]|| ||x_perp||
upper(j) = z_refined[j] + ||E[j]|| ||x_perp||
```

Commit only when:

```text
lower(c) > max_{j != c} upper(j)
```

Any accepted token different from the exact FP16 argmax is a soundness failure.
No confidence threshold or empirical calibration can override this contract.

## 405B LM-head metadata projection

For vocabulary 128,256 and hidden size 16,384, rank 32 FP32 metadata is below
0.02 GiB. Rank 64 remains far below the 8 GiB device limit. This first gate uses
exact hidden states and therefore proves only the output-layer mechanism, not a
complete model runtime.

## Pretrained sweep

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- authoritative continuation: 32 greedy tokens
- hot LM head: per-row Q4
- sketch ranks: 4, 16, 64
- exact hidden states isolate the LM-head proof
- metrics: refined top-1/top-32, exact certificate rate, unsafe certificates,
  remainder energy and metadata bytes

## Promotion rule

Advance the architecture only when:

```text
unsafe certificates == 0
certificate rate >= 50%
```

If certification rises monotonically but stays below 50%, sweep a larger rank
under an explicit metadata/compute budget. If rank 64 provides no meaningful
increase, reject global orthogonal sketches and move to adaptive signed tile
support functions for only the hot top-K decision rows.
