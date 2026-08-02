# Experiment 022 — Exact Top-K Row Decision Proof

Evidence level: **E1 exact proof primitive / E2 pretrained LM-head gate**

## Observation

The Q4 TinyLlama LM head retained the exact token in top-32 for all 32 tested
positions and matched exact top-1 on 30/32 positions. Unsigned bounds alone
certified zero positions because they had to cover close top competitors and all
vocabulary rows simultaneously.

## Candidate

For each token:

1. compute all Q4 LM-head logits;
2. select the hot top-K rows;
3. read only those K rows' omitted residual values and compute their logits
   exactly;
4. choose the exact winner among the refined rows;
5. use resident row/block norm metadata to upper-bound every unread outside row;
6. commit only when the exact selected winner exceeds every outside upper bound.

This remains sound when the true winner is absent from hot top-K: its outside
upper bound blocks certification.

## 405B LM-head incremental cost

With hidden size 16,384, FP16 source weights and Q4 hot rows, the omitted
residual is 12 bits/element. For K=32:

```text
32 × 16,384 × 12 / 8 = 786,432 bytes/token
```

This is below 0.001 GiB/token. Row/block metadata at 16-column blocks is larger
than the unsigned experiment but still a small fraction of 8 GiB. The Q4
LM-head itself is also only a partial architecture component; this gate does not
claim a complete 405B runtime.

## Exactness contract

Let `S` be the selected hot top-K rows. For each `r in S`, its exact logit is
computed. For each `j not in S`, only a conservative upper bound is used.

```text
winner = argmax_{r in S} exact_logit(r)

certify iff
exact_logit(winner) > max_{j not in S} upper_bound(j)
```

Any certified winner different from the exact FP16 argmax is an implementation
failure.

## Sweep

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- exact hidden states and authoritative 32-token continuation
- hot LM head: Q4
- top-K: 1, 4, 16, 32
- outside metadata: 16-column residual blocks

## Promotion rule

Advance to internal adjoint-guided exact tile refinement only when:

```text
unsafe certificates == 0
certificate rate >= 95%
projected 405B exact residual read < 0.01 GiB/token
```

If outside bounds remain too loose, retain exact top-K refinement and replace
unsigned outside bounds with directional or hierarchical support metadata.
