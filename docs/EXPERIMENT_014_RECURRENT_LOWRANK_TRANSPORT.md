# Experiment 014 — resident recurrent dictionary with low-rank layer residuals

Evidence level: **E2 weight-only resident representation + causal path gate**

## Representation

Every original target matrix is represented as:

```text
W_l ~= diag(a_l) D_cluster diag(b_l) + U_l V_l.T
```

- `D_cluster` is one of three resident target matrices quantized to Q4.
- `a_l` and `b_l` are deterministic FP16 diagonal transports.
- `U_l` and `V_l` are deterministic randomized residual factors built only from
  checkpoint weights and stored at FP8.
- no activation data, labels, gradients, fine-tuning, distillation or learned
  adapter is used.

## 405B envelope

With untied embeddings and a 1 GiB workspace:

```text
three-layer Q4 dictionary + IO: 7.4102 GiB
all FP16 diagonal metadata:      about 0.0807 GiB
remaining FP8 residual capacity: rank 13 per matrix
rank 14: exceeds 8 GiB
```

The residual arithmetic is small relative to the original 405B dense compute.
Rank 13 remains inside the committed 1.2x native-4B compute proxy at the assumed
160 TOPS effective throughput.

## First sweep

Rank 1 and rank 4 are measured first to estimate the functional gain per resident
byte. Both points use the `uniform:nearest` dictionary because it was the best
plain recurrent schedule, preserving the exact target path for one token.

For each point:

1. generate exact TinyLlama reference decisions;
2. quantize the three representatives and IO to Q4;
3. fit diagonal transports for all original layer matrices;
4. fit FP8 randomized low-rank residual factors;
5. materialize the factorized representation only for numerical evaluation;
6. measure weight error, teacher-forced ranks, autonomous greedy prefix and a
   causal top-32 tree.

## Decision

The sweep is promoted toward rank 13 only if exact-prefix survival or top-32 path
survival improves monotonically enough to justify the additional resident bytes.
A path that still requires a 189 GiB target stream for every short block does not
satisfy the final runtime contract; exact certification remains a separate Gate.
