# Experiment 023 — Adaptive Exact-Row Branch-and-Bound

Evidence level: **E1 exact proof primitive / E2 pretrained LM-head gate**

## Motivation

A fixed hot top-K may contain the exact token, yet conservative bounds for unread
rows can remain above the exact refined winner. Increasing K blindly wastes
traffic on rows that cannot affect the decision.

This gate treats exact row reads as branch-and-bound refinements.

## Algorithm

```text
Q4 logits
  -> exact-refine initial top-K rows
  -> current exact winner
  -> find unread rows with upper_bound >= winner
  -> exact-refine only strongest ambiguous rows
  -> repeat until no unread row can win
```

The procedure always remains sound. When the refinement budget is exhausted it
abstains instead of committing an unproven token. Reading all rows necessarily
recovers the exact argmax, but would fail the traffic objective; the experiment
measures how early certification occurs.

## 405B cost

Each exact residual LM-head row costs:

```text
16,384 columns × 12 residual bits / 8 = 24,576 bytes
```

Therefore:

```text
32 rows    = 0.000732 GiB/token
1,024 rows = 0.023438 GiB/token
4,096 rows = 0.093750 GiB/token
```

The experiment allows at most 4,096 rows and requires p95 projected residual
traffic at or below 0.1 GiB/token.

## Sweep

- checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 32 exact-prefix tokens and exact hidden states
- Q4 LM head
- initial top-K: 32
- refinement batch: 32
- maximum refined rows: 4,096
- outside-bound column blocks: 16, 64, 256

## Promotion rule

```text
unsafe certificates == 0
certificate rate >= 95%
p95 projected 405B residual traffic <= 0.1 GiB/token
```

A passing LM-head result promotes the same branch-and-bound principle to
internal projections: propagate the exact top-two output margin backward, rank
residual tiles by their certified contribution, and read only tiles that can
still flip the final decision.
