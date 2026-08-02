# Experiment 006 — block-shared traffic and compute gate

Evidence level: **E1 exact-target block oracle**

## Correction to Experiment 005

A shared tile set changes storage traffic and arithmetic differently.

If a selected exact weight subset is streamed once for a block of `A` committed tokens, its storage traffic is amortized:

```text
traffic efficiency E = A / rho
rho = selected weight bytes / full model weight bytes
```

However, those selected exact weights must still be multiplied by every token activation. The arithmetic cost is therefore:

```text
C/token = C_hot + rho * C_full_exact
```

It is **not** divided by `A`.

The earlier observation that 512 MiB shared over 64 tokens crosses the traffic gate is true but insufficient. At the 405B target ratio, that selected fraction would add more than 100 GFLOP/token and fail the 4B-class compute limit.

## Combined target equations

The block candidate must satisfy both:

```text
A / rho >= 491.2991599793
```

and:

```text
3.531515136 + rho * 845.521354752 <= 12.1327352832 GFLOP/token
```

The compute equation limits the exact selected fraction to roughly 1.02% of the full target computation. On the TinyLlama test model this corresponds to approximately 43 MiB of selected O/down weights, even for a 64-token block.

## Oracle protocol

`scripts/run_oracle_block_shared_gate.py`:

1. generates an exact continuation up to 64 tokens;
2. builds rank-32 O/down capsules using disjoint prompts;
3. uses the exact target continuation and exact teacher-forced adjoints to rank 128x128 residual tiles;
4. selects one fixed tile prefix for the entire block;
5. measures the exact causal-prefix length before the first token divergence;
6. charges selected weight bytes once for traffic;
7. charges selected exact tile FLOPs for every token;
8. evaluates traffic and compute gates together.

## Promotion rule

A candidate advances only when one tested shared tile set has:

```text
committed causal prefix > 0
traffic gate = pass
compute gate = pass
```

The experiment is still an optimistic upper bound because exact target tokens and exact gradients choose the tile set. A deployable runtime must replace those oracles and retain the same resource envelope.

## Command

```bash
python scripts/run_oracle_block_shared_gate.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --block-tokens 64 \
  --build-new-tokens 1 \
  --max-rank 32 \
  --row-tile 128 \
  --col-tile 128 \
  --eval-prompt "한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘." \
  --output oracle_block_shared_gate.json
```
