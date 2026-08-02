# Experiment 005 — 64-token block-shared adjoint repair

Evidence level: **E1 exact-target block oracle**

## Why this experiment exists

All tested per-token repair mechanisms are rejected on TinyLlama 1.1B:

- exact-span Atlas;
- exact layer-suffix repair;
- output-row tile repair;
- residual-energy 2D tiles;
- exact-target adjoint 2D tiles.

The strongest per-token oracle still required 512 MiB of exact O/down weights for one token-equivalent repair:

```text
E = 8.195999 tokens / full-model repair equivalent
Gate 0 minimum = 491.299160
```

The remaining path is to read one exact tile set once and apply it while solving a long proposed token block.

## Block accounting

For a selected exact tile set of `B` bytes shared by `A` committed tokens:

```text
rho = B / full_model_weight_bytes
E   = A / rho
```

Using the observed 512 MiB tile set from Experiment 004 on the 1.1B test model:

```text
minimum committed tokens for E >= 491.299160: 60
64 committed tokens would give E ~= 524.5
```

This is the first measured repair size for which block sharing can analytically enter the Gate 0 traffic envelope.

## Oracle protocol

`scripts/run_oracle_block_shared_adjoint.py` performs an intentionally optimistic test:

1. Generate the exact target continuation, up to 64 tokens.
2. Build rank-32 O/down capsules from disjoint prompts.
3. Use the exact target sequence and exact teacher-forced gradients to rank 128x128 residual tiles by positive logit-margin contribution per byte.
4. Select one fixed tile prefix.
5. Reuse that same tile set for the entire autoregressive continuation.
6. Measure the longest exact causal prefix committed before the first token divergence.
7. Compute `E` from committed prefix tokens and tile bytes read once for the block.

The experiment tests promotion, Gate 0, rejection, and larger tile prefixes.

## Pass condition

The logical traffic gate passes only when:

```text
committed causal-prefix tokens > 0
E >= 491.299160
```

Promotion requires:

```text
E >= 600
```

The result remains E1 because it uses exact target tokens and exact gradients to choose the tile set. A deployable runtime would still need to replace the oracle with a target-independent selector and a sound commit certificate.

## Rejection condition

Reject rank-32 block-shared adjoint repair when the best tested tile set remains below:

```text
E < 300
```

or when exact continuation requires a repair size whose compute per token breaks the 4B compute gate even if storage traffic is shared.

## Command

```bash
python scripts/run_oracle_block_shared_adjoint.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --block-tokens 64 \
  --build-new-tokens 1 \
  --max-rank 32 \
  --row-tile 128 \
  --col-tile 128 \
  --eval-prompt "한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘." \
  --output oracle_block_shared_adjoint.json
```

The GitHub Actions artifact name is:

```text
tinyllama-block-shared-adjoint
```
