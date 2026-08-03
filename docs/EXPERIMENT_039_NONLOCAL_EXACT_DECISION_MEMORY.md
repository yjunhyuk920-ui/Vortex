# Experiment 039 — Nonlocal Exact Decision Memory

Last updated: 2026-08-03 (Asia/Seoul)

## Evidence level and scope

This is an E1/E2 optimistic representation-capacity gate. It does not prove a complete runtime.

The fixed objective remains an arbitrary unmodified Hugging Face dense transformer, one 8 GiB GPU, original-model decisions, and p50 warm decode within 1.2x a native 4B Q4 baseline. Only a real 405B end-to-end run is E4.

PRs #36–#38 rejected tokenwise state objects:

- selected semantic programs changed about every token;
- prompt-compiled Hankel dynamics preserved at most two future tokens;
- an impossible perfect-token repair oracle still executed the exact target on 68%–89% of tokens.

Experiment 039 therefore stops extrapolating hidden state. It asks whether an exact future decision block already seen at a nonlocal prompt position can be replayed as a unit.

## Representation

For every prompt position `i` with at least one following prompt token, build one memory entry:

```text
key_i   = normalize(P^T (h_i - mean_prompt_hidden))
block_i = prompt_token_ids[i+1 : i+1+L]
```

where `P` is built only from prompt hidden states. No continuation token or continuation hidden state is used to build keys, blocks, centering statistics, or the projection.

At a held-out block boundary, one exact target interaction emits an anchor token and its exact final hidden state. That interaction is charged to the block. The lookup key is then:

```text
query_t = normalize(P^T (q_t - mean_prompt_hidden))
```

where `q_t` is the exact hidden state after consuming the anchor or a later exact committed token. Replayed targets begin strictly after the anchor.

The deployable diagnostic returns the nearest prompt entry by cosine similarity. Stronger upper bounds are also measured:

1. best exact-prefix match among the top-k nearest entries;
2. best exact-prefix match among every prompt entry, ignoring hidden retrieval entirely.

The global token oracle sees the future continuation only for evaluation. It is impossible to deploy, but it is the strongest possible upper bound for this representation. If it cannot find a 247-token prompt suffix matching a continuation suffix, no key rank, ANN index, or router can make prompt-only exact token-block memory succeed.

## Alignment contract

`h_i` is the final hidden state after prompt token `i` and predicts token `i+1` through the original LM head. The final prompt state produces `continuation_token_ids[0]`, which is the exact boundary anchor and is not counted as replay. `continuation_hidden_states[0]` is the final hidden state after consuming that anchor and predicts `continuation_token_ids[1]`, the first replay target. Therefore the 256 evaluation pairs are:

```text
query_t  = continuation_hidden_states[t]
target_t = continuation_token_ids[t+1]
for t = 0..255
```

The final prompt position remains excluded from memory because its following token is the held-out anchor.

## 405B memory and lookup equations

For:

```text
E = memory entries
r = key rank
L = stored token-block length
b_k = key bits
b_t = token-id bits
H = target hidden size
```

resident storage is:

```text
M_keys   = E * r * b_k / 8
M_blocks = E * L * b_t / 8
M_index  = index_overhead_fraction * (M_keys + M_blocks)
M_total  = M_keys + M_blocks + M_index
```

A conservative brute-force lookup including full-hidden projection is:

```text
C_lookup = 2 * H * r + 2 * E * r FLOP/query
```

At `E=65,536`, `r=128`, `L=256`, `b_k=16`, `b_t=32`, and 25% index overhead:

```text
M_keys   = 16 MiB
M_blocks = 64 MiB
M_index  = 20 MiB
M_total  = 100 MiB
C_lookup = 20.97 MFLOP/query = 0.02097 GFLOP/query
```

Thus this family is not rejected by metadata size or lookup arithmetic. It lives or dies on exact future-block recurrence.

One optimistic exact block-boundary interaction is charged as the previously established 405B lower-bound proxy:

```text
B_boundary = 188.9883 GiB
C_boundary = 811.6985 GFLOP
```

The inherited strong amortization requirement is:

```text
accepted exact block length >= 247 tokens after the anchor
```

## Real-model gate

Use the same three long TinyLlama prompts as Experiments 037–038. Charge the exact first continuation token as the boundary anchor, then evaluate the following 256 greedy decisions.

For key ranks `8, 16, 32, 64, 128, 256`, measure:

- nearest-entry exact-prefix length for every continuation query;
- top-4, top-16, and top-64 oracle exact-prefix lengths;
- global future-token-oracle exact-prefix lengths;
- first-query, maximum, mean, p95, and coverage at lengths 1/4/16/64/247;
- post-anchor EOS position, maximum identical-token run, and unique-token fraction;
- actual-prompt and 65,536-entry projected memory/lookup budgets.

## Promotion and rejection

Representation capacity advances only if, on every prompt:

```text
global oracle maximum exact prefix >= 247
no trivial post-anchor EOS before token 247
maximum identical-token run <= 16
unique-token fraction >= 0.05
```

A key configuration advances to a deployable retrieval follow-up only if the same rank also reaches:

```text
nearest-state maximum exact prefix >= 247
M_total + M_KV + M_work <= 8 GiB
lookup arithmetic within the 4B-class envelope
```

Immediate rejection:

- the global oracle fails 247 on any prompt;
- a result depends on continuation data during memory build;
- the boundary anchor is incorrectly counted as replay;
- success comes only from EOS or a repeated-token loop;
- reported storage omits blocks or index overhead.

## Falsification interpretation

A global-oracle failure is stronger than a nearest-neighbor failure. It closes prompt-only exact token-suffix memory regardless of signature rank or indexing method. Do not respond by tuning ANN parameters, adding more static key ranks, or changing distance metrics unless a new source of nonlocal exact decision content is introduced.
