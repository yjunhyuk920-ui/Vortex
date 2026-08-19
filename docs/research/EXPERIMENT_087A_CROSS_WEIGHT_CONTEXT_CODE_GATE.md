# EXP-087A — Cross-Weight Context Dictionary Gate

## Question

Can the aligned exact SwiGLU triplet

```text
T[i,j] = (W_gate[i,j], W_up[i,j], W_down[j,i])
```

be represented by a small checkpoint-static cross-weight generator plus a low-entropy XOR residual?

EXP-086A's fixed sign/exponent/MSB-prefix source retained about 95% of exact weight information. EXP-087A does not refine individual weight words. It predicts complete aligned triplets from other roles, previous triplets, or procedural coordinate classes and repairs every prediction exactly.

## Frozen population

```text
checkpoint   HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2
runtime      official LlamaForCausalLM, BF16/eager
layers       first, floor(middle), last
matrices     complete gate_proj, up_proj, down_proj
forwards     zero
periods      2,4,8,16,32
```

## Exact code families

- **One-source aligned table:** encode one role; a 16-bit oracle context predicts the other two words; exact two-lane XOR residual repairs every position.
- **Two-source aligned table:** encode an aligned pair; a sparse 32-bit context table predicts the third word; exact XOR residual repairs it.
- **Previous-triplet table:** literal first triplet per row; sparse 48-bit-key/48-bit-value table predicts the next triplet; exact three-lane residual repairs it.
- **Coordinate-tile generator:** procedural `(row mod p,column mod p)` context selects one oracle-majority triplet; exact residual repairs it.

## Favorable accounting

Source and residual streams receive ideal zero-order entropy coding. Probability tables, framing, arithmetic coder, perfect-hash overhead, address generation, decoder code, and kernel work are free. Literal seeds and context-table key/value bits are charged. Every candidate must reconstruct all three BF16 matrices exactly.

For `N` aligned positions, the denominator is

```text
N [H(W_gate) + H(W_up) + H(W_down^T)].
```

The code family has a real positive control: a periodic triplet field crosses the 20% Gate with a small procedural coordinate table and zero residual.

## Gates

```text
all reconstructions exact
best exact information fraction p50 <=20%
best exact information fraction p95 <=25%
```

Decisions:

```text
INVALID_CROSS_WEIGHT_CONTEXT_CODE_RECONSTRUCTION_FAILURE
REJECT_ALIGNED_CROSS_WEIGHT_CONTEXT_DICTIONARY_AS_COLD_QUERY_CORE
PROMOTE_CROSS_WEIGHT_CONTEXT_CODE_TO_FUSED_QUERY_GATE
```

## Stop rule

After rejection, do not sweep layers, coordinate periods, table width, residual operator, or thresholds. Reopening requires a sublinear cross-layer generator or a different causal transition dependency. A larger literal/context dictionary is not a new mechanism.

## Claim boundary

This is a no-forward exact checkpoint-description Gate. Even a pass would not prove reduced dense arithmetic, complete operation replacement, successor-state equality, CUDA performance, 405B execution, 8-GiB residency, or native-4B-class latency.
