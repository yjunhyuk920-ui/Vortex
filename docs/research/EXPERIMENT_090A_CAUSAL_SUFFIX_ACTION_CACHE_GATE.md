# EXP-090A — Causal Suffix-Action Cache Gate

## Status

```text
FROZEN BEFORE PUBLIC-CHECKPOINT RESULT
Phase: C cheapest decisive drafting/source Gate
Evidence ceiling: E3 causal held-out drafting signal
TARGET-W execution / 8 GiB measurement / 4B-class latency: NOT TESTED
```

## 1. Why this experiment exists

EXP-088B rejected exact retain/zero page programs: only the complete two-layer checkpoint was exact. Corrected EXP-087A rejected a compact quadratic BF16 MLP generator: it interpolated all build vectors and produced zero exact unseen vectors. EXP-089A then established a different state contract:

```text
behavioral compiled state = exact token prefix + exact RNG bytes
```

For the pinned checkpoint and ABI, official replay reconstructed every measured logit and complete KV byte exactly. Therefore an accelerated path does not need to make an approximate draft hidden state equal to the target hidden state. It may instead:

1. produce candidate tokens causally from the exact prefix;
2. run an unchanged exact target block verifier;
3. commit the accepted exact token prefix;
4. retain the verifier's official successor state.

EXP-090A tests the cheapest high-upside causal source for a long candidate chain before any block verifier, CUDA kernel, or 405B implementation is built.

## 2. Frozen mechanism

The draft keeps one complete first decoder layer in BF16 and a row-wise symmetric Q4 copy of the LM head. An exact prefill supplies a bounded history of already verified suffix effects. During the speculative block the history is frozen; no target result from that block may enter the draft.

For decision state `t`, let:

```text
h_t^(1)  = hidden state after the first official decoder layer
q_t      = Q4Head(RMSNorm(h_t^(1)))
l_t*     = official final target logits
c_t      = BF16(l_t* - q_t)
```

`c_t` is called the **suffix action**. It combines the effect of every omitted decoder layer, the exact final norm/head difference, and Q4-head error at one already verified state.

The draft score is:

```text
s_t = q_t + c_hat_t
```

and the candidate token is the deterministic argmax of `s_t`.

## 3. Frozen predictor language

The compiler evaluates exactly four modes, in the preregistered order.

### `raw_q4`

```text
c_hat_(t+j) = 0
```

### `carry_bf16`

```text
c_hat_(t+j) = c_(t-1)
```

The most recent already verified suffix action is carried through the block.

### `secant_bf16`

```text
c_hat_(t+j) = BF16(c_(t-1) + (j+1)(c_(t-1) - c_(t-2)))
```

This is a finite-word causal extrapolation. Both source actions precede the speculative block.

### `nearest_history_bf16`

For a frozen history `H` of verified shallow states:

```text
i* = smallest argmax_i cosine(h_(t+j)^(1), h_i^(1))
c_hat_(t+j) = c_i*
```

The tie rule is fixed. The query is the current draft shallow state; no target state, target token, prefix hash table, or response table is used.

## 4. Causal order

For every case the runner must execute in this order:

1. official prompt prefill;
2. derive verified prompt-history `(h_i^(1), c_i)` and the exact first generated token;
3. freeze the history;
4. generate the complete 128-token draft chain using only the first layer, Q4 head, frozen history, and earlier draft tokens;
5. only after the draft exists, execute the official target continuation and compare it with the candidate chain.

Build target continuations may choose one of the four frozen modes. Holdout target continuations are not executed until one mode has been selected and frozen from the build population.

## 5. Exactness boundary

The candidate path itself is approximate. Exactness comes from ordinary lossless target verification:

```text
accepted_length = longest common prefix(candidate_tokens, target_tokens)
```

A future block executor may commit only accepted target tokens and the official target cache returned by verification. Consequently, wrong drafts cannot silently alter the declared output/state contract.

This Gate does not implement that physical verifier. It measures whether a candidate source exists that is strong enough to justify implementing one.

## 6. Public-checkpoint population

```text
DEV-W       HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
runtime     transformers 4.46.3 / torch 2.5.1+cpu
reference   official eager BF16 LlamaForCausalLM
shallow     first complete decoder layer
block       128 candidate tokens
history     at most 64 verified prompt positions
```

Build cases: English, Korean, and code. Holdout cases: mathematics, structured JSON, and repeated patterns. Prompts and mode order are frozen in `experiments/exp_090a/config.json`.

## 7. Measurements

For every mode and build case:

```text
candidate token sequence
exact target token sequence
accepted prefix length
first mismatch position
target-token rank under the causal score at each true-path state
top-1 true-path agreement
history index chosen by nearest-history mode
candidate generation calls
exact target calls
```

After build selection, the same measurements are made for the selected mode on untouched holdout cases.

True-path ranks after a chain mismatch are an explicitly labeled favorable oracle diagnostic for a possible candidate-tree continuation. They do not change the already generated candidate chain and do not count as deployable acceptance.

## 8. Build selection

The selected mode maximizes, lexicographically:

```text
minimum accepted length across build cases
sum of accepted lengths
negative p95 final-token rank
negative maximum final-token rank
frozen mode-order preference
```

No threshold, prompt, history size, depth, Q4 scheme, or mode is changed after observing build or holdout results.

## 9. Target hot-state equation

The optimistic Llama-3.1-405B projection freezes:

```text
D=16,384
I=53,248
Hq=128
Hkv=8
head_dim=128
V=128,256
Lmax=131,072
history=64
```

One BF16 decoder layer contains:

```text
N_layer = 2D^2 + 2D(Hkv*head_dim) + 3DI
        = 3,187,671,040 parameters
B_layer = 6,375,342,080 bytes
```

The favorable resident equation is:

```text
B_hot = B_layer_BF16
      + ceil(V*D/2)                 # packed Q4 head
      + 2V                          # BF16 row scales
      + 2*history*V                 # suffix actions
      + 2*history*D                 # shallow-state keys
      + 2*Lmax*Hkv*head_dim*2       # first-layer BF16 K and V
      + 2D                           # final RMSNorm
      = 7,981,689,344 bytes
      = 7.433527470 GiB
```

The unallocated difference to 8 GiB is `608,245,248` bytes before allocator fragmentation, packed-kernel workspace, token buffers, addresses, and synchronization. Therefore this is a projection and a necessary residency screen, not a measured 8-GiB implementation pass.

Embedding rows are assumed fetched on demand and are not resident. That traffic must be charged in a physical follow-up.

## 10. Verification traffic equation

If one complete exact checkpoint sweep verifies a block of `K` accepted tokens, the most favorable logical weight fraction is:

```text
rho_W(K, C) = 1 / (K*C)
```

where `C>=1` is the lossless checkpoint compression ratio. At the frozen `K=128`:

```text
rho_W(128, 1)        = 0.781250000%
rho_W(128, 1.261972) = 0.619070788%
```

Both are below the registered p50 whole-model fraction `1.185185185%` **only if all 128 candidates are accepted**. Partial acceptance is charged with its actual length.

The verifier still performs the complete target arithmetic. This experiment does not claim an arithmetic reduction; block GEMM utilization and physical latency require a later Gate.

## 11. Integrity controls

The run is invalid unless all controls pass:

- row-wise Q4 quantization is deterministic and its packed-byte equation is exact;
- BF16 suffix-action storage round-trips deterministically;
- stable rank handles score ties by token ID;
- synthetic carry and nearest-history positive controls recover their designed chains;
- build selection is deterministic;
- candidate chains are hashed before target continuation begins;
- build and holdout prompts are disjoint;
- the selected holdout mode was frozen before any holdout target continuation;
- every page/history/resource count is nonnegative and internally consistent;
- the official checkpoint and pinned runtime load successfully.

## 12. Decisions

### Chain promotion

```text
PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_EXACT_BLOCK_EXECUTOR_GATE
```

requires:

- every integrity control passes;
- the selected mode accepts `128/128` tokens on every holdout case;
- projected hot state is at most 8 GiB;
- no-compression verification fraction is at most `1.185185185%`.

This promotes only an exact block-verifier implementation. It is not final-target success.

### Candidate-tree continuation

```text
PROMOTE_CAUSAL_SUFFIX_ACTION_CACHE_TO_CANDIDATE_TREE_GATE
```

requires the chain Gate to fail but, across untouched holdout true-path diagnostics:

```text
p95 target rank <= 4
maximum target rank <= 16
```

This is only a necessary signal. Tree node growth, branch-state storage, and exact verification remain untested.

### Rejection

Otherwise:

```text
REJECT_CAUSAL_SUFFIX_ACTION_CACHE_AS_128_TOKEN_DRAFT_CORE
```

## 13. Stop rule

After rejection, do not sweep:

- history length;
- shallow depth;
- polynomial order;
- Q4 range or scale format;
- prompt subset;
- block length;
- nearest-neighbor metric;
- mode order;
- thresholds.

Reopening requires a materially new causal suffix representation, a different exact decision certificate, or measured candidate-tree structure that changes the dominant information source. A learned draft model, checkpoint modification, hidden target call, or uncharged dense fallback is outside the fixed mission.

## 14. Claim boundary

A pass does not prove TARGET-W execution, target VRAM, target SSD/PCIe behavior, exact packed-Q4 draft kernels, target block arithmetic speed, or 4B-class p50/p95. Those remain `NOT TESTED` until a complete existing-ISA block executor is measured.
