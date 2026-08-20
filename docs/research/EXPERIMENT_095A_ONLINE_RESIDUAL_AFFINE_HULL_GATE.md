# EXP-095A — Online Deep-Residual Affine-Hull Gate

## 1. Question

Can one unchanged-target guessed-context sweep expose a **current-block deep-residual bank** whose branch-conditioned affine hull is rich enough to recover or tightly rank the exact tokens on a corrected causal prefix, without another fine checkpoint sweep?

EXP-094A transported only the position-aligned residual

```text
R_i = F_i(g) - G_i(g)
P_i = G_i(c) + R_i
```

and made the true-token rank worse. EXP-095A changes the correction dependency rather than tuning that formula. The entire online bank

```text
R = [R_0, R_1, ..., R_127]
```

becomes one nonseparable superinstruction. A corrected shallow state may combine residuals from other positions. This is not a static candidate list, a second fine sweep, a prompt-history cache, or a small extension of the guessed activation span.

## 2. Frozen public-checkpoint boundary

```text
DEV-W       HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
reference   transformers==4.46.3 LlamaForCausalLM
dtype       BF16, eager CPU
coarse      official first decoder layer + official final RMSNorm
head        rowwise-symmetric INT4 simulation, BF16 scales
seed        prompt_suffix_cycle_16 inherited from EXP-091A
block       K=128
population  build English/Korean/code; untouched holdout math/JSON/repetition
```

Every candidate chain is completed and hashed before the official incremental target continuation begins.

## 3. Online source

For guessed input block `g`, one official full target sweep produces fine logits

```text
F_i(g),  i=0..127.
```

The official one-layer coarse executor produces normalized shallow states and logits

```text
h_i(g), G_i(g).
```

The online deep-residual bank is

```text
R_i(g) = float64(F_i(g)) - float64(G_i(g)).
```

The bank is prompt/block specific and is produced online. It contains no target continuation token, corrected-prefix state, target-state hash, or stored response table.

## 4. Causal affine transport

At corrected position `i`, the coarse executor has seen only the committed prefix and the earlier candidate tokens. It produces `h_i(c), G_i(c)`.

Choose the eight nearest rows of `h(g)` by squared L2 distance after float64 row normalization. Let the selected matrix be `H_N` and residual rows be `R_N`. The frozen affine ridge coefficients solve

```text
min_w ||w H_N - h_i(c)||_2^2 + 2^-10 ||w||_2^2
subject to sum(w) = 1.
```

The equality is eliminated by a deterministic augmented normal equation. Coefficients are clipped to `[-4,4]`, then renormalized to sum to one. The causal score is

```text
P_i^causal = G_i(c) + w R_N.
```

The selected token is the stable lowest-ID argmax. The token becomes the next coarse input. No target token or target logit participates in the chain.

## 5. Two favorable capacity oracles

These diagnostics execute only after the candidate is complete and the delayed official target path begins. They never count as a deployable candidate.

### O1 — best single online residual

For true coarse state `G_i(t)` and true token `y_i`, O1 evaluates

```text
G_i(t) + R_j(g),  j=0..127
```

and records the minimum exact stable rank of `y_i`. This is a finite exhaustive target-seeing oracle. If O1 fails, no router selecting one bank row can rescue the source.

### O2 — complete row-span L2 projection

Let `B` be the `128 x V` residual bank. The delayed target provides the diagnostic true residual

```text
T_i = F_i(t) - G_i(t).
```

Using only the row Gram matrix

```text
K = B B^T,
```

O2 computes the deterministic float64 pseudoinverse with relative eigenvalue cutoff `2^-40` and projects

```text
T_hat = T B^T K^+ B.
```

The oracle score is

```text
P_i^span = G_i(t) + T_hat_i.
```

This grants unavailable target residuals and optimal L2 coefficients. It is a favorable capacity screen for this exact frozen projection algorithm, not a causal executor and not a proof about every coefficient objective.

## 6. Why this is materially different

- EXP-092A tested whether exact target **operator inputs** lie in a small linear extension of guessed inputs. EXP-095A uses a full current-block bank of nonlinear deep **logit residuals** and asks only for exact token decisions.
- EXP-093A kept each guessed-sweep logit row static. EXP-095A changes scores with the corrected shallow state and can borrow residuals across positions.
- EXP-094A used exactly one position-aligned residual. EXP-095A allows a branch-conditioned affine combination and measures the complete online residual row span.
- EXP-086A functional microprograms were checkpoint-static low-rank affine approximations of one MLP. EXP-095A is an online block superinstruction at the accepted token-decision boundary established by prefix-state bisimulation.

## 7. Exactness and controls

The experiment must fail closed unless all of the following hold:

1. pinned runtime and checkpoint identities match;
2. all six prompt hashes are distinct;
3. fine guessed sweep completes before target generation;
4. causal chain completes and is hashed before target generation;
5. the first corrected state equals the first guessed state, so the first causal residual reproduction is byte-identical to EXP-094A transport;
6. all fine, coarse, causal, and target calls equal the frozen block length;
7. every rank uses the exact stable lowest-token-ID tie rule;
8. the Gram matrix is symmetric and finite;
9. retained eigenvalues and effective row-span rank are recorded;
10. synthetic affine-residual positive and orthogonal-negative controls pass;
11. a one-score-bit perturbation changes the corresponding hash;
12. no NaN, infinity, empty neighbor set, singular unresolved solve, or target-order violation occurs.

Causal candidate tokens are compared to the delayed target. Prefix-state bisimulation means a fully exact token block could later reconstruct the official successor state by replay; EXP-095A does not credit that replay for free.

## 8. Resource equations

The online residual bank already required by the source contains

```text
K * V * 8 bytes
```

for float64 residuals. The shallow-state bank adds

```text
K * H * 8 bytes.
```

The Gram matrix adds

```text
K^2 * 8 bytes.
```

For TARGET-W (`K=128`, `V=128256`, `H=16384`) these are charged in addition to the inherited one-layer/Q4-head/first-layer-KV hot ledger.

Causal affine scoring performs at most eight residual-vector combinations per token. The favorable parameter-equivalent scalar work is

```text
8 * V + O(8H + 8^3)
```

per candidate token. Full-span O2 is an offline favorable diagnostic and is not credited as runtime work.

One complete fine sweep still performs `100%` of dense target arithmetic. At perfect release of 128 tokens its logical checkpoint traffic is

```text
1/128 = 0.78125%
```

or

```text
1/(128 * 1.261972) = 0.6190707876%
```

under the inherited favorable lossless ratio. A source pass therefore authorizes only another exact certificate/physical block Gate; it is not final latency success.

## 9. Frozen decisions

### Exact causal chain

Every build and holdout case must accept `128/128` tokens:

```text
PROMOTE_CAUSAL_RESIDUAL_AFFINE_HULL_TO_EXACT_CERTIFICATE_GATE
```

### Causal candidate-tree source

If the exact chain fails, both build and holdout causal true-token ranks must satisfy

```text
p95 <= 4
maximum <= 16
```

for:

```text
PROMOTE_CAUSAL_RESIDUAL_AFFINE_HULL_TO_BRANCH_GATE
```

### Oracle span capacity

If the causal source fails, O2 must satisfy the same rank thresholds on build and holdout for:

```text
PROMOTE_ONLINE_RESIDUAL_SPAN_TO_CAUSAL_COEFFICIENT_GATE
```

Otherwise:

```text
REJECT_ONLINE_DEEP_RESIDUAL_AFFINE_HULL_AS_405B_SOURCE
```

Any integrity failure yields:

```text
INVALID_ONLINE_RESIDUAL_AFFINE_HULL_CONTROL_FAILURE
```

## 10. Stop rule

On rejection, do not sweep neighbor count, ridge coefficient, clipping bound, hidden metric, seed, block length, coarse depth, residual dtype, eigenvalue cutoff, prompts, or rank thresholds. Do not replace the L2 projection with another post-target coefficient optimizer and call it the same source.

Reopening requires a new information dependency: a finite-word token-margin certificate, a branch-specific residual generated without target information, a higher-order correction whose coefficients are checkpoint-derived rather than fitted from the delayed target, or an exact symbolic transition that also reduces the fine dense arithmetic.

## 11. Claim boundary

A pass would establish only a small-real-checkpoint token-source signal. TARGET-W execution, physical 8-GiB allocation, exact token certificate, successor-state construction cost, CUDA/SASS, SSD/PCIe/HBM traffic, fine arithmetic reduction, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
