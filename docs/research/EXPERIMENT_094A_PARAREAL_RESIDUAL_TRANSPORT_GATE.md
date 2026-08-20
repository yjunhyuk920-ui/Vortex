# EXP-094A — Parareal Deep-Residual Transport Gate

## 1. Purpose

The post-EXP-093A static candidate source failed decisively: later true tokens had holdout rank p50 `1,152`, p95 `16,703.9`, maximum `42,864`, and only `12.5%` top-16 coverage. Enlarging that static list cannot create branch-dependent information.

EXP-094A changes the source rather than the thresholds. One unchanged-target guessed-context sweep supplies a position-aligned deep residual. A cheap causal coarse executor then transports that residual over its **corrected** prefix.

This is the Transformer analogue of one Parareal correction:

\[
\boxed{
\widehat z_i(c)
=
F_i(g)+G_i(c)-G_i(g)
}
\]

where:

- `g` is the frozen prompt-only guessed block;
- `F_i(g)` is the complete official target logit row from one K=128 sweep;
- `G_i(g)` is the coarse logit row on the same guessed causal prefix;
- `c` is the candidate chain produced so far;
- `G_i(c)` is the coarse logit row on that corrected causal prefix.

Unlike EXP-092A, this is not a small linear span of target activations. Unlike EXP-093A, later scores change causally when earlier candidate tokens change. Unlike EXP-090A, the deep residual comes from the current online guessed block rather than a frozen prompt-history entry.

## 2. Frozen public-checkpoint boundary

```text
DEV-W: HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
fine F: complete official transformers 4.46.3 LlamaForCausalLM, eager BF16
coarse G: official first decoder layer + official final RMSNorm + rowwise symmetric Q4 LM head
K: 128
seed: prompt_suffix_cycle_16
residual arithmetic: float64 over represented BF16/FP32 values
build: English, Korean, code
untouched holdout: mathematics, structured JSON, repeated pattern
```

## 3. Causal execution order

For every case:

1. prefill the exact prompt and freeze its complete cache plus boundary token;
2. build the prompt-only seed and guessed input `(boundary, g_1, ..., g_127)`;
3. execute one complete official target block sweep to obtain `F(g)`;
4. run the coarse executor sequentially on the same guessed input to obtain `G(g)`;
5. form the exact float64 residual block `R_i = F_i(g)-G_i(g)`;
6. from a fresh first-layer cache, generate the candidate sequentially using `argmax(G_i(c)+R_i)`;
7. complete and hash the entire candidate before any target continuation starts;
8. run the official incremental target for 128 steps as a delayed control;
9. on the later true path, evaluate `G_i(y)+R_i` to measure favorable oracle true-token ranks.

No target token, target state, or target rank influences `g`, `F(g)`, `G(g)`, `R`, or the candidate chain.

## 4. Exact finite-word transport

`F(g)` and `G(g)` are represented FP32 rows. The Gate promotes those values and `G(c)` to float64 before subtraction/addition. Every BF16/FP32 value is exactly representable in float64. Therefore, when the old and new coarse states coincide,

\[
G_i(c)=G_i(g)
\quad\Longrightarrow\quad
\widehat z_i(c)=F_i(g)
\]

exactly in the frozen working arithmetic. The first position is an integrity control: old/new coarse logits must be byte-identical, the transported score must recover the fine first decision, and both candidate and delayed target tokens must agree.

## 5. Promotion Gates

### Exact chain signal

```text
every build and holdout candidate accepts 128/128 target tokens
```

Decision:

```text
PROMOTE_PARAREAL_TRANSPORT_TO_EXACT_TOKEN_CERTIFICATE_GATE
```

This still requires a sound online certificate before commitment.

### Branch-source signal

If the chain is not exact, use the transported scores on the delayed true path only as a favorable oracle diagnostic. Promotion requires both build and holdout:

```text
true-token rank p95 <= 4
true-token rank maximum <= 16
```

Decision:

```text
PROMOTE_PARAREAL_TRANSPORT_TO_BRANCH_SUCCESSOR_GATE
```

Otherwise:

```text
REJECT_PARAREAL_RESIDUAL_TRANSPORT_AS_405B_BLOCK_SOURCE
```

Integrity failure produces:

```text
INVALID_PARAREAL_RESIDUAL_TRANSPORT_CONTROL_FAILURE
```

## 6. Resource equations

### Fine sweep

One lossless target sweep over K=128 has favorable logical checkpoint traffic

\[
1/(128\cdot1.261972)=0.6190707876\%\text{ per represented token}.
\]

It still performs `100%` of the dense target arithmetic per represented token.

### Coarse executor

For Llama-3.1-405B dimensions, the first dense layer plus LM head has parameter-equivalent arithmetic fraction

\[
\frac{P_{\rm layer}+P_{\rm head}}{405\times10^9}
\approx1.30593\%.
\]

This is additional to the fine sweep.

### Residual buffer

The favorable float64 residual block occupies

\[
B_R=K\cdot |V|\cdot8.
\]

For TARGET-W dimensions `K=128`, `|V|=128,256`:

\[
B_R=131,334,144\text{ bytes}.
\]

The registered first-layer/Q4-head/full-context-KV hot projection with one boundary-history row is `7.41655 GiB`; including the residual buffer is approximately `7.53887 GiB`, before allocator fragmentation, packed-kernel workspace, token buffers, and synchronization.

## 7. Controls

- pinned runtime/checkpoint/tokenizer identity;
- exact model-free recovery `G(old)+F(old)-G(old)=F(old)`;
- float64 tie and true-token rank controls;
- Q4 head shape and artifact equations;
- six distinct build/holdout prompts;
- fine sweep, old coarse path, and candidate all complete before target generation;
- exactly 128 fine rows, candidate coarse calls, old coarse calls, target calls, and true-path coarse calls;
- first-position old/new coarse byte equality;
- first fine, candidate, and target token equality;
- first static and transported true-token ranks equal one;
- complete cache/logit/candidate/residual hashes and immutable checksums.

## 8. Stop rule

On rejection, do not sweep coarse depth, seed, K, Q4 range, residual dtype, prompt subset, rank thresholds, or add a scalar damping coefficient. Do not replace the first layer with another nearby shallow prefix and call it new.

Reopening requires a new correction dependency, such as:

- a higher-order or branch-conditional residual whose construction does not require another fine sweep;
- a sound token certificate that consumes the transported scores and excludes all competitors;
- an exact symbolic transition composition that reduces fine dense arithmetic;
- a checkpoint-static nonlinear coarse model not obtained by training or modifying the target.

## 9. Claim boundary

A pass is only a candidate-source signal. EXP-094A does not construct a sound certificate, authorize token commitment, reduce fine dense arithmetic, run TARGET-W, measure physical 8-GiB allocation, or establish same-machine 4B-class p50/p95.
