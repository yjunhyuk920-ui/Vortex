# EXP-093A — One-Sweep True-Token Rank and Static Branch-Source Gate

## 1. Purpose

EXP-091A showed that the unchanged target Jacobi operator released zero exact holdout tokens on its first sweep. EXP-092A then granted the entire guessed-block activation span and still certified 64 additional independent target directions against an eight-direction budget.

The next cheapest admissible question is discrete rather than linear:

> Before any target continuation is generated, does one unchanged-target guessed-context sweep at least place every future exact token inside a small static candidate set?

A positive result would not solve branch execution. It would justify constructing an exact nonlinear branch-closure mechanism. A negative result eliminates this static candidate source before any tree, verifier, CUDA kernel, or 405B implementation is built.

## 2. Frozen public-checkpoint boundary

```text
DEV-W: HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
reference: transformers 4.46.3 LlamaForCausalLM, eager BF16 CPU
K: 128
seed: prompt_suffix_cycle_16, inherited unchanged from EXP-091A
build: English, Korean, code
untouched holdout: mathematics, structured JSON, repeated pattern
```

For each case:

1. prefill the exact committed prompt and compute the exact boundary token;
2. generate the frozen prompt-only 128-token seed;
3. execute one official target block sweep under the guessed context;
4. retain the complete FP32 vocabulary-logit row for each of the 128 positions;
5. complete and hash that sweep before starting target generation;
6. run the official incremental greedy target for 128 one-token steps;
7. evaluate the rank of each exact target token in the already-computed guessed-context logits.

The target tokens never influence the guessed sweep, seed, logits, or candidate ordering.

## 3. Exact stable rank

Let `z_i(v)` be the guessed-sweep FP32 logit for token `v` at position `i`, and let `y_i` be the later official incremental target token. The frozen reference uses `torch.argmax`, whose tie rule chooses the smallest token ID. The exact one-based rank is therefore

\[
R_i
=
1
+
\left|\{v:z_i(v)>z_i(y_i)\}\right|
+
\left|\{v<y_i:z_i(v)=z_i(y_i)\}\right|.
\]

Thus `R_i <= B` if and only if the true token is inside the stable top-`B` set of that already-computed row.

The first position must have rank one because both the guessed sweep and the ordinary incremental reference receive the same exact boundary token and prefix cache. Failure of that control invalidates the run.

## 4. Why this is an exact branch-source Gate

A static candidate source with width `B` supplies

\[
C_i(B)=\operatorname{TopB}(z_i)
\]

before target generation. Any branch mechanism restricted to these candidate sets can contain the true path only if

\[
y_i\in C_i(B)
\quad\forall i.
\]

Consequently, any observed `R_i > B` is a decisive miss for this source. It cannot be repaired by a faster tree evaluator, cache layout, verifier, or kernel because the required token is absent from the source set.

A pass is only necessary, never sufficient. Alternate branch tokens change later hidden states and logits. EXP-093A neither assumes nor credits those successor transitions.

## 5. Frozen thresholds

The candidate-tree signal from EXP-090A is retained unchanged:

```text
pooled true-token rank p95 <= 4
worst-case true-token rank <= 16
every individual build and holdout case satisfies both thresholds
```

Promotion requires all controls plus the threshold on both build and untouched holdout populations:

```text
PROMOTE_ONE_SWEEP_STATIC_TOPK_TO_EXACT_BRANCH_CLOSURE_GATE
```

Otherwise:

```text
REJECT_ONE_SWEEP_STATIC_TOPK_BRANCH_SOURCE_AS_405B_CORE
```

Integrity failure produces:

```text
INVALID_ONE_SWEEP_TRUE_TOKEN_RANK_CONTROL_FAILURE
```

## 6. Favorable resource accounting

One block sweep reads each lossless checkpoint page once and amortizes logical checkpoint traffic over 128 positions:

\[
\rho_{\rm raw}=1/128=0.78125\%,
\]

and under the favorable DEV-W lossless ratio `1.261972x`:

\[
\rho_{\rm compressed}=1/(128\cdot1.261972)
=0.6190707876244481\%.
\]

A static top-16 token-ID table for a vocabulary below `2^16` costs only

\[
128\cdot16\cdot2=4096\text{ bytes}.
\]

These are deliberately favorable source costs. The sweep still executes 100% of the dense arithmetic for 128 positions. Branch successor construction, branch verification, rollback, and final state commitment remain unmeasured and receive no success credit.

For diagnostics, the oracle-minimum static Cartesian candidate volume is reported as

\[
I_{\rm path}=\sum_{i=1}^{128}\log_2 R_i.
\]

It is not a deployable coding theorem because the ranks are observed using the later true path.

## 7. Controls

- exact runtime and checkpoint revision pins;
- six distinct prompt hashes and frozen build/holdout split;
- guessed sweep completion timestamp no later than target start;
- exactly 128 guessed rows and 128 incremental target calls per case;
- first-position proposal equality and true-token rank one;
- finite FP32 logits;
- exact descending-logit, ascending-token-ID tie control;
- positive and negative synthetic rank-Gate controls;
- vocabulary fits the registered two-byte token representation;
- checksummed checkpoint identity, raw rows, deterministic core, and result.

## 8. Stop rule

On rejection, do not sweep seed, K, prompt subset, rank thresholds, top-k ordering, dtype, temperature, tokenizer, or model size around this source. Do not replace static top-k with a larger static list and call it a new branch mechanism.

Reopening requires a materially new source that changes candidate information, for example:

- a causal nonlinear finite-word token certificate;
- exact branch-dependent successor logits generated without another dense checkpoint sweep;
- a globally shared symbolic branch transition;
- a proof that excludes competitors without enumerating the missing true token from this static row.

## 9. Claim boundary

A pass authorizes only an exact branch-successor construction Gate. It does not establish a causal tree, token commitment, arithmetic reduction, TARGET-W behavior, 8-GiB residency, CUDA/SASS behavior, or same-machine 4B-class p50/p95. A rejection applies only to the frozen one-sweep static top-k source.
