# EXP-077A -- Activation-Informed Fractal MLP Oracle Gate

## Status

Complete. The authoritative E1 result rejects the registered 10% path. Source
commit `33fed17ed6abed7c8c14eec543efc620e4fe537d`; evidence commit
`0970c6626ff848c5026b684b3e2d1bb479e96603`.

## Hypothesis

An unchanged Transformer MLP can be decomposed exactly along its SwiGLU
intermediate channels:

```text
MLP(x) = sum_i W_down[:, i] * (SiLU(W_gate[i, :] x) * W_up[i, :] x)
```

For a particular causal activation, a small input-dependent subset may preserve
the target logit distribution closely enough that Qwen3.5-122B-A10B's existing
10B activated path could later be reduced toward 1B. All checkpoint weights
remain unchanged and recoverable; only the executor's selected operations
change.

This reopens neither exact-zero activation skipping nor absolute-unread exact
certification. EXP-061 found no exact zeros. EXP-077A explicitly tests a declared
quality contract using nonzero, activation-conditioned contribution scores.

## E0 scorecard

### Target-scale upside

The official Qwen3.5-122B-A10B structure reports 122B total and 10B activated
parameters per token. The final surrogate target therefore requires an inner
activated-path fraction no larger than:

```text
1B / 10B = 10%
```

If all later operator families admitted the same fraction and fully charged
selector/correction costs fit inside the remainder, the composition has an
order-of-magnitude active-work ceiling. Storage remains cold-backed and is not
claimed solved.

### New information source

The selector signal is the current causal activation's post-SiLU SwiGLU
channel magnitude combined with the original down-projection column norm. It
is not weight equality, an exact zero, temporal replay, a draft token, future
target text, or a static low-rank representation.

### Scaling premise

The decomposition applies automatically to any compatible SwiGLU MLP. Wider
intermediate dimensions provide more selection granularity, but concentration
improvement with scale is unverified. The Qwen MoE outer router is not tested
on the dense 0.8B checkpoint.

### Fully charged boundary

This first Gate deliberately grants the non-deployable oracle all selection
cost, full gate/up activation computation, down-column norms, mask construction,
and full-output error measurement for free. Therefore success is only a
necessary favorable ceiling. A later Gate must charge all of them or replace
the oracle with a causal selector that does not read skipped rows.

Attention, Gated DeltaNet, embedding, LM head, KV, fallback, RAM, SSD, PCIe,
VRAM, and physical kernel costs remain outside EXP-077A. This omission can only
make the Gate easier and cannot promote a runtime.

### Cheapest decisive falsification

Use the already downloaded unchanged Qwen3.5-0.8B BF16 checkpoint. Replace each
MLP output in the frozen EXP-076 causal trace replay with the output from the
top fraction of intermediate channels. Compare against the unmodified target
logits on the same proposal-conditioned path. Do not train a selector or
download a new checkpoint before this Gate passes.

## Registered inputs

```text
checkpoint          Qwen/Qwen3.5-0.8B
revision            2fc06364715b967f1860aea9cf38778875588b17
weight SHA-256      04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
prompts             experiments/exp_076/prompts.json
prompt SHA-256      46c9779c24e59f356247145f7ef870a1b82ba4907eb9dffcd48652e519f7612a
target traces       results/exp_076/raw/case_rows.jsonl
trace SHA-256       1e921698ce8ee522c0d3cb9b8b9004139beb54cc2dee82e1a2aa0a08fa245e4f
trace evidence      55b79937c1f21887ae76b7e56ad61ba7dde8322a
tokens per prompt   8
fractions           5%, 10%, 20%
```

The six families are English, Korean, code, structured JSON, math, and
adversarial low-acceptance. The build/evaluation split remains exactly the
EXP-076 split; only the held-out evaluation split decides promotion.

## Oracle semantics

For every token position and MLP:

```text
z_i     = SiLU(W_gate[i,:] x) * W_up[i,:] x
score_i = abs(z_i) * L2_norm(W_down[:,i])
k       = max(1, floor(intermediate_size * registered_fraction))
mask    = top-k(score)
y_hat   = W_down (z * mask)
```

The oracle reads the complete `z`, so it is explicitly non-deployable. EXP-076
first prefills the exact prompt, then verifies the frozen sequence
`[first_target_token, *mtp_proposal_tokens]` on a cloned causal cache. EXP-077A
replays that exact two-stage conditioning for both target and candidate and
scores `[first_target_token, *target_verification_tokens]`. Verification outputs
are labels, never recycled as inputs. No later token is visible to an earlier
position. This isolates target-distribution preservation on a registered causal
trace and does not claim free-running autoregressive quality.

Rounding is downward because the registered fraction is a hard ceiling. On the
fixed 3,584-channel checkpoint, the 10% arm keeps 358 channels (`9.9888%`).

The CPU reference groups only equal-length prefixes, uses no padding, and
executes the same prefix-cache then verify-cache path as EXP-076. The
zero-mismatch control covers all 192 registered target decisions. This batching
changes host efficiency only; it does not change the causal state path, oracle
formula, split, fraction, or Gate.

### Pre-authoritative control correction

An initial implementation was correctly classified invalid: it recycled target
verification outputs as the next inputs and used a padded full-sequence
`use_cache=False` path. That produced 42 baseline trace mismatches. Replaying the
right proposal inputs reduced this to one; matching the original two-stage cache
path reduced it to zero for the isolated case and then zero across all 192
registered decisions. No invalid quality metric is scientific evidence. The
authoritative run starts only from the corrected source commit.

## Quality Gate at 10%

Promotion requires all of:

- zero mismatch between frozen target trajectory tokens and the unmodified
  baseline logits;
- realized selected MLP parameter fraction at most `10%`, counting the selected
  gate row, up row, and down column for every selected channel;
- held-out token top-1 agreement at least `99%`;
- every required family top-1 agreement at least `95%`;
- held-out mean `KL(target || candidate)` at most `0.02` nats;
- held-out p95 KL at most `0.05` nats;
- all metrics finite and all manifests/checksums valid.

Pass:

```text
PROMOTE_TO_FRACTAL_ALL_OPERATOR_AND_SELECTOR_COST_GATE
```

Scientific failure:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

Control failure:

```text
INVALID_FRACTAL_ORACLE_CONTROL_FAILURE
```

Dependency, checkpoint, unsupported-operation, storage, or timeout failure is:

```text
INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
```

## Stop and promotion rules

On failure, do not rescue this registered score with post-result fractions,
selected layers, selected prompts, a trained router, a physical sparse kernel,
or a 35B/122B download. Record the failure of activation-norm fracturing and
return to a materially different selector or execution dependency.

Passing authorizes only a new preregistered all-operator and deployable-selector
cost Gate. It does not establish full-model 10x work reduction because dense
attention, DeltaNet, LM head, selector, correction, and fallback remain. It does
not authorize target-server mutation, 35B/122B download, Phase D, E2-E7, or a
dense-405B claim.

## Evidence ceiling

Phase C small-real-checkpoint favorable-oracle observation, classified at E1.
The unchanged target supplies real weights and logits, and MLP outputs are
replaced, but the free non-deployable oracle prevents E2 operation-replacement
credit. Structurally valid conditions may be established; large-model and
physical performance remain unverified.

## Authoritative result

All 24 cases and 192 registered causal positions replayed the unchanged target
with zero mismatch. The 10% arm kept 358 of 3,584 channels per MLP, a realized
MLP parameter fraction of `9.9888392857%`.

| Fraction | Top-1 agreement | Mean KL | p95 KL |
|---:|---:|---:|---:|
| 5% | 55.5556% | 1.875917 | 5.475127 |
| 10% | 71.5278% | 0.884161 | 3.080752 |
| 20% | 83.3333% | 0.373114 | 1.264202 |

At 10%, the weakest family was English at `54.1667%` top-1 agreement; the best
was math at `83.3333%`. Every family missed the registered `95%` minimum. The
population missed `99%` top-1 by 27.4722 percentage points, mean KL exceeded
the `0.02` ceiling by `44.2081x`, and p95 KL exceeded `0.05` by `61.6150x`.

Decision:

```text
REJECT_ACTIVATION_NORM_FRACTAL_MLP_10PCT_PATH
```

The result closes only individual-channel activation-norm fracturing under this
favorable oracle. It does not prove all input-conditioned sparse execution
impossible. Do not rescue this score with post-selected fractions, layers,
prompts, a trained selector, or a larger checkpoint. The deterministic core is
`e25083693c6db21a0da16c9305816958b149865f2fcfde0bd9b7e95c43022411`.
