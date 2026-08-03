# EXP-051 — Oracle Layer-Finalization and Tail-Skip Gate

## Status

```text
Implementation branch: research/exp-051-layer-finalization-tail-skip
Gate registration: COMMITTED BEFORE REAL-CHECKPOINT RUN
Scientific result: PENDING
Phase: A/B with small-checkpoint current-token observation
Evidence ceiling: E1
Real skipped-layer operation replacement: false
Phase D: NOT TESTED
```

## Question

With the exact target greedy prefix fixed, how early in Transformer depth does the current next-token decision equal the full-depth token and remain unchanged by every later block?

This is an upper-bound audit for tail-layer skipping, not a deployable early-exit runtime.

## Intermediate probe definition

For a target with `L` GPT-Neo blocks and exact current model output:

```text
hidden_states[0]   embedding output before block 1
hidden_states[d]   state after d blocks for 1 <= d < L
hidden_states[L]   state after final block and target final ln_f
```

GPT-Neo applies `transformer.ln_f` before appending the final hidden-state entry. Therefore:

```text
probe d < L: lm_head(ln_f(hidden_states[d][:, -1]))
probe d = L: lm_head(hidden_states[L][:, -1])
```

Applying `ln_f` twice at final depth is forbidden.

The final-depth probe logits must reconstruct `outputs.logits[:, -1]` within tolerance and have the same argmax as the exact generated token. Any mismatch aborts the experiment.

## Oracle definitions

Let `z_d` be the intermediate argmax token and `z_L` the exact target token.

```text
first_match_depth = min d: z_d == z_L
suffix_stable_depth = min d: for all j >= d, z_j == z_L
```

A first match may be transient. Only suffix-stable depth is used for the favorable tail-skip upper bound.

Both depths are offline reference labels. Suffix stability consumes later layer outputs and cannot be a runtime selector.

## Conditions

### L0 — full exact cached baseline

For every target/prompt, generate exactly 64 target tokens with KV cache and `output_hidden_states=true`. Record one token-state row per generated token.

### L1 — first-match diagnostic

Record earliest depth matching the final token, token changes across depth, and wrong depths after first match.

### L2 — suffix-stable oracle

Record the earliest depth after which every later token probe equals the exact final target token. This is the primary favorable oracle.

### L3 — pre-registered fixed depths

For each target, map fractions to depths using `ceil(fraction * L)`:

```text
fractions = 0%, 12.5%, 25%, 50%, 75%, 100%
```

For the current 8-block checkpoints this yields depths `0,1,2,4,6,8`. Record exact token agreement by target and family.

### L4 — per-state reference depth selection

Use exact `suffix_stable_depth` independently per token. This selector is non-deployable and deliberately favorable. It pays only one output-head probe at the chosen depth.

### L5 — late-decision residual adversary

Use a two-dimensional residual chain and identity two-token readout:

```text
initial hidden [2, 0] -> token 0
early residuals [0, 0]
final residual [-4, +3]
final hidden [-2, 3] -> token 1
```

The exact decision flips only at the final block. Expected:

```text
first_match_depth = L
suffix_stable_depth = L
stable byte fraction = 100%
```

This is an executable counterexample to a universal fixed early-exit depth for arbitrary residual targets.

### L6 — sound nonlinear tail certificate

Forbidden until L2 survives the empirical oracle Gate. Multi-layer token agreement, margin growth, or exact-reference stability is not a certificate. A valid certificate must bound every omitted attention/MLP residual's effect on the final top-1 decision without executing the omitted blocks.

## Corpus

Pinned unchanged checkpoints/tokenizer:

```text
EleutherAI/gpt-neo-125M @ 21def0189f5705e2521767faed922f1f15e7d7db
roneneldan/TinyStories-1M @ 77f1b168e219585646439073245fe87e56b3023e
roneneldan/TinyStories-3M @ cfaf26ec85ecdfc1bd7c2638104cce55cb67f894
roneneldan/TinyStories-8M @ 8612e3b15c66ffa94eaa6ee0de5c96edd2d630af
```

Six held-out families are English narrative, Korean, code, mathematics, structured JSON, and identifier boundary.

Expected rows:

```text
3 targets * 6 prompts * 64 exact generated tokens = 1,152 token states
```

Context-limit exclusions are preserved and cannot disappear silently.

## Logical traffic

For each target:

```text
B_embed_row = one current token embedding row + one position embedding row
B_block[d] = logical bytes of block d parameters
B_ln_f = logical bytes of final norm parameters
B_head = logical bytes of full LM head weight and bias if present
B_full = B_embed_row + sum(B_block) + B_ln_f + B_head
```

At favorable oracle depth `d`:

```text
B_oracle(d) = B_embed_row + sum_{j<=d} B_block[j] + B_ln_f + B_head
fraction(d) = B_oracle(d) / B_full
```

This assumes:

- the exact depth is known for free;
- only one final-norm/head probe is executed;
- omitted blocks are never read;
- KV/selector/certificate cost is zero.

It is therefore a favorable lower bound on deployable traffic. Actual early-exit can only cost more.

Tied input/output weights are counted as a full logical LM-head read because producing all logits consumes the output matrix even if storage is shared.

## Per-token raw fields

```text
model/prompt/family/token index
exact generated token
intermediate token for every depth
intermediate top1 margins
final reconstruction max absolute error
first-match depth/fraction
suffix-stable depth/fraction
suffix-stable logical byte fraction
post-first-match wrong depths
token changes across depth
fixed-depth exactness map
future generated token use=false
```

## Aggregate fields

MEASURED:

- token-state count/exclusions;
- reconstruction mismatch and maximum error;
- p50/p90 first-match and stable depths;
- p50/p90 stable byte fraction;
- head fraction and per-block bytes;
- transient-match rate;
- fixed-depth accuracy by target/family;
- family/model stable-depth medians;
- full forward/probe CPU time and RSS;
- adversarial late-flip result.

DERIVED:

- empirical oracle Gate booleans;
- universal late-decision Gate;
- target-size trend;
- PROJECTED 405B bytes under observed fractions;
- gap to 1.185185%.

UNVERIFIED:

- deployable selector/certificate;
- real omitted-layer operation replacement;
- 70B/405B finalization behavior;
- 8 GiB execution;
- CUDA/PCIe/SSD/TTFT/tokens per second.

## Pre-registered Gate

### Empirical oracle rejection

Reject adaptive layer-tail skipping before selector work if any holds:

```text
final-depth reconstruction mismatch >0
future generated token use >0
median suffix-stable logical fraction >10%
p90 suffix-stable logical fraction >25%
median suffix-stable block fraction >10%
any family median stable block fraction >50%
largest-model median stable fraction >1.25x smallest-model median
```

Failure decision:

```text
REJECT_LAYER_FINALIZATION_TAIL_SKIP_CORE_RETAIN_ORACLE_AUXILIARY
```

### Universal claim rejection

If the late-decision residual target succeeds, reject fixed-depth exact early exit for the arbitrary-target mission:

```text
REJECT_LAYER_FINALIZATION_TAIL_SKIP_AS_UNIVERSAL_CORE
```

If the empirical oracle passes but the universal adversary succeeds, only a restricted adaptive tail-certificate Gate may continue, with an explicit non-universal claim.

## Promotion Gate

L6 and real operation replacement may proceed only when the empirical oracle survives. Final promotion still requires:

```text
zero final reconstruction mismatch
zero future information
sound causal tail certificate
actual skipped blocks during complete generation
p90 fully accounted fraction <=0.011851851851851851
useful savings in every family
non-degrading size trend
claim scope consistent with late-decision counterexample
```

## Projection boundary

```text
405B Q4 full stream 188.592821 GiB
1.2x 4B Q4 allowance 2.235174 GiB/token
required fraction 1.185185%
```

These are parameter-count projections, not target measurements.

## Commands

```bash
python -m pytest -q tests/exp_051
bash experiments/exp_051/run_current_env.sh
bash experiments/exp_051/reproduce.sh
```
