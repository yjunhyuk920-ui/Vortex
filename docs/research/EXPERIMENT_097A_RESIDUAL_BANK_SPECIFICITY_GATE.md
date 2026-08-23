# EXP-097A — Residual Bank Specificity Gate

## Purpose

EXP-096A proved that a target-seeing minimum-L1 LP can make every delayed target
token the unique winner using a K=128 online residual bank. The E0 random-geometry
control on this branch then showed that six of six unrelated Gaussian K=128 banks
at the same 49,152-token width can also certify arbitrary targets with sparse
solutions.

EXP-097A therefore asks the cheapest remaining question before any causal
coefficient generator is built:

> Does the real public-checkpoint residual bank contain target-coordinate
> information that an equally shaped but semantically broken control bank does
> not contain?

This is a specificity falsification Gate, not a runtime.

## Frozen public checkpoint

- `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- official `transformers==4.46.3`, BF16 eager CPU path
- inherited K=128 `prompt_suffix_cycle_16` guessed sweep
- inherited official first decoder layer + rowwise-symmetric Q4 LM head
- inherited build English/Korean/code and untouched holdout math/JSON/repetition
- sampled positions `0,16,32,48,64,80,96,127`

Every one of the six seed residual banks is generated and hashed before the first
delayed true target continuation starts.

## Three bank variants

For target case `i` and its true shallow logits `c_t` / target token `y_t`:

1. `same`: the original bank generated from case `i`;
2. `cross_prompt`: the complete bank from the next frozen prompt;
3. `vocab_permuted`: the original bank with one deterministic permutation of
   its vocabulary columns.

The third control preserves K, every residual value, row/column norms, marginal
distribution, and high-dimensional geometry while breaking the association
between a residual column and its token ID.

All variants use the unchanged EXP-096A minimum-L1, unbounded FP64 margin LP and
full-vocabulary roundoff scan. Only the initial active-set source is reduced to
the common true shallow score so that a variant does not receive a favorable
variant-specific warm start.

## Decision

A control is considered to match the same-bank capacity if, on untouched
holdouts:

```text
control certificate fraction >= same certificate fraction
and
control support p95 <= 2 * same support p95
```

The factor two is deliberately permissive. This Gate does not try to prove that
coefficient support maps to runtime cost; it asks only whether the real bank has
an obvious specificity advantage over semantic-destruction controls.

Decisions:

```text
REJECT_UNBOUNDED_RESIDUAL_MARGIN_CAPACITY_AS_MODEL_SPECIFIC_SOURCE
SURVIVES_RESIDUAL_BANK_SPECIFICITY_REQUIRES_FINITE_CODEBOOK_GATE
REJECT_RESIDUAL_BANK_SPECIFICITY_SOURCE
INVALID_RESIDUAL_BANK_SPECIFICITY_CONTROL_FAILURE
```

If either semantic-destruction control matches, the target-seeing continuous
residual LP is removed as evidence for reducing dense arithmetic `r`. The
mathematical EXP-096A capacity result remains true but is treated as generic
high-dimensional controllability.

If the real bank survives, the next Gate must replace per-target continuous LP
solves with a finite build-only coefficient/codeword set before any causal
router is attempted. No parameter sweep of the present unbounded LP is allowed.

## 10x boundary

This specificity Gate itself receives zero `r` credit. A later residual-derived
core must demonstrate both:

```text
fine dense arithmetic fraction r <= 0.1
and
exact/bisimilar committed successor state
```

with all refreshes, misses, repairs, coefficient generation, candidate nodes,
and cold bytes charged. A causal predictor that still requires the full fine
sweep remains auxiliary only.

## Stop rule

On rejection, do not sweep K, margin, solver, prompt subset, sample positions,
permutation seed, or support-ratio threshold. Reopening the residual route
requires a materially different checkpoint-specific invariant that remains
predictive under semantic-destruction controls and supplies an exact verifier,
not another unconstrained target-seeing linear combination.
