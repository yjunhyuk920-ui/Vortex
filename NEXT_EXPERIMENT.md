# Next Experiment

## EXP-047 — Causal Probabilistic Tile Certificate

### Classification

- core research: yes
- Phase A: required
- Phase B: required in current branch
- Phase C: follows only if Phase B promotion thresholds pass
- Phase D: NOT TESTED
- initial evidence: E0

## Direct connection to the final objective

EXP-047 attempts to avoid applying every input-dimension tile of selected dense Transformer linear operators on an unseen causal state.

It does not replay stored responses, memorize prompt paths, or assume future tokens.

## Hypothesis

For real Transformer linear operators, signed tile contributions to a decision-relevant scalar or low-dimensional projection exhibit enough cancellation that a sequential sample-without-replacement estimator plus a valid finite-population confidence sequence can certify the final decision after reading substantially fewer tiles than deterministic worst-case residual norms require.

## Original operation skipped

For `y = W x`, partition `W` by input dimension:

```text
W = [W_1 ... W_T]
x = [x_1 ... x_T]
y = sum_i W_i x_i
```

The optimized executor evaluates only a prefix of a randomized tile permutation when its certificate proves that every omitted contribution cannot change the declared downstream decision. Otherwise it reads all remaining tiles and returns exact `W x`.

## Causal decision

The tile order, observed contributions, confidence state, current activation, and declared decision margin are available at the current token. Future generated tokens are forbidden.

## Correctness contract

Two modes are separated:

1. **strict exact fallback mode:** a token is committed only after a deterministic exact certificate, or after all tiles have been evaluated;
2. **probabilistic certified mode:** a token may be committed when a time-uniform finite-population bound limits wrong certification probability to `delta_total` after union accounting across operators and tokens.

Every result must report which mode produced it. A probabilistic result must never be described as deterministic exactness.

## Failure detection and fallback

- certificate invalid, non-finite, or numerically unstable: evaluate all remaining tiles;
- confidence radius does not close before the rejection threshold: evaluate all remaining tiles;
- predicted decision differs from exact reference during Phase B/C validation: record a certification failure and reject the implementation;
- any unaccounted future information: invalidate the experiment.

The fallback output must be bitwise or tolerance-equivalent to the declared reference implementation.

## Phase A obligations

- derive a finite-population confidence sequence for signed tile contributions;
- state assumptions explicitly and register them;
- derive selector cost, bytes read, arithmetic, state, and union-bound accounting;
- derive a 405B resource model with MEASURED/DERIVED/PROJECTED/UNVERIFIED labels;
- identify adversarial tile distributions that force full fallback;
- compare against previous deterministic signed-residual failures.

## Phase B implementation

Implement two independent paths:

- slow reference: exact full tile summation;
- optimized candidate: randomized sequential tile evaluation plus certificate and exact fallback.

Required tests:

- randomized property tests;
- all-positive, alternating-sign, heavy-tail, one-dominant-tile, zero-margin, and cancellation cases;
- deterministic replay under fixed seed;
- malformed configuration and numerical fault injection;
- selector overhead and tile-read scaling for increasing `T`;
- zero silent wrong accepts in the test corpus;
- empirical wrong-accept count reported separately from mathematical `delta`.

## Phase B promotion criteria

All must pass:

- reference/fallback agreement: 100%;
- silent wrong accepts: 0;
- certificate implementation matches an independently computed bound;
- no future information usage;
- at least one nontrivial synthetic family certifies after reading <=25% of tiles;
- adversarial families correctly fall back to 100% tiles;
- selector arithmetic and metadata are charged;
- 405B projection fields are complete, even if unfavorable.

Failure of the <=25% positive control does not prove the idea impossible, but blocks Phase C until the estimator or certificate is revised.

## Phase C pre-registration

Only after Phase B promotion:

- unmodified checkpoints, preferably at least three available sizes;
- held-out prompts and task families;
- replace a real linear operation, not an offline hook-only analysis;
- report logits, tokens, exact forward calls, evaluated tile fraction, fallback, CPU time, RAM, and model-size trend;
- first falsification target: LM head decision certificate, then selected MLP/down projections only if the complete causal accounting remains sound.

TinyLlama results will be described only as early falsification evidence.

## Strongest falsification experiment

Construct and search real/synthetic states where many individually small signed tiles align late in the permutation and flip the winner after an apparently stable partial estimate. If the time-uniform certificate ever accepts such a state incorrectly, reject the implementation. If real-model runs require nearly all tiles at useful confidence, reject CPTC as a primary execution mechanism.

## Required output structure

```text
docs/research/EXPERIMENT_047_CAUSAL_PROBABILISTIC_TILE_CERTIFICATE.md
experiments/exp_047/
results/exp_047/
tests/exp_047/
.github/workflows/exp_047_gate.yml
```
