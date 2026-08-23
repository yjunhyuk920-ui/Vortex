# E0 Residual-Margin Random-Geometry Audit

## Status

```text
Phase: E0 / model-free falsification control
Parent signal: EXP-096A residual direct-margin certificate
Scientific decision:
  DEMOTE_EXP096_SPARSE_SUPPORT_AS_MODEL_SPECIFIC_R_SIGNAL
  REQUIRE_REAL_CHECKPOINT_BANK_SPECIFICITY_BEFORE_RESIDUAL_CORE
```

## Why this Gate exists

EXP-096A certified every build and untouched-holdout target token with a
minimum-L1 coefficient vector over a K=128 residual bank. The observed
coefficient support was small (holdout p50 5, p95 13), which initially looked
like evidence that exact token decisions might depend on only a few deep
residual directions.

That interpretation contains a hidden assumption:

> sparse LP support implies checkpoint-specific decision information.

A 128-dimensional continuous control space over a 49,152-token vocabulary can
instead make many arbitrary vocabulary columns exposed by pure high-dimensional
geometry. If an unrelated random bank produces the same qualitative certificate
shape, EXP-096A remains a valid *capacity* result but its sparse support is not
evidence that the 405B dense arithmetic fraction `r` can be reduced.

This audit attacks that assumption before spending another official-checkpoint
run on a causal coefficient generator.

## Frozen synthetic control

For each deterministic seed:

```text
V = 49,152
K = 128
B[j,v] ~ iid Normal(0,1)
c[v]   ~ iid Normal(0,1)
y      = uniform vocabulary ID
margin = 2^-20
```

The LP is deliberately the same favorable control shape as EXP-096A:

```text
min ||a||_1

subject to

c[y] + a^T B[:,y]
    >=
c[v] + a^T B[:,v] + 2^-20

for every v != y.
```

The returned finite FP64 words are rescored against the complete vocabulary.
The audit also samples K=8,16,32,64 to expose the dimensional phase behavior.

This is not intended to model real residual statistics. It is a falsification
control for the inference that *small support by itself* proves model-specific
structure. A synthetic unrelated bank succeeding is sufficient to invalidate
that inference.

## Result

At the exact SmolLM2 vocabulary width used by EXP-096A:

```text
K=128 unrelated Gaussian trials       6 / 6 certified
support p50 / p95                      14 / 20.5
L1 p50 / p95                           1.725086 / 2.540288
K=8                                    infeasible
K=16                                   certified, support 9
K=32                                   certified, support 17
K=64                                   certified, support 19
```

For comparison, EXP-096A reported holdout support p50/p95 of 5/13. The random
control is not numerically identical, nor should it be; the decisive point is
that an unrelated bank at the same K and vocabulary width can certify arbitrary
targets with comparably sparse basic solutions.

Therefore:

```text
EXP-096A 768/768 capacity result        PRESERVED
EXP-096A sparse-support interpretation  DEMOTED
r-reduction claim from support alone    FORBIDDEN
causal generator promotion              BLOCKED pending specificity Gate
```

## Three materially different execution principles after this correction

The next constructive round may investigate these principles, but none receives
credit from the random-geometry result.

### P1 — Persistent checkpoint-specific correction field

Break the assumption that a deep residual bank expires after one K=128 block.

One target sweep creates a checkpoint/session-specific field `B_seed`. If the
same *frozen* field certifies unseen future causal states over a horizon `H`,
the seed dense arithmetic can in principle amortize as roughly `K/H` before
refresh cost. A 10x arithmetic reduction requires a demonstrated reuse horizon
of at least `H >= 10K`, with all refreshes and successor-state work charged.

Before this experiment, an actual-checkpoint specificity control is mandatory:
the same bank must outperform vocabulary-permuted and cross-prompt banks under
a frozen finite coefficient envelope. Unbounded oracle feasibility alone is no
longer evidence.

### P2 — Finite certificate transition automaton

Break the assumption that each token's certificate must be solved from the
target token or high-dimensional target state.

Treat the previous finite certificate as causal state:

```text
q_t = (support IDs, coefficient words, certified margin, committed token)
q_{t+1} = Phi(q_t, cheap causal state, new token suffix)
```

`Phi` must be compiled only on build traces, emit every word before target
continuation, and pass untouched holdouts. Its storage, instruction count,
state traffic, misses, and repairs are charged. It is useful only if it also
feeds a verifier whose fine arithmetic fraction is < 0.1 for the first 10x Gate.

### P3 — Witness-directed exact target evaluation

Break the verification unit itself. Do not run one complete dense target column
and then ask which token won.

A causal generator first emits a finite witness set of checkpoint operations and
competitors. The executor computes those exact witnesses and an exact enclosure
for everything omitted. It may commit only when:

```text
lower_bound(winner) > max upper_bound(competitors)
```

The crucial difference from generic progressive reading is that the witness
selection must be checkpoint-specific and causal, and the Gate charges the
actual dense operations required to produce the bound. Promotion requires
`r <= 0.1` on every untouched holdout with zero repair/fallback laundering.

## Selected next Gate

P1 gets the cheapest next test because it can falsify a 10x opportunity before
building a causal automaton or partial-target kernel.

The next official-checkpoint Gate will:

1. generate all seed banks before any delayed target continuation;
2. freeze each seed bank;
3. compare same-session, cross-prompt, and vocabulary-permuted banks;
4. use finite coefficient limits frozen from build-only data, not target-seeing
   unbounded LP capacity;
5. test untouched future offsets, including an offset at or beyond `10K`;
6. report exact certificate coverage, support/word traffic, `A`, `N`, and the
   implied dense refresh fraction;
7. reject the mechanism if controls certify equally well or if the same bank
   cannot survive the 10K horizon;
8. make no causal-generator or final-runtime claim from an oracle-only pass.

## Claim boundary

This E0 audit contains no public-checkpoint forward pass and no 405B execution.
It does not invalidate the mathematical correctness of EXP-096A. It invalidates
only the stronger interpretation that its sparse minimum-L1 supports, without
specificity controls, are evidence for reducing the dense target arithmetic
fraction `r`.
