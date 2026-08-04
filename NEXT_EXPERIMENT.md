# Next Experiment

## Closed Gate — EXP-067

EXP-067 changed execution class from single-matrix factorization to exact common arithmetic across projections evaluated from the same activation.

Authoritative result:

```text
24/24 complete Q/K/V groups
10,752 Q4 rows
exact reusable rows: 0
maximum reusable-row fraction: 0%
operation p50/p90: 100% / 100%
storage p50/p90: 107.4142% / 114.1204%
common-right rank p50/p90: 100% / 100% of input width
```

Decision:

```text
REJECT_REAL_Q4_EXACT_JOINT_ROW_REUSE_AS_CORE_RETAIN_GROUP_CERTIFIER_AUXILIARY
```

Exact equality, sign, primitive integer-proportional row reuse, registered repeated row blocks, and exact common low-width right factors are closed as the primary core for the measured population. No joint kernel or model integration is authorized.

## EXP-068 — Oracle Global Demand-Certificate Lower-Bound Gate

### Execution-class change

EXP-068 does not seek static structure in the weights. It asks whether the current activation and final token margin allow most dense weight tiles to remain unread while an exact fail-closed certificate proves the same greedy token.

The candidate execution model is:

```text
read a subset of weight tiles
compute exact contributions for those tiles
bound every unread tile's maximum effect on the final target-token margin
commit only when the remaining global bound cannot change the token
otherwise expand and eventually fall back to the exact full model
```

This is activation-conditioned demand-driven execution. It is distinct from static compression, fixed drafting, and single-layer progressive LM-head certification.

### E0 prior

Potential upside:

- query-dependent reads rather than full parameter streaming;
- exact fail-closed fallback;
- no model modification or training;
- a genuine route to reducing both weight traffic and arithmetic if global margins are large and sensitivities concentrate.

Reasons for low prior probability:

- exact intermediate activations feed nonlinear residual depth, so local uncertainty can propagate widely;
- dense learned weights have repeatedly shown general-matrix structure;
- previous range and progressive certificates often required substantial residual reads;
- a final token margin may be small even when many local outputs look stable.

### Cheapest decisive experiment

Before implementing a scheduler or kernel, compute a deliberately favorable non-deployable oracle lower bound on the amount of exact tile work required.

For pinned small checkpoints and exact greedy target traces:

1. run the full target once to capture the exact committed token and final logit margin;
2. partition every dense projection into fixed registered tiles;
3. compute each tile's exact contribution on the recorded activation;
4. assign a sound upper bound to the effect of omitting that contribution on the final winning-logit margin through the remaining network;
5. let an oracle reveal tiles in the most favorable order;
6. stop only when the sum of all unread influence bounds is strictly below the exact final margin;
7. report the minimum favorable tile/byte fraction required for certification.

A weak but sound bound is acceptable for rejection. No deployable scheduling claim is made.

### Registered bound families

EXP-068 may evaluate only these bounded variants:

```text
A. norm-product global influence bound
B. exact local contribution magnitude with registered downstream operator-norm products
C. residual-path-separated bound where mathematically valid
```

No learned sensitivity predictor, activation table, target-future oracle at runtime, or unbounded bound search is allowed.

### Population

Use unchanged pinned TinyStories-1M/3M/8M revisions and the existing held-out prompt families. Every exact reference token must match the standard full-model greedy replay.

Required coverage includes:

```text
English narrative
code
mathematics
identifier boundary
Korean
structured JSON
```

### Favorable accounting

Charge:

```text
all revealed target weight bytes
all revealed tile multiply-accumulates
activation and bound metadata reads
bound aggregation work
full exact fallback work when certification fails
one-time oracle ordering is reported separately and excluded only as a favorable upper bound
```

Report both per-token and family-level p50/p90 fractions. A certificate that saves arithmetic but not weight traffic, or vice versa, does not pass.

### Controls

- exact full replay token matches the registered target;
- a synthetic large-margin sparse-influence network certifies early;
- a late-flip adversarial residual chain requires all decisive tiles;
- one unread tile capable of flipping the winner prevents commitment;
- every bound is checked against exact omission effects on bounded synthetic cases;
- malformed or non-finite bounds fail closed;
- zero target-future information is available to a deployable path.

### Promotion Gate

```text
zero exact-reference/control/bound violation
100% required prompt-family coverage
p50 favorable target weight-byte fraction <=10%
p90 favorable target weight-byte fraction <=25%
p50 favorable target operation fraction <=10%
p90 favorable target operation fraction <=25%
zero incorrect early commitments
no required family with p90 byte fraction >25%
no largest-model degradation >25%
```

Passing the oracle lower-bound Gate authorizes only a deployable certificate/scheduler Gate. It does not authorize a physical kernel claim.

### Failure decision

```text
REJECT_GLOBAL_DEMAND_CERTIFICATE_AS_CORE_RETAIN_BOUND_AUDITOR_AUXILIARY
```

On failure, registered norm-based exact lazy execution is closed as a primary core. The next candidate must introduce a new exact information source or a stronger theorem, not merely tune tile sizes or orderings.

### Stop rule

Before survival, prohibit:

```text
CUDA kernels
model-wide lazy runtime integration
learned tile predictors
approximate commitments
unbounded tile-size/order sweeps
405B implementation work
```

### Claim boundary

Phase A/B/C small-model oracle-bound evidence, ceiling E1. A deployable scheduler, actual skipped Transformer operations, 405B execution, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, and tokens/second remain NOT TESTED.
