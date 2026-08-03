# EXP-047 — Causal Probabilistic Tile Certificate

## Status

- core research;
- Phase A/B active;
- evidence ceiling in this branch: E1;
- Phase C: NOT TESTED;
- Phase D: NOT TESTED.

## Motivation

Prior deterministic signed residual certificates observed real cancellation but remained too conservative: roughly 90–98% of residual work still had to be refined, producing projected traffic hundreds of GiB/token.

EXP-047 does not change only the partition size or norm. It tests a different correctness regime:

> evaluate weight tiles in a causal random order and use an anytime-valid finite-population confidence sequence to exploit observed cancellation. If confidence does not close, evaluate every remaining tile and return the exact result.

## Direct objective connection

Original operation:

```text
y = W x
```

Partition by input dimension:

```text
W = [W_1 ... W_N]
x = [x_1 ... x_N]
y = sum_i W_i x_i
```

For a declared scalar decision projection `q`, tile `i` contributes:

```text
z_i = q^T W_i x_i
```

The decision margin is:

```text
M = M_base + sum_i z_i
```

EXP-047 attempts to decide the sign of `M` after reading only a subset of `W_i`, thereby skipping original dense tile reads and multiply-accumulates.

The Phase-B primitive receives synthetic `z_i` values. A later Phase-C implementation must derive them from an actual unmodified Transformer operation and must account for how `q`, `M_base`, and sound tile bounds are obtained.

## Causal information contract

Permitted before token commit:

- current activation/state;
- current candidate decision projection;
- checksummed static tile-bound metadata;
- current random tile permutation seed;
- contributions from tiles already evaluated;
- accumulated confidence state.

Forbidden:

- future generated tokens;
- target continuation;
- future hidden states;
- full exact result used secretly to select or validate the early stop.

The Phase-B validator computes the exact reference separately only after the optimized logical decision has been produced.

## Phase-A theorem used by the reference implementation

Assume a finite population of `N` scalar tile contributions:

```text
z_1, ..., z_N in [a, b]
```

Sample uniformly without replacement. At fixed sample count `n`, let `mean_n` be the sample mean and `mu` the population mean.

The implemented two-sided Serfling form uses:

```text
P(|mean_n - mu| >= epsilon)
  <= 2 exp(-2 n epsilon^2 / ((1 - (n-1)/N) (b-a)^2))
```

For fixed-step failure probability `delta_n`, the total-sum radius is:

```text
R_n = N (b-a)
      sqrt((1 - (n-1)/N) log(2/delta_n) / (2n))
```

and therefore:

```text
sum_i z_i in [N mean_n - R_n, N mean_n + R_n]
```

To allow adaptive stopping, allocate:

```text
delta_n = delta_total * 6 / (pi^2 n^2)
```

Since:

```text
sum_n delta_n <= delta_total
```

a union bound makes all fixed-step intervals jointly valid with probability at least `1-delta_total`.

The implementation accepts positive when:

```text
M_base + N mean_n - R_n > 0
```

and negative when:

```text
M_base + N mean_n + R_n < 0
```

Otherwise it continues until the configured early-stop limit, then evaluates every remaining tile exactly.

## Correctness classification

### Probabilistic certified commit

If the declared range is sound and random sampling assumptions hold, the wrong-commit probability for one decision is bounded by `delta_total` through alpha spending.

This is not deterministic exactness.

### Exact fallback

When no certificate closes, every unsampled tile is evaluated. The returned sign equals the full reference sum, subject only to the declared floating-point reference contract.

### Invalid metadata or numeric state

Non-finite values, invalid bounds, out-of-range observed values, or malformed configuration cause rejection. They are not accepted as approximate results.

## Unresolved sound-bound problem

The Phase-B experiment receives a declared `[a,b]`. A real linear operator must derive a valid bound without reading every tile weight.

One conservative future candidate is:

```text
|q^T W_i x_i|
  <= ||q||_2 ||W_i||_F ||x_i||_2
```

where `||W_i||_F` is precomputed and checksummed. This may be too loose. No Phase-C success is claimed until a real bound derivation, metadata cost, and actual operation replacement pass.

This dependency is registered as A-001 through A-004.

## Selector and metadata accounting

For `n` sampled tiles before certification:

```text
weight reads: n tiles
sample updates: O(n)
state: sampled sum, n, range, delta schedule, permutation state
static metadata: at least tile-bound data plus integrity metadata
fallback: N-n additional tile reads and exact accumulation
```

Full expected cost:

```text
B/token = B_selector + E[f_certified] B_sampled
          + P(fallback) B_full_remaining
```

No selector, metadata, permutation, or fallback work may be omitted from later accounting.

## 405B traffic gap

Using the Phase-B config:

```text
target parameters: 405,000,000,000
baseline parameters: 4,000,000,000
bits/weight: 4
```

DERIVED full sequential Q4 streams:

```text
405B: about 188.59 GiB
4B:   about   1.86 GiB
1.2x 4B allowance: about 2.24 GiB/token
```

Before selector and fallback, the target average evaluated-weight fraction must satisfy approximately:

```text
fraction <= 1.2 * 4B / 405B ~= 1.185%
```

Therefore the Phase-B <=25% positive-control threshold is only a primitive sanity check. It remains roughly twenty-one times above the ultimate traffic fraction and is not a target-performance claim.

## Pre-registered Phase-B cases

- positive signed cancellation;
- negative signed cancellation;
- exact zero margin;
- one dominant tile;
- a misleading positive sample prefix with negative exact total;
- randomized bounded populations;
- increasing population sizes 64–1024;
- invalid range/configuration and NaN/Inf fault injection;
- deterministic fixed-seed replay.

## Promotion criteria

All are required:

- zero silent wrong accepts in the committed corpus;
- zero fallback/reference mismatches;
- independent formula implementation matches every accepted interval;
- largest positive control certifies after <=25% tiles;
- zero-margin, dominant, and misleading-prefix adversaries fall back to all tiles;
- result JSON separates MEASURED/DERIVED/PROJECTED/UNVERIFIED;
- Phase D remains `NOT TESTED`;
- workflow and raw checksums are committed.

A pass promotes only to Phase-C design, not to target feasibility.

## Rejection criteria

Reject or revise when:

- the independent bound disagrees;
- a wrong certified accept occurs in the Gate corpus;
- adversarial cases are silently accepted incorrectly;
- certificate overhead exceeds saved work even synthetically;
- positive controls cannot close substantially before full evaluation;
- the real-bound dependency cannot be made sound without reading most weights;
- Phase-C real operations require nearly every tile at useful error budgets.

## Strongest falsification

The most dangerous case is an adaptive-looking stable partial mean whose unseen bounded tiles collectively flip the exact sign. The misleading-prefix test constructs this explicitly under the fixed permutation. A valid confidence sequence must refuse early commitment and exact-fallback.

Future real-model falsification must search held-out states for late aligning tile contributions and record the full certificate trajectory.

## Files

```text
vortex_runtime/cptc.py
tests/exp_047/test_cptc.py
experiments/exp_047/
.github/workflows/exp_047_gate.yml
results/exp_047/
```

## Required communication after the Gate

Use wording equivalent to:

> Phase B, E1: the finite-population certificate matched the independent reference on the tested synthetic cases and exact-fallback handled adversarial cases. Real Transformer operation skipping, sound checkpoint-derived bounds, 405B scaling, 8 GiB execution, and target speed remain unverified.
