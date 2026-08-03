# EXP-047 — Causal Probabilistic Tile Certificate

## Final decision for this Gate

```text
Phase: A/B
Evidence: E1
Correctness primitive: ACCEPT
Global-range CPTC-v1 core architecture: REVISE
Real-model operation replacement: NOT TESTED
Phase D: NOT TESTED
```

Authoritative evidence:

```text
PR: #56
workflow: 30791851508
source head: d395d0eada15fd7ef9b09ce5ccb561a921bb6b7b
results: results/exp_047/
```

## Motivation

Prior deterministic signed-residual certificates observed cancellation but still required roughly 90–98% refinement. EXP-047 tested a materially different correctness regime:

> sample weight-tile decision contributions without replacement in a causal random order, use an anytime-valid finite-population confidence interval, and exact-fallback when the interval does not certify the decision.

## Direct connection to the target

For a linear operation:

```text
y = W x
W = [W_1 ... W_N]
x = [x_1 ... x_N]
y = sum_i W_i x_i
```

For decision projection `q`:

```text
z_i = q^T W_i x_i
M = M_base + sum_i z_i
```

If the sign of `M` can be certified after evaluating a small subset of tiles, the runtime can skip the omitted original weight reads and arithmetic. If not, it evaluates every remaining tile and returns the full reference decision.

Phase B receives synthetic scalar contributions. It does not prove that sound real-checkpoint bounds or useful skip fractions exist.

## Causal and fallback contract

Permitted:

- current activation/state;
- current decision projection;
- checksummed static tile metadata;
- current random permutation;
- observed tile contributions;
- confidence state.

Forbidden:

- future generated tokens;
- target continuation;
- future hidden states;
- secretly computing the exact answer before the optimized stop.

Invalid bounds, non-finite state, or failure to certify cause rejection or exact full-tile fallback.

## Implemented Phase-A certificate

For finite population `z_1,...,z_N in [a,b]`, sampled uniformly without replacement, the reference uses a two-sided Serfling total radius:

```text
R_n = N (b-a)
      sqrt((1 - (n-1)/N) log(2/delta_n) / (2n))
```

Adaptive stopping uses alpha spending:

```text
delta_n = delta_total * 6 / (pi^2 n^2)
sum_n delta_n <= delta_total
```

Accept positive when:

```text
M_base + N mean_n - R_n > 0
```

Accept negative when:

```text
M_base + N mean_n + R_n < 0
```

Otherwise evaluate all remaining tiles exactly.

This is a probabilistic certificate under the declared range and random-sampling assumptions, not deterministic exactness. Exact fallback is deterministic under the reference floating-point contract.

## Phase-B implementation

Files:

```text
vortex_runtime/cptc.py
tests/exp_047/test_cptc.py
experiments/exp_047/
.github/workflows/exp_047_gate.yml
results/exp_047/
```

Tests cover:

- independent alpha/radius calculation;
- positive and negative controls;
- exact zero margin;
- one dominant tile;
- misleading positive sample prefix with negative exact total;
- 200 randomized property cases in unit tests;
- deterministic replay;
- invalid bound/configuration and NaN/Inf rejection.

The experiment runner adds 525 measured cases across populations 64–1024, raw case logs, scaling output, stdout, summary, and checksums.

## Authoritative MEASURED result

```text
unit/property tests: 10 passed
cases: 525
certified decisions: 4
exact fallbacks: 521 = 99.238%
wrong certified accepts: 0
fallback/reference mismatches: 0
independent-bound mismatches: 0
adversarial exact fallback: 15/15
future generated tokens used: false
Phase D: NOT TESTED
```

Largest positive cancellation control:

```text
population: 1,024 tiles
certificate after: 107 tiles
sample fraction: 10.449%
reference decision agreement: pass
```

Broad scaling:

| Tiles | Certified fraction | Fallback fraction | Mean evaluated fraction |
|---:|---:|---:|---:|
| 64 | 0% | 100% | 100% |
| 128 | 0% | 100% | 100% |
| 256 | 0% | 100% | 100% |
| 512 | 1.905% | 98.095% | 98.519% |
| 1,024 | 1.905% | 98.095% | 98.294% |

Measured Python optimized/reference mean time was roughly 8.8–9.1x in the size buckets. This is current CPU implementation evidence only.

## DERIVED and PROJECTED target gap

For 405B Q4 versus 4B Q4:

```text
405B full Q4 stream: 188.593 GiB
4B full Q4 stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required average evaluated fraction before selector/fallback: 1.185%
```

The positive-control 10.449% fraction is 8.817x above that simple pre-overhead target. The broad corpus mean near 98% is far worse.

These values are PROJECTED from parameter counts, not measured target traffic.

## Interpretation

### Accepted

- causal sample order;
- fixed-step Serfling plus alpha-spending implementation;
- independent interval verification;
- deterministic replay;
- invalid-state rejection;
- exact fallback correctness in the synthetic corpus.

### Not accepted

- useful general skip coverage;
- selector performance advantage;
- real Transformer operation skipping;
- model-wide nonlinear correctness;
- 405B scaling;
- 8 GiB execution;
- target PCIe/SSD or wall clock.

## Decision

> Phase B, E1: the finite-population certificate and exact fallback were correct on the committed synthetic corpus. The current global-range form certified only 4/525 cases and evaluated about 98% of tiles overall, so it is not promoted as the core executor.

The pre-registered primitive Gate passed because the positive control certified below 25% and correctness checks passed. The architecture-level interpretation is stricter: **REVISE**, not performance promotion.

## Next decisive Gate

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit`, defined in `NEXT_EXPERIMENT.md`.

It will use available unmodified small checkpoints and held-out prompts to compare:

- current global range;
- non-deployable exact per-state oracle range;
- deployable static stratified checkpoint-derived bounds;
- independently justified variance-adaptive bounds.

If even the oracle-tight certificate requires high tile fractions, range-only CPTC is rejected rather than tuned.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q tests/exp_047
bash experiments/exp_047/reproduce.sh
```

Authoritative raw files and hashes are in `results/exp_047/`.
