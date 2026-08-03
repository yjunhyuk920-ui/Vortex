# EXP-047 — Causal Probabilistic Tile Certificate

## Final Gate classification

```text
Phase A/B
Evidence E1
Correctness primitive ACCEPT
Global-range CPTC-v1 architecture REVISE
Real-model operation replacement NOT TESTED
Phase D NOT TESTED
```

Authoritative machine-readable evidence:

```text
results/exp_047/summary.json
results/exp_047/raw/cases.jsonl
results/exp_047/checksums.sha256
```

Current frozen summary records:

```text
PR #56
workflow 30793232558
source SHA 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
```

## Mechanism

For a linear operation partitioned by input dimension:

```text
y = W x = sum_i W_i x_i
z_i = q^T W_i x_i
M = M_base + sum_i z_i
```

sample decision contributions `z_i` uniformly without replacement in a causal random order. Commit only when an anytime-valid interval excludes zero. Otherwise evaluate every remaining tile and return the full reference sign.

Phase B uses synthetic contributions. It does not establish sound real-checkpoint bounds or useful real-model skipping.

## Causal/safety contract

Permitted: current state, current decision projection, checksummed static metadata, random permutation, observed tiles, and confidence state.

Forbidden: future generated tokens, target continuation, future hidden states, or secretly computing the exact result before early stop.

Invalid range, non-finite state, or absent certificate triggers rejection or exact full-tile fallback.

## Confidence sequence

For `N` contributions in `[a,b]`:

```text
R_n = N (b-a)
      sqrt((1 - (n-1)/N) log(2/delta_n) / (2n))
```

Adaptive alpha spending:

```text
delta_n = delta_total * 6/(pi^2 n^2)
sum_n delta_n <= delta_total
```

Positive accept:

```text
M_base + N mean_n - R_n > 0
```

Negative accept:

```text
M_base + N mean_n + R_n < 0
```

Otherwise exact fallback. Early acceptance is probabilistic under declared assumptions, not deterministic exactness.

## Files

```text
vortex_runtime/cptc.py
tests/exp_047/test_cptc.py
experiments/exp_047/
.github/workflows/exp_047_gate.yml
results/exp_047/
```

## Frozen MEASURED result

```text
10 tests passed
525 cases
certified 4
fallback 521 = 99.238%
wrong accepts 0
fallback mismatches 0
independent-bound mismatches 0
adversarial exact fallback 15/15
future generated tokens false
Phase D NOT TESTED
```

Positive control:

```text
107/1024 tiles = 10.449%
```

Scaling:

| Tiles | Certified | Fallback | Mean evaluated |
|---:|---:|---:|---:|
| 64 | 0% | 100% | 100% |
| 128 | 0% | 100% | 100% |
| 256 | 0% | 100% | 100% |
| 512 | 1.905% | 98.095% | 98.519% |
| 1,024 | 1.905% | 98.095% | 98.294% |

Python optimized/reference mean time was about 8.6–9.1x. This is CPU implementation evidence only.

## PROJECTED target gap

```text
405B Q4 stream 188.593 GiB
4B Q4 stream 1.863 GiB
1.2x allowance 2.235 GiB/token
required average evaluated fraction before overhead 1.185%
positive-control fraction / target 8.817x
```

These are parameter-count projections, not measured target traffic.

## Interpretation

Accepted at E1:

- causal sample order;
- alpha-spending Serfling implementation;
- independent interval verification;
- deterministic replay;
- invalid-state rejection;
- exact fallback correctness in the corpus.

Not accepted:

- useful broad skip coverage;
- selector speed advantage;
- real Transformer operation replacement;
- deployable checkpoint-derived ranges;
- model-wide nonlinear certification;
- 70B/405B trend;
- 8 GiB or target wall clock.

## Decision

> Phase B, E1: the certificate and exact fallback were correct on the committed synthetic corpus. The global-range form certified only 4/525 cases and evaluated about 98% of tiles overall, so it is not promoted as the core executor.

The pre-registered primitive Gate passed. The architecture decision is **REVISE**.

## Next Gate

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit` in `NEXT_EXPERIMENT.md`.

On held-out current-token states from available unmodified small checkpoints compare:

- current global range;
- exact per-state min/max as a non-deployable oracle;
- deployable checkpoint-derived stratified bounds;
- independently justified variance-adaptive finite-population bounds.

If oracle-tight intervals still require high tile fractions, reject range-only CPTC rather than tune thresholds. Offline audit remains below E2.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q tests/exp_047
bash experiments/exp_047/reproduce.sh
cd results/exp_047 && sha256sum -c checksums.sha256
```
