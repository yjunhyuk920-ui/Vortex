# EXP-047 — Causal Probabilistic Tile Certificate

## Final Gate classification

```text
Phase: A/B
Evidence: E1
Correctness primitive: ACCEPT
Global-range CPTC-v1 core architecture: REVISE
Real-model operation replacement: NOT TESTED
Phase D: NOT TESTED
```

Final authoritative evidence:

```text
PR: #56
workflow: 30792813542
source implementation SHA: 08e8b35f48b1b616147f22dce046ab93218265c9
committed evidence head after workflow: 3359371762c004db3532ebb16872b4eee85accf6
results: results/exp_047/
```

## Mechanism

For a linear operation partitioned by input dimension:

```text
y = W x = sum_i W_i x_i
z_i = q^T W_i x_i
M = M_base + sum_i z_i
```

sample scalar decision contributions `z_i` uniformly without replacement in a causal random order. Commit a sign decision only when an anytime-valid interval excludes zero. Otherwise evaluate every remaining tile and return the full reference sign.

This attempts to skip original weight-tile reads and arithmetic. Phase B uses synthetic contributions and does not establish real Transformer bounds or savings.

## Causal and safety contract

Permitted before commit:

- current activation/state;
- current decision projection;
- checksummed static tile metadata;
- current random permutation and observed tiles;
- confidence state.

Forbidden:

- future generated tokens;
- target continuation;
- future hidden states;
- exact full result used secretly to select early stop.

Invalid range, non-finite state, or failure to certify triggers rejection or exact full-tile fallback.

## Implemented confidence sequence

For `N` finite contributions in `[a,b]`, the implemented fixed-step Serfling total radius is:

```text
R_n = N (b-a)
      sqrt((1 - (n-1)/N) log(2/delta_n) / (2n))
```

Adaptive stopping uses:

```text
delta_n = delta_total * 6/(pi^2 n^2)
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

Otherwise exact-fallback.

The early certificate is probabilistic under declared range/random-sampling assumptions. It is not deterministic exactness. The fallback path evaluates all contributions.

## Implementation and evidence files

```text
vortex_runtime/cptc.py
tests/exp_047/test_cptc.py
experiments/exp_047/
.github/workflows/exp_047_gate.yml
results/exp_047/
```

Raw evidence includes all 525 cases, processed scaling, stdout, summary, logs, artifact contract, and checksums.

## Final MEASURED result

```text
unit/property tests: 10 passed
cases: 525
certified: 4
fallback: 521 = 99.238%
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
reference agreement: pass
```

Broad scaling:

| Tiles | Certified | Fallback | Mean evaluated |
|---:|---:|---:|---:|
| 64 | 0% | 100% | 100% |
| 128 | 0% | 100% | 100% |
| 256 | 0% | 100% | 100% |
| 512 | 1.905% | 98.095% | 98.519% |
| 1,024 | 1.905% | 98.095% | 98.294% |

Measured Python optimized/reference mean time ratio was about 9.2–9.7x across size buckets. This is CPU implementation evidence, not accelerator projection.

## PROJECTED target gap

```text
405B Q4 full stream: 188.593 GiB
4B Q4 full stream: 1.863 GiB
1.2x allowance: 2.235 GiB/token
required average target weight fraction before selector/fallback: 1.185%
positive-control fraction / target fraction: 8.817x
```

These are parameter-count projections, not measured traffic.

## Interpretation

Accepted at E1:

- causal sample order;
- alpha-spending Serfling calculation;
- independent interval verification;
- deterministic replay;
- invalid-state rejection;
- exact fallback correctness on the corpus.

Not accepted:

- useful broad skip coverage;
- selector speed advantage;
- real Transformer operation replacement;
- deployable sound checkpoint-derived ranges;
- model-wide nonlinear certification;
- 70B/405B trend;
- 8 GiB execution or target wall clock.

## Decision

> Phase B, E1: the certificate and exact fallback were correct on the committed synthetic corpus. The global-range form certified only 4/525 cases and evaluated about 98% of tiles overall, so it is not promoted as the core executor.

The pre-registered primitive Gate passed because correctness checks and the <=25% positive control passed. The architecture-level decision is stricter: **REVISE**.

## Next Gate

`EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit` in `NEXT_EXPERIMENT.md`.

On held-out current-token states from available unmodified small checkpoints, compare:

- current global range;
- exact per-state min/max as a non-deployable oracle upper bound;
- deployable checkpoint-derived stratified bounds;
- only independently justified variance-adaptive finite-population bounds.

If oracle-tight intervals still require high tile fractions, reject range-only CPTC rather than tune thresholds.

Offline full-contribution analysis remains below E2. Actual operation replacement follows only after a deployable bound shows useful held-out coverage.

## Reproduction

```bash
git checkout research/governance-exp047-cptc
python -m pytest -q tests/exp_047
bash experiments/exp_047/reproduce.sh
cd results/exp_047 && sha256sum -c checksums.sha256
```
