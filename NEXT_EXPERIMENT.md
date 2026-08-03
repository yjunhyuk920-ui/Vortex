# Next Experiment

## EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit

### Classification

- core research: yes;
- previous EXP-047 Phase-B correctness primitive: accepted E1;
- current architecture decision: REVISE;
- next phase: Phase C analysis/falsification on available unmodified small checkpoints;
- evidence ceiling before real operation replacement: E1, even though real checkpoints are inspected;
- Phase D: NOT TESTED.

No new experiment number is created merely to hide CPTC-v1's weak skip rate.

## Why revision is mandatory

Authoritative EXP-047 run `30791851508` measured:

```text
525 cases
4 certificates
521 exact fallbacks
mean evaluated fraction at N=1024: 98.294%
positive control: 107/1024 tiles = 10.449%
required simple 405B traffic fraction before overhead: about 1.185%
```

The correctness mechanism passed, but the global-range Serfling form is not a plausible primary executor.

## Decisive question

> Is CPTC failing because real Transformer decision-tile contributions are intrinsically uncertifiable early, or because one global range bound is too loose?

The next Gate must answer this before any deployable Phase-C backend is built.

## Real-checkpoint audit scope

Use several available unmodified dense causal checkpoints, aiming for at least three sizes when GitHub storage/time permit.

For held-out prompts and current-token states:

1. run the exact baseline forward pass;
2. select a declared scalar decision, initially LM-head winner versus runner-up;
3. partition the corresponding dot-product difference into input-dimension tiles;
4. save exact tile contributions only as a non-deployable analysis oracle;
5. replay causal random tile orders and compare certificate families.

The exact baseline and full contributions are used only to measure the upper bound and catch wrong accepts. They may not be hidden from forward-call or cost accounting.

## Certificate families to compare

### C0 — global range Serfling

Current EXP-047 implementation. Expected control baseline.

### C1 — oracle-tight global range

Use the exact min/max of the current state's full contribution vector. This is future/full-work oracle metadata and is not deployable.

Purpose:

- strongest quick falsification of the range looseness hypothesis;
- if C1 still needs nearly all tiles, reject range-only CPTC as a core path.

### C2 — causal stratified static bounds

Group tiles using checkpoint-derived, checksummed static metadata. Candidate bound:

```text
|q^T W_i x_i| <= ||q||_2 ||W_i||_F ||x_i||_2
```

or a tighter operator-norm/block bound if computable without reading the skipped tile at runtime.

Sample and certify strata separately, then combine intervals with explicit delta allocation.

### C3 — empirical-Bernstein/variance-adaptive finite-population bound

Only include after the mathematical adaptive-stopping contract is independently derived and tested. Do not import an asymptotic IID bound into sampling-without-replacement without proof.

## Required measurements

MEASURED:

- checkpoint ID/revision and bytes;
- prompt split and hashes;
- exact baseline forward calls;
- current/future information audit;
- layer/operator/tile counts;
- exact winner/runner-up logit margin;
- per-certificate accepted fraction;
- sampled/evaluated tile fraction;
- wrong accepts;
- fallback;
- CPU time and peak RSS;
- static metadata construction time/bytes;
- model-size trend.

DERIVED:

- union error budget;
- selector operations;
- logical weight bytes saved under each certificate;
- exact fallback cost.

PROJECTED:

- 405B weight fraction and bytes/token under the measured fractions;
- gap to 1.185% traffic fraction;
- memory requirements for static bounds.

UNVERIFIED:

- GPU kernel overhead;
- target PCIe/SSD behavior;
- 8 GiB execution;
- 70B/405B quality and wall clock.

## Future-information audit

Every result row must state:

```text
uses_exact_full_contributions
uses_exact_baseline_winner
uses_future_generated_tokens
is_deployable
```

C1 is intentionally non-deployable. Future generated tokens remain forbidden for every family.

## Pre-registered decision thresholds

### Reject range-only CPTC core

Reject C0/C1 as a primary mechanism if, on held-out real states:

- oracle-tight C1 median evaluated fraction >10%; or
- C1 p90 >25%; or
- any wrong accept occurs; or
- certificate overhead plus fallback exceeds full exact reference cost.

The 10% threshold is still far above the final 1.185% target; it is only an early rejection threshold.

### Continue stratified-bound research

Continue C2 only when:

- zero wrong accepts;
- median deployable evaluated fraction materially improves over C0;
- metadata and selector cost are fully charged;
- at least one held-out task family has nonzero certified coverage;
- trend does not worsen with checkpoint size.

### Real operation replacement promotion

Only after the audit shows a meaningful deployable certificate should a subsequent Gate replace the actual LM-head operation during generation. Offline contribution analysis alone does not qualify as E2.

## Strongest falsification

Search held-out states with:

- tiny exact top-1 margin;
- late aligned contributions;
- heavy-tailed tiles;
- winner changes under partial evaluation;
- domain/language shift;
- code, mathematics, Korean, English, structured output, and long context.

If even oracle-tight intervals do not close early, reject the current statistical-certificate family rather than tuning thresholds.

## Required implementation additions

```text
docs/research/EXPERIMENT_047R_ORACLE_TIGHT_TILE_BOUND_AUDIT.md
experiments/exp_047r/
results/exp_047r/
tests/exp_047r/
.github/workflows/exp_047r_gate.yml
```

Reuse EXP-047 reference code and provenance schema. Do not overwrite EXP-047 raw evidence.
