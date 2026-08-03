# Next Experiment

## EXP-047R — Oracle-Tight and Stratified Tile-Bound Audit

### Current status

```text
Implementation branch: research/exp-047r-oracle-stratified-audit
Gate registration: COMMITTED
Primitive tests: NOT YET RUN IN GITHUB ACTIONS
Real-checkpoint audit: NOT YET RUN
Scientific decision: PENDING
Evidence ceiling before real operation replacement: E1
Phase D: NOT TESTED
```

No new experiment number is being used to hide CPTC-v1's weak skip rate.

## Previous authoritative evidence

Authoritative EXP-047 evidence:

```text
PR: #56
workflow: 30793232558
source SHA: 74ac92e9b1c8fffbc50a2322d9b36dd3c05f0d79
phase: A/B
evidence: E1
```

MEASURED:

```text
525 cases
4 certificates
521 exact fallbacks
mean evaluated fraction at N=1024: 98.294%
positive control: 107/1024 tiles = 10.449%
wrong accepts: 0
```

PROJECTED:

```text
required simple 405B Q4 traffic fraction before overhead: about 1.185%
```

The correctness mechanism passed, but the global-range Serfling form did not earn promotion as an execution architecture.

## Decisive question

> Is CPTC failing because real Transformer decision-tile contributions are intrinsically uncertifiable early, or because one global range bound is too loose?

The next Gate answers this before any deployable Phase-C backend is built.

## Real-checkpoint audit scope

Use three available unmodified trained dense causal checkpoints:

```text
roneneldan/TinyStories-1M
roneneldan/TinyStories-3M
roneneldan/TinyStories-8M
```

The runner resolves an exact Hugging Face revision SHA before downloading or executing each model, then records a file SHA-256 manifest. The first successful workflow must be inspected and its exact revisions frozen before evidence becomes authoritative.

For fixed held-out prompts and current-token states:

1. execute the exact baseline forward pass;
2. obtain the exact final hidden state and logits;
3. select top-1 versus runner-up as the declared scalar decision;
4. partition that exact pairwise margin into input-dimension tiles;
5. save exact contributions only as a non-deployable analysis oracle;
6. compare C0, C1, and C2 under deterministic causal random orders.

This is offline full-contribution observation, not real operation replacement and not E2.

## Certificate families

### C0 — global checkpoint-derived range

For output-weight matrix `W`, compile per-hidden-dimension column spans:

```text
s_j = max_o W[o,j] - min_o W[o,j]
```

For current hidden state `h` and tile `T_i`:

```text
|c_i| <= B_i = sum_{j in T_i} |h_j| s_j
```

C0 applies one range:

```text
[-max_i B_i, +max_i B_i]
```

The metadata is checkpoint-derived and the activation factor is available at the current token. It does not require reading the selected output rows to establish the bound.

### C1 — exact per-state oracle range

Use the exact min/max of the fully materialized current-state contribution vector. This is intentionally non-deployable and is the strongest favorable range-only control.

If C1 misses the rejection thresholds, further C0/C2 range metadata tuning stops.

### C2 — checkpoint-span stratified range

Group tiles by `B_i` magnitude. Sample without replacement inside each stratum and sum per-stratum Serfling intervals.

Union accounting:

```text
delta_s   = delta * 6 / (pi^2 (s+1)^2)
delta_s,n = delta_s * 6 / (pi^2 n^2)
```

Current code validates every bound against exact materialized contributions because this is an offline audit. A later deployable runtime would require checksummed metadata and fail-closed validation without consuming skipped rows.

### C3 — variance-adaptive finite-population bound

Status: `NOT IMPLEMENTED`.

C3 is forbidden from influencing the Gate until an independent proof, slow reference calculator, property tests, adaptive-stopping union accounting, and fault injection are committed. An IID or asymptotic empirical-Bernstein formula may not be imported without a valid sampling-without-replacement proof.

## Required measurements

MEASURED in the valid current environment:

- exact model/tokenizer revision and file hashes;
- prompt hashes;
- exact baseline forward states;
- layer/hidden/vocabulary/tile counts;
- top-1/runner-up logit margin;
- LM-head and tile reconstruction errors;
- C0/C1/C2 evaluated fractions;
- wrong accepts and bound violations;
- fallback/full-evaluation behavior;
- CPU primitive timing after contributions are materialized;
- peak RSS;
- metadata bytes and model-size trend.

DERIVED:

- two-dimensional union error budget;
- Gate booleans;
- decision classification;
- 405B fraction gap from measured oracle fractions.

PROJECTED:

- 405B Q4 weight bytes/token under the observed fractions;
- gap to the 1.185% pre-overhead traffic fraction.

UNVERIFIED:

- C3 correctness;
- real LM-head operation replacement;
- model-wide nonlinear propagation;
- GPU selector cost;
- PCIe/SSD behavior;
- 8 GiB execution;
- 70B/405B quality and wall clock.

## Future-information and deployability fields

Every row records the equivalent of:

```text
future_generated_tokens_used = false
real_operation_replacement = false
offline_full_contribution_oracle = true
```

C1 is non-deployable by definition. C0/C2 bounds are sound candidates, but the present audit still materializes full logits and contributions for validation and therefore is not a deployable executor.

## Pre-registered decision thresholds

### Reject range-based CPTC core

Reject the range family from the core execution path if any condition holds:

```text
C1 oracle median evaluated fraction >10%
C1 oracle p90 evaluated fraction >25%
wrong certified accept >0
checkpoint-derived bound violation >0
C2 materialized-contribution CPU selector/fallback median cost > full materialized sum
```

The 10% threshold is deliberately lenient and remains far above the final PROJECTED 1.185% requirement.

### Continue only if the oracle survives

Continue to independently proven C3 and then real operation replacement only when all rejection conditions are avoided. Offline observation alone cannot earn E2.

## Strongest falsification

The C1 exact per-state min/max oracle is tighter than any sound state range derived without already knowing every realized contribution. If C1 still requires high coverage, the range-only family is intrinsically too weak for the declared core role and must be retired rather than tuned.

## Active files

```text
docs/research/EXPERIMENT_047R_ORACLE_TIGHT_STRATIFIED_TILE_BOUND_AUDIT.md
vortex_runtime/cptc_audit.py
experiments/exp_047r/
tests/exp_047r/
.github/workflows/exp_047r_gate.yml
```

EXP-047 frozen evidence must not be overwritten.

## Next exact action

Open the branch PR, run `.github/workflows/exp_047r_gate.yml`, inspect the complete logs and uploaded candidate evidence, then classify the outcome as:

```text
CONTINUE_TO_INDEPENDENT_C3_AND_REAL_OPERATION_REPLACEMENT
REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY
INFRASTRUCTURE FAILURE — NO SCIENTIFIC DECISION
```
