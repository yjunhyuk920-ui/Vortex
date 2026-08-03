# Assumption Register

No unverified assumption may be used as a success condition.

## A-001 — Signed tile cancellation is broadly exploitable

Assumption:

Real Transformer decision-tile contributions permit valid early certification under causal randomized order.

Current evidence:

- Prior deterministic residual work observed cancellation but needed 90–98% refinement.
- EXP-047 global-range synthetic run certified 4/525 cases and evaluated about 98% of tiles overall.
- A strong synthetic positive control certified after 10.449% tiles.

Status: WEAKENED / UNVERIFIED ON REAL MODELS.

Contradiction test:

EXP-047R oracle-tight held-out real-checkpoint audit. Reject range-only CPTC if oracle-tight median evaluated fraction >10% or p90 >25%.

Dependencies: EXP-047R.

## A-002 — Alpha-spending Serfling implementation is valid under declared assumptions

Assumption:

Sampling without replacement plus fixed-step Serfling intervals and `delta_n = delta*6/(pi^2 n^2)` is valid under adaptive stopping when every contribution lies in the declared range.

Current evidence:

10 unit/property tests; 525 cases; zero independent-bound mismatches; zero wrong accepts in the corpus; adversarial exact fallback 15/15.

Status: SUPPORTED AT PHASE B / E1. Mathematical and implementation scope only.

Remaining risk:

A real checkpoint must derive sound ranges without reading all skipped tiles.

Dependencies: EXP-047R.

## A-003 — Certificate overhead is smaller than skipped work

Assumption:

Selector/statistics cost is materially smaller than evaluated dense work.

Current evidence:

Python Phase-B optimized path was about 8.8–9.1x slower than simple full summation in measured buckets, while fallback was 99.238%.

Status: CONTRADICTED FOR CURRENT PYTHON CPTC-v1; UNVERIFIED FOR VECTORIZED ACCELERATOR IMPLEMENTATION.

Contradiction/promotion test:

Charge selector operations and real-operation wall clock. No hardware projection may override current measured CPU evidence.

Dependencies: EXP-047R and later real replacement.

## A-004 — A decision-relevant low-dimensional projection is sufficient

Assumption:

A token/operator decision can be certified without reconstructing every omitted output coordinate.

Current evidence:

LM-head top-1 certificate primitives and constructed decision bounds exist, but model-wide nonlinear propagation is unresolved.

Status: PARTIALLY SUPPORTED / UNVERIFIED MODEL-WIDE.

Contradiction test:

Real operation replacement requires near-complete hidden-vector precision before final certification.

## A-005 — Probabilistic certification is acceptable

Assumption:

A union-bounded wrong-commit mode is acceptable alongside strict exact fallback mode.

Current evidence:

No final product acceptance criterion. EXP-047 uses `delta=1e-8` per synthetic decision and observed zero wrong accepts, but this is not model-wide union accounting.

Status: UNVERIFIED REQUIREMENT.

Contradiction test:

Strict bitwise exactness is mandatory or usable model-wide delta makes all certificates ineffective.

## A-006 — Small-model certification trends predict larger models

Current evidence: none.

Status: UNVERIFIED FOR LARGE MODELS.

Contradiction test: same held-out protocol across at least three sizes, then 30B/70B/405B in later phases.

## A-007 — Target RAM/SSD capacity and bandwidth are sufficient

Current evidence: small mmap tests and formulas only.

Status: UNVERIFIED; Phase D NOT TESTED.

## A-008 — Full hot state fits 8 GiB including KV/buffers/fallback

Current evidence: no complete architecture and no target GPU measurement.

Status: UNVERIFIED; E0.

## A-009 — 4B-class speed can coexist with exact fallback

Current evidence:

Prior repair mechanisms failed; CPTC-v1 fallback was 99.238% in synthetic cases.

Status: HIGH-RISK AND CURRENTLY UNSUPPORTED.

Contradiction test:

Measured fallback/cold bytes exceed target budget. A simple same-bit comparison requires about 1.185% average target weight evaluation before overhead.

## A-010 — Auxiliary DAG/VM components aid the final runtime

Current evidence: bounded functional components only.

Status: OPTIONAL/UNVERIFIED. They must not constrain core research.

## A-011 — Loose range metadata, not intrinsic tile behavior, caused CPTC-v1 failure

Assumption:

Per-state oracle-tight or deployable stratified bounds materially reduce certificate sample fractions.

Current evidence:

Not tested. Current global range was deliberately broad `[-1,1]`.

Status: ACTIVE UNVERIFIED ASSUMPTION.

Contradiction test:

EXP-047R compares global, oracle-tight, and deployable stratified ranges on held-out real-checkpoint tile contributions.

Decision:

If oracle-tight C1 remains above rejection thresholds, reject range-only CPTC rather than tuning sample fractions.

## A-012 — Sound static tile metadata can be computed automatically

Assumption:

Checkpoint-derived tile norms or tighter bounds can be precomputed without training/model modification, stored compactly, and combined with the current activation without reading skipped weights.

Current evidence: formula candidates only.

Status: UNVERIFIED.

Contradiction test:

Metadata is too large, runtime activation bounds too loose, or construction requires forbidden calibration/training.

Dependencies: EXP-047R C2.
