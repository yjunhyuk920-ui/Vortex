# Assumption Register

No unverified assumption may be used as a success condition.

## A-001 — Signed tile cancellation is broadly exploitable by range certification

Assumption: real Transformer pair-margin tile contributions permit valid early range-based certification under causal randomized order.

Evidence:

- deterministic residual work required roughly 90–98% refinement;
- EXP-047 global-range synthetic run evaluated about 98% overall;
- EXP-047R exact per-state range oracle evaluated 100% at median and p90 across 18 states from three pinned trained checkpoints.

Status: CONTRADICTED FOR RANGE-BASED CPTC AS CORE.

The existence of cancellation is not enough; range-only certification did not expose it early enough.

## A-002 — Alpha-spending Serfling implementation is valid under declared assumptions

Evidence: EXP-047 and EXP-047R unit/property/adversarial checks, zero wrong accepts in committed corpora, zero bound violations in EXP-047R, deterministic replay, and exact fallback.

Status: SUPPORTED AT PHASE A/B, E1, WITHIN DECLARED FINITE-POPULATION ASSUMPTIONS.

This does not establish useful savings or model-wide correctness.

## A-003 — Certificate overhead is smaller than skipped work

Evidence:

- EXP-047 Python reference was about 8.6–9.1x simple full summation;
- EXP-047R C2 materialized-contribution CPU primitive/full-sum median was 2165.057x;
- EXP-047R evaluated essentially all contributions.

Status: CONTRADICTED FOR CURRENT REFERENCE IMPLEMENTATIONS.

The EXP-047R ratio is not an optimized lower bound and is not GPU evidence. The range family was already rejected by C1 oracle coverage independently of timing.

## A-004 — A decision-relevant low-dimensional projection is sufficient

Current evidence: pairwise LM-head decision reconstruction is exact, but model-wide nonlinear propagation and candidate selection remain unresolved.

Status: PARTIALLY SUPPORTED / UNVERIFIED MODEL-WIDE.

## A-005 — Probabilistic certification is acceptable

Current evidence: no final product acceptance criterion. Exact fallback exists; model-wide delta accounting does not.

Status: UNVERIFIED REQUIREMENT.

## A-006 — Small-model trends predict larger models

Current evidence: three small checkpoints only.

Status: UNVERIFIED FOR 70B/405B.

## A-007 — Target RAM/SSD capacity and bandwidth are sufficient

Status: UNVERIFIED; Phase D NOT TESTED.

## A-008 — Full hot state fits 8 GiB including KV/buffers/fallback

Status: UNVERIFIED; E0; Phase D NOT TESTED.

## A-009 — 4B-class speed can coexist with exact fallback

Current evidence: prior repair mechanisms and CPTC fallback/coverage failed. Same-bit traffic requires about 1.185% average target-equivalent stream fraction before overhead.

Status: HIGH-RISK AND UNSUPPORTED.

## A-010 — Auxiliary DAG/VM/certificate components aid the final runtime

Status: OPTIONAL. They may be reused only after a new core mechanism independently changes cost.

## A-011 — Loose global metadata, not intrinsic range behavior, caused CPTC-v1 failure

EXP-047R contradiction test:

```text
C1 exact realized min/max oracle median 100%
C1 p90 100%
pre-registered limits 10% / 25%
```

Status: CONTRADICTED.

Decision: reject range-only CPTC; do not tune C2/C3 to rescue it.

## A-012 — Sound static tile metadata can be computed automatically and be useful

Evidence:

- checkpoint output-weight column spans were computed automatically without training;
- zero bound violations across EXP-047R;
- C2 median and p90 still 100%, best 99.21875%.

Status: SOUNDNESS SUPPORTED AT E1; USEFULNESS CONTRADICTED FOR THE TESTED CORE ROLE.

## A-013 — One full target stream can be amortized across many exact accepted tokens

Assumption:

A sufficiently accurate block proposal, one exact teacher-forced target pass, and longest-prefix verification can preserve exact greedy output while amortizing target weight traffic.

EXP-048 evidence:

```text
B1 perfect future oracle
96 exact tokens / 1 target pass
fraction 1.0416667%
projected requirement 1.185185%
exact mismatches 0
```

Status: ARITHMETIC AND VERIFIER CONTRACT SUPPORTED AT E1 UNDER A PERFECT NON-DEPLOYABLE PROPOSAL; CAUSAL PROPOSAL SOURCE REMAINS UNSOLVED.

B1 is not runtime evidence because it uses future target tokens.

## A-014 — Early target layers can act as a useful training-free draft

EXP-048 contradiction evidence:

```text
18 cases, 54 fixed variants
maximum matching proposal prefix 1
p50 committed tokens per verification 1
minimum accounted fraction 1333.463%
p90 accounted fraction 2893.843%
```

Status: CONTRADICTED FOR THE TESTED SAME-CHECKPOINT PARTIAL-LAYER + FULL-LM-HEAD MECHANISM.

Changing only the selected early-layer count, temperature, or proposal tree does not address the failed exact-prefix prediction and repeated LM-head cost.

## A-015 — Hard target-only Jacobi can provide cheap long exact blocks

EXP-048 evidence:

```text
p50 58 target passes / 32 exact tokens
p50 fraction 181.25%
p90 fraction 193.75%
maximum matching prefix 3
```

Status: CONTRADICTED FOR THE TESTED HARD JACOBI CONTROL.

Every target pass and failed iteration was charged.

## A-016 — Continuous soft states and Anderson acceleration propagate useful causal information faster than hard Jacobi

Assumption:

A large block of soft token embeddings, updated by a small number of full target passes and bounded Anderson mixing, can yield a much longer hard exact prefix than discrete Jacobi without future information or training.

Status: ACTIVE UNVERIFIED ASSUMPTION FOR EXP-049.

Contradiction tests:

- exact verifier mismatch or future information;
- NaN/Inf, coefficient explosion, or unhandled numerical fallback;
- p50 exact matching prefix <16 after at most four target solver passes;
- p90 accounted target-equivalent fraction >10%;
- less than 4x p50 prefix improvement over hard Jacobi;
- worsening checkpoint-size trend.

## A-017 — An arbitrary causal target permits universal faster-than-one-position-per-round block solving

Risk:

Because token position `i` depends causally on the resolved token at `i-1`, a black-box target-only synchronous solver may be unable to guarantee more than one new exact position per target round in the worst case.

Status: ACTIVE THEORETICAL RISK FOR EXP-049; NOT YET PROVED.

Required contradiction/proof audit:

- formal target interface;
- adversarial finite causal model family;
- indistinguishability of later positions before predecessor resolution;
- exact scope of any average-case versus universal claim;
- whether continuous embeddings/Anderson actually add information or only extrapolate prior outputs.

A valid one-position-per-round lower bound would reject universal exact target-only fixed-point acceleration for the fixed arbitrary-model objective even if some prompts empirically improve.
