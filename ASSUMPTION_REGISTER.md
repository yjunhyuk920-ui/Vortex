# Assumption Register

No unverified assumption may be used as a success condition.

## A-001 — Signed tile cancellation is broadly exploitable by range certification

Evidence: deterministic residual required 90–98% refinement; EXP-047 evaluated about 98%; EXP-047R exact-state range oracle evaluated 100% at median and p90.

Status: CONTRADICTED FOR RANGE-BASED CPTC AS CORE.

## A-002 — Alpha-spending Serfling implementation is valid under declared assumptions

Evidence: EXP-047/047R reference/property/adversarial tests, zero committed-corpus wrong accepts, zero bound violations, deterministic replay, exact fallback.

Status: SUPPORTED AT PHASE A/B, E1, WITHIN DECLARED FINITE-POPULATION ASSUMPTIONS.

This does not establish useful savings or model-wide correctness.

## A-003 — Certificate overhead is smaller than skipped work

Evidence: EXP-047 Python path about 8.6–9.1x full sum; EXP-047R materialized C2 primitive/full-sum median 2165.057x; nearly all contributions evaluated.

Status: CONTRADICTED FOR TESTED REFERENCE IMPLEMENTATIONS.

## A-004 — A decision-relevant low-dimensional projection is sufficient

Pairwise LM-head reconstruction is exact; model-wide nonlinear propagation and candidate selection remain unresolved.

Status: PARTIALLY SUPPORTED / UNVERIFIED MODEL-WIDE.

## A-005 — Probabilistic certification is acceptable

No final product acceptance criterion or model-wide delta accounting exists.

Status: UNVERIFIED REQUIREMENT.

## A-006 — Small-model trends predict larger models

Current evidence is three very small checkpoints only.

Status: UNVERIFIED FOR 70B/405B.

## A-007 — Target RAM/SSD capacity and bandwidth are sufficient

Status: UNVERIFIED; Phase D NOT TESTED.

## A-008 — Full hot state fits 8 GiB including KV/buffers/fallback

Status: UNVERIFIED; E0; Phase D NOT TESTED.

## A-009 — 4B-class speed can coexist with exact fallback

Same-bit arithmetic requires about 1.185185% average target-equivalent stream fraction before overhead. All deployable proposal mechanisms tested through EXP-049 remain far above it.

Status: HIGH-RISK AND UNSUPPORTED.

## A-010 — Auxiliary DAG/VM/certificate/verifier components aid the final runtime

Status: OPTIONAL. They may be reused only after a new core information/cost mechanism survives its own Gate.

## A-011 — Loose global metadata, not intrinsic range behavior, caused CPTC-v1 failure

EXP-047R exact realized range oracle median/p90 100% versus limits 10%/25%.

Status: CONTRADICTED.

## A-012 — Sound static tile metadata can be automatically useful

Column-span metadata was sound with zero violations but C2 median/p90 remained 100%.

Status: SOUNDNESS SUPPORTED E1; USEFULNESS CONTRADICTED FOR TESTED CORE ROLE.

## A-013 — One full target stream can be amortized across many exact accepted tokens

EXP-048 perfect future oracle: 96 exact tokens / one target pass =1.0416667%, exact 18/18.

Status: ARITHMETIC AND EXACT VERIFIER SUPPORTED E1 UNDER A NON-DEPLOYABLE PERFECT PROPOSAL. CAUSAL PROPOSAL SOURCE UNSOLVED.

## A-014 — Early target layers can act as a useful training-free draft

EXP-048: maximum matching proposal prefix 1, p50 committed 1, minimum fraction 1333.463%, p90 2893.843%.

Status: CONTRADICTED FOR SAME-CHECKPOINT PARTIAL-LAYER + FULL-LM-HEAD MECHANISM.

## A-015 — Hard target-only Jacobi can provide cheap long exact blocks

EXP-048 complete-generation control p50 58 target passes/32 tokens, p50 181.25%, p90 193.75%, maximum prefix 3.

Status: CONTRADICTED FOR TESTED HARD JACOBI CONTROL.

## A-016 — Continuous soft states and Anderson propagate useful causal information faster than hard Jacobi

EXP-049 favorable checkpoint evidence:

```text
oracle-best S1/S2 p50 prefix 4.5
maximum prefix 6
p90 fraction 168.778596%
S0 p50 prefix after four passes 4
S2 Anderson p50 prefix after four passes 1
S2/S0 improvement 0.25x
```

The reference was allowed to choose the best fixed S1/S2 trajectory per case.

Status: CONTRADICTED FOR THE TESTED TARGET-ONLY CONTINUOUS/ANDERSON FAMILY.

Changing only solver hyperparameters does not address the missing causal future information.

## A-017 — An arbitrary causal target permits universal faster-than-one-position-per-round target-only solving

EXP-049 adversarial evidence:

```text
Picard prefixes by round 1,2,3,4
Anderson prefixes by round 1,2,3,3
hidden suffix transcript indistinguishability true
```

Interface: one synchronous black-box causal target block evaluation per round; exact prefix, fixed initialization, all prior states/outputs, arbitrary continuous arithmetic/history; no external future information.

Status: CONTRADICTED WITHIN THE DECLARED INTERFACE.

This is a worst-case universal result, not a statement that every real prompt advances exactly one position.

## A-018 — A fixed target-independent external draft can provide long exact prefixes across arbitrary targets

Assumption:

An already published, unmodified small draft model can generate a long causal proposal for an arbitrary target, while exact block verification preserves target output and draft cost remains compatible with the 4B-class budget.

Universal risk:

For any fixed target-independent draft first token `a`, an arbitrary target can choose greedy token `b != a` on the same prompt. Therefore no fixed external draft can guarantee a nonzero exact prefix for every arbitrary target.

Status: ACTIVE FOR EXP-050 PRACTICAL FIXED-POOL AUDIT; UNIVERSAL GUARANTEE ALREADY HIGH-RISK.

Contradiction tests:

- executable first-token adversarial target;
- cross-checkpoint draft pool among TinyStories-1M/3M/8M;
- exact-reference favorable draft selection;
- p50 exact prefix <16 or p90 normalized fraction >10%;
- any required family with zero proposal acceptance;
- worsening target-size trend.

## A-019 — A 4B draft can satisfy the final target traffic budget if proposals are sufficiently long

PROJECTED same-bit arithmetic:

```text
draft/target ratio = 4/405 = 0.0098765432
required total fraction = 0.01185185185
fraction with perfect K-token proposal = 4/405 + 1/K
minimum K = 507
```

Status: ARITHMETICALLY DERIVED; EXACT 507-TOKEN CROSS-MODEL PREFIX IS UNVERIFIED AND EXPECTED TO BE DIFFICULT.

A shorter 85-token threshold applies only to a zero-cost proposal and may not be used for an actual 4B draft.

## A-020 — A causal target-independent draft selector can choose the useful external model without target future information

Current evidence: none. EXP-050 permits exact-reference oracle selection only as a favorable falsification upper bound.

Status: UNVERIFIED.

A positive pool oracle does not establish a deployable selector. A negative pool oracle eliminates selector engineering for that pool.
