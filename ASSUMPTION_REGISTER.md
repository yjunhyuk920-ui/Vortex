# Assumption Register

No unverified assumption may be used as a success condition.

## A-001 — Range-certified signed cancellation is broadly exploitable

EXP-047/047R evaluated about 98–100% of contributions even under exact realized ranges.

Status: CONTRADICTED FOR RANGE-BASED CPTC AS CORE.

## A-002 — Alpha-spending Serfling implementation is valid in declared scope

Reference/property/adversarial checks, zero committed-corpus wrong accepts, zero bound violations, deterministic replay, exact fallback.

Status: SUPPORTED PHASE A/B E1; DOES NOT ESTABLISH SAVINGS.

## A-003 — Certificate overhead is smaller than skipped work

EXP-047 Python path about 8.6–9.1x full sum; EXP-047R C2 reference much slower while reading nearly all contributions.

Status: CONTRADICTED FOR TESTED IMPLEMENTATIONS.

## A-004 — Low-dimensional decision projection is sufficient model-wide

Pairwise LM-head reconstruction is exact; nonlinear model-wide propagation remains unresolved.

Status: PARTIAL / UNVERIFIED MODEL-WIDE.

## A-005 — Probabilistic certification is product-acceptable

No final model-wide delta or product requirement exists.

Status: UNVERIFIED.

## A-006 — Tiny-model trends predict 70B/405B

Three tiny checkpoints only.

Status: UNVERIFIED.

## A-007 — Target RAM/SSD capacity and bandwidth are sufficient

Status: UNVERIFIED; Phase D NOT TESTED.

## A-008 — Target/draft/KV/work state fits 8 GiB

Status: UNVERIFIED; E0; Phase D NOT TESTED.

## A-009 — 4B-class speed coexists with exact fallback

All deployable proposal mechanisms through EXP-050 remain far above 1.185185% target-equivalent traffic.

Status: HIGH-RISK AND UNSUPPORTED.

## A-010 — Auxiliary VM/DAG/certificate/verifier components aid final runtime

Status: OPTIONAL. Reuse only after a new core mechanism survives its own Gate.

## A-011 — Loose range metadata caused CPTC failure

EXP-047R exact realized range median/p90 100%.

Status: CONTRADICTED.

## A-012 — Sound static tile metadata is useful

Soundness passed; usefulness failed with median/p90 100%.

Status: SOUND E1, NOT USEFUL FOR TESTED CORE.

## A-013 — One target stream can verify many exact tokens

EXP-048 future oracle verified 96 exact tokens/one pass =1.0416667%.

Status: VERIFIER ARITHMETIC SUPPORTED E1; CAUSAL PROPOSAL SOURCE UNSOLVED.

## A-014 — Early target layers provide useful recursive draft tokens

EXP-048 max matching prefix 1, p50 committed 1, p90 2893.843%.

Status: CONTRADICTED FOR TESTED PARTIAL-LAYER DRAFT.

## A-015 — Hard Jacobi provides cheap long exact blocks

EXP-048 p50 58 target passes/32 tokens.

Status: CONTRADICTED.

## A-016 — Continuous Picard/Anderson propagates exact causal information faster

EXP-049 favorable p50 4.5, maximum 6, p90 168.778596%; Anderson/Jacobi 0.25x.

Status: CONTRADICTED FOR TESTED TARGET-ONLY FAMILY.

## A-017 — Arbitrary causal target permits universal >1 exact position/round target-only solving

Hidden triangular transcripts remained indistinguishable before predecessor resolution.

Status: CONTRADICTED WITHIN DECLARED BLACK-BOX ROUND INTERFACE.

## A-018 — Fixed target-independent external draft provides long exact prefixes across arbitrary targets

EXP-050 universal first-token counterexample produced matching prefix zero.

Status: CONTRADICTED FOR UNIVERSAL ARBITRARY-TARGET GUARANTEE.

## A-019 — Tested fixed external draft pool is practically useful

EXP-050 favorable exact-reference selection:

```text
p50 prefix 0.5
maximum prefix 3
p90 normalized fraction 163.20987654%
Korean and structured JSON coverage false
```

Status: CONTRADICTED FOR TINYSTORIES 1M/3M/8M FIXED POOL.

## A-020 — A 4B external draft can satisfy final budget if exact prefixes are long

PROJECTED:

```text
4/405 + 1/K <=0.01185185185
K >=507
```

Status: ARITHMETICALLY DERIVED; 507-TOKEN CROSS-MODEL EXACT PREFIX CONTRADICTED BY CURRENT POOL AND UNVERIFIED GENERALLY.

## A-021 — A causal target-independent selector can choose a useful external draft

The EXP-050 selector used exact reference and still failed. No deployable selector exists.

Status: UNVERIFIED GENERALLY; IRRELEVANT FOR TESTED POOL AFTER FAVORABLE ORACLE FAILURE.

## A-022 — Final next-token decision becomes suffix-stable after very few target layers

Assumption:

With the exact target greedy prefix fixed, the intermediate hidden state after a shallow block prefix, passed through the original final norm and LM head, already yields the final target token and no later block changes it.

This is different from recursive partial-layer drafting because the input prefix remains exact for every token state.

Status: ACTIVE FOR EXP-051.

Contradiction tests:

- non-deployable suffix-stable oracle median logical fraction >10%;
- p90 >25%;
- median block-depth fraction >10%;
- any required family median stable depth >50%;
- worsening depth with model size;
- output head alone consumes excessive fraction;
- late-decision adversarial residual chain finalizes only at last block.

## A-023 — A sound causal selector can know suffix stability without executing omitted layers

Current evidence: none. Exact-reference suffix-stable depth uses later target outputs and is non-deployable.

Status: UNVERIFIED.

A shallow oracle depth is necessary but not sufficient. A deployable tail certificate must bound all omitted nonlinear attention/MLP residual effects without executing them.

## A-024 — One LM-head probe plus shallow block prefix can approach 4B-class traffic

Assumption:

The target LM head and final norm, combined with a small prefix of block weights, fit below 1.185185% of the full 405B target stream.

Current evidence: no 405B architecture-specific measurement. EXP-051 will measure actual logical head/layer shares only on pinned tiny checkpoints.

Status: UNVERIFIED / PROJECTED ONLY.

A large tied output embedding may set a nonzero traffic floor even at depth zero.

## A-025 — A fixed early-exit depth works for every arbitrary target

Universal risk:

An arbitrary residual network can keep token `a` dominant through every early layer and flip to `b` only in the final layer.

Status: ACTIVE ADVERSARIAL CONTRADICTION TEST FOR EXP-051; EXPECTED UNSUPPORTED UNIVERSALLY.

<!-- EXP-052-AUTHORITATIVE-FINAL -->
## A-026/A-027/A-028 — Advice coverage, reuse, circuit compilation

A-026 enumerative exact advice generalizes across unseen families: CONTRADICTED (0% held-out hits). A-027 natural exact states repeat at least 85 times: CONTRADICTED on the corpus (median/max 1/1). A-028 a non-enumerative bit-exact weight-derived circuit remains compact: ACTIVE UNVERIFIED for EXP-053.
