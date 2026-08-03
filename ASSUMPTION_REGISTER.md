# Assumption Register

No assumption may be used as a success condition while its status is unverified.

## A-001 — Signed tile cancellation is exploitable

Assumption:

Transformer linear-operator tile contributions, under a causal randomized order, have enough cancellation that a valid finite-population confidence sequence often closes before most tiles are read.

Current evidence:

Deterministic signed residual experiments observed cancellation but required roughly 90–98% refinement. No valid probabilistic tile certificate has been tested.

Status: UNVERIFIED.

Contradiction test:

EXP-047 Phase B adversarial distributions and Phase C held-out real-model states require near-100% tiles or produce wrong accepts.

Dependencies: EXP-047.

## A-002 — Time-uniform finite-population bounds can be implemented without hidden independence violations

Assumption:

Sampling tiles without replacement and using a declared confidence sequence yields valid coverage under adaptive stopping.

Current evidence:

Mathematical derivation and independent reference implementation pending.

Status: UNVERIFIED.

Contradiction test:

Exact enumeration for small populations plus Monte Carlo property tests find empirical undercoverage beyond tolerance or a proof review identifies an invalid assumption.

Dependencies: EXP-047 Phase A/B.

## A-003 — Certificate overhead is smaller than skipped work

Assumption:

Selector, statistics, projections, and confidence updates cost materially less than evaluating the omitted dense tiles.

Current evidence:

None for CPTC.

Status: UNVERIFIED.

Contradiction test:

MEASURED Phase B/C selector time or operations exceed saved tile work at the promotion thresholds.

Dependencies: EXP-047.

## A-004 — A decision-relevant low-dimensional projection is sufficient

Assumption:

A candidate can certify a token or operator decision without constructing every omitted output coordinate exactly.

Current evidence:

LM-head top-1 certificates and constructed decision lower bounds exist, but model-wide nonlinear propagation remains unresolved.

Status: PARTIALLY SUPPORTED, UNVERIFIED MODEL-WIDE.

Contradiction test:

Real-operation replacement shows downstream nonlinearities require near-complete hidden-vector precision before any token certificate can close.

Dependencies: future EXP-048+.

## A-005 — Probabilistic certification is acceptable under a declared error budget

Assumption:

A runtime mode with an explicitly union-bounded wrong-commit probability can be considered an original-output-preserving mode for some users/tests.

Current evidence:

No product acceptance criterion is defined. Deterministic exact mode remains the strict reference.

Status: UNVERIFIED REQUIREMENT.

Contradiction test:

Project acceptance requires bitwise determinism only, or union accounting makes the usable `delta` too small for any savings.

Dependencies: EXP-047 and final validation protocol.

## A-006 — Small-model tile-certification trends predict larger models

Assumption:

Certified skip fractions remain stable or improve with width/depth.

Current evidence:

None. TinyLlama results cannot establish this.

Status: UNVERIFIED FOR LARGE MODELS.

Contradiction test:

Measurements across at least three sizes show worsening tile fractions, fallback, or certificate overhead.

Dependencies: Phase C/E3 and Phase D/E5.

## A-007 — 405B cold storage and host memory are sufficient

Assumption:

A target machine can hold the original checkpoint and runtime metadata in RAM and/or SSD while sustaining the required access pattern.

Current evidence:

Only formulas and small mmap tests. No 405B file or target SSD measurement.

Status: UNVERIFIED; Phase D NOT TESTED.

Contradiction test:

Actual capacity, random-access latency, endurance, or bandwidth fails `HARDWARE_VALIDATION_PLAN.md` thresholds.

Dependencies: Phase D.

## A-008 — 8 GiB hot-state budget can include KV, buffers, and fallback tile

Assumption:

The final architecture can schedule all GPU-resident state within 8 GiB at the declared context.

Current evidence:

No complete candidate and no target GPU measurement.

Status: UNVERIFIED; E0.

Contradiction test:

Analytical memory certificate or measured peak allocation exceeds 8 GiB.

Dependencies: full architecture Gate and Phase D.

## A-009 — 4B-class speed can coexist with exact fallback

Assumption:

Normal-path certification succeeds often enough that fallback amortization fits the 4B-class latency budget.

Current evidence:

Prior repair mechanisms failed badly; CPTC has no measurements.

Status: UNVERIFIED AND HIGH RISK.

Contradiction test:

Required fallback frequency or cold bytes/token exceeds the declared budget.

Dependencies: EXP-047 onward.

## A-010 — Auxiliary DAG/VM components will be useful in the final runtime

Assumption:

Existing exact pointer VM and suffix-DAG work will store certificates, capsules, or repeated execution states in a future operation-skipping architecture.

Current evidence:

Functional bounded implementations only.

Status: OPTIONAL/UNVERIFIED.

Contradiction test:

The winning execution principle requires no such state or their storage/build cost exceeds benefit.

Dependencies: none; must not constrain core research.
