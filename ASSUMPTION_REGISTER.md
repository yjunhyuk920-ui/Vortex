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

A causal training-free proposal mechanism can supply a long block, one exact teacher-forced target pass can verify it, and the longest matching prefix can preserve exact greedy output.

Known arithmetic:

```text
required target-equivalent stream fraction 0.01185185
zero-cost perfect-proposal minimum accepted block 85 tokens
real draft cost raises the requirement
```

Current evidence:

- exact causal block evaluation is structurally possible;
- an existing Jacobi implementation exists as a control;
- no committed training-free partial-layer self-draft evidence yet meets the accounting target.

Status: ACTIVE UNVERIFIED ASSUMPTION FOR EXP-048.

Contradiction test:

- exact mismatch or future information;
- p50 accepted tokens per verification <16 at the early Gate;
- p90 target-equivalent stream fraction >10%;
- cost not lower than exact sequential decoding;
- non-degrading 85-token acceptance cannot be approached across checkpoint sizes/tasks.

## A-014 — Early target layers can act as a useful training-free draft

Assumption: the same checkpoint's partial-layer hidden state plus its own output head predicts enough final-layer tokens to amortize exact block verification without an adapter.

Status: ACTIVE UNVERIFIED ASSUMPTION FOR EXP-048 B3.

All partial-layer reads, sequential proposal steps, output-head work, rejected positions, and correction passes must be charged.
