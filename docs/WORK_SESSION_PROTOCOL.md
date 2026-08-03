# VORTEX Work-Session Protocol

Last updated: 2026-08-03 Asia/Seoul

This protocol is mandatory for every meaningful research session.

## Startup checklist

Read and verify in this order:

1. `AGENTS.md`
2. `RESEARCH_STATE.md`
3. `FAILED_APPROACHES.md`
4. `DECISION_LOG.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`
8. `ARCHITECTURE.md`
9. `HARDWARE_VALIDATION_PLAN.md`
10. `REPRODUCIBILITY.md`
11. `docs/PROOF_FIRST_CONTRACT.md`
12. this file;
13. active experiment document, code, config, tests, workflow, latest result JSON, PR, and workflow logs.

Verify:

- current branch;
- head SHA;
- open PR;
- latest CI/workflow state;
- authoritative result commit;
- whether current hardware permits the declared phase.

Never rely on chat memory as the source of truth.

## Mandatory experiment lifecycle

### 1. Reuse prior evidence

Before selecting a hypothesis:

- read `FAILED_APPROACHES.md`;
- identify reusable code/data;
- identify the exact previous failure being addressed;
- reject a renamed repetition.

### 2. Select one core hypothesis

The hypothesis must directly answer the core twelve questions in `RESEARCH_STATE.md`. Auxiliary work must be labeled auxiliary.

### 3. Pre-register the Gate

Commit before interpreting results:

- phase;
- evidence ceiling;
- correctness/error contract;
- future-information policy;
- success thresholds;
- rejection thresholds;
- strongest counterexample;
- 405B equations;
- unverified assumptions.

### 4. Implement independently

For Phase B:

- slow reference;
- optimized candidate;
- independent bound/oracle where possible;
- randomized/property tests;
- boundary and adversarial cases;
- fault injection;
- deterministic replay.

For Phase C:

- unmodified pinned real checkpoint;
- real operation replacement;
- disjoint prompts;
- future-information audit;
- exact forward/layer/tile accounting.

### 5. Run only valid phases

Current GitHub environment may run Phase A/B and available small-model Phase C. Phase D is `NOT TESTED` until target hardware exists.

Never emulate a GPU, PCIe measurement, 405B run, TTFT, or tokens/second result with a CPU projection.

### 6. Save evidence

Use the experiment layout:

```text
docs/research/EXPERIMENT_XXX_<NAME>.md
experiments/exp_xxx/
results/exp_xxx/
tests/exp_xxx/
.github/workflows/exp_xxx_gate.yml
```

Save:

- raw stdout/stderr;
- environment inventory;
- config;
- raw metrics;
- processed summary;
- artifacts;
- checksums.

Separate `MEASURED`, `DERIVED`, `PROJECTED`, and `UNVERIFIED` in machine-readable results.

### 7. Interpret conservatively

- Infrastructure failures are not scientific failures.
- Synthetic success is not real-model success.
- Small-model success is not 405B success.
- Future-token oracle results are not causal execution.
- A probabilistic certificate is not deterministic exactness.
- A fallback-heavy path is not a speedup.

### 8. Update durable state

Before a progress response, update all applicable files:

```text
RESEARCH_STATE.md
NEXT_EXPERIMENT.md
DECISION_LOG.md
FAILED_APPROACHES.md
ARCHITECTURE.md
ASSUMPTION_REGISTER.md
VALIDATION_MATRIX.md
HARDWARE_VALIDATION_PLAN.md
REPRODUCIBILITY.md
```

Also update experiment docs/results, the older chronological ledger/session handoff when still used, PR decision, and workflow references.

### 9. Commit and validate

Required current-environment commands:

```bash
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_xxx/run_current_env.sh
```

Inspect actual logs. A status badge alone is insufficient.

### 10. Decide

Use one explicit decision:

- PROMOTE;
- REVISE;
- REJECT CORE / RETAIN AUXILIARY;
- REJECT;
- INFRASTRUCTURE FAILURE — NO SCIENTIFIC DECISION.

Record why and which assumption changed.

## Workflow isolation

Every experiment workflow must:

- verify referenced paths exist;
- pin dependencies;
- run experiment-specific tests first;
- use branch-specific concurrency;
- fail on missing metrics, NaN, wrong accepts, future-information leakage, or provenance violations;
- upload artifacts;
- commit evidence only after the Gate passes;
- never label current CPU runs as Phase D.

## Completion checklist

A session is not complete until:

- prior state/failures/decisions were read;
- one hypothesis and thresholds were committed;
- code/reference/tests were implemented;
- valid current-environment execution was attempted;
- raw logs and checksums were saved, or infrastructure failure documented;
- provenance labels were checked;
- evidence level was assigned;
- failed assumptions were recorded;
- all durable root documents were updated;
- future GPU script and hardware plan were updated when applicable;
- next session can continue using GitHub alone;
- user-facing wording matches committed evidence.
