# VORTEX Work-Session Protocol

Last updated: 2026-08-04 Asia/Seoul

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
12. `docs/RESEARCH_EFFICIENCY_CONTRACT.md`
13. this file;
14. active experiment document, code, config, tests, workflow, latest result JSON, PR, and workflow logs.

Verify:

- current branch;
- head SHA;
- open PR;
- latest CI/workflow state;
- authoritative result commit;
- whether current hardware permits the declared phase.

Never rely on chat memory as the source of truth.

## Mandatory experiment lifecycle

### 1. Reuse prior evidence and closed families

Before selecting a hypothesis:

- read `FAILED_APPROACHES.md`;
- identify reusable code/data;
- identify the exact previous failure being addressed;
- reject a renamed repetition;
- identify whether the mechanism family is already closed by a lower bound, oracle ceiling, or real-checkpoint population result;
- require a new information source, asymptotic mechanism, execution dependency, or measured fact before reopening a closed family.

### 2. Apply E0 candidate-efficiency triage

Before opening an experiment branch, record the scorecard required by `docs/RESEARCH_EFFICIENCY_CONTRACT.md`:

- optimistic fully charged operation, traffic, and storage fractions;
- credible path toward at least one order-of-magnitude improvement and ultimately the approximately 1.185% target-equivalent fraction;
- mechanism novelty versus rejected approaches;
- real evidence, theorem, or architectural reason supporting plausibility;
- reason the effect should survive or strengthen with scale;
- arbitrary-checkpoint automation and universality;
- correctness/fail-closed contract;
- cheapest decisive falsification;
- exact implementation stage authorized only if that cheap Gate survives.

Reject or label auxiliary immediately when the favorable ceiling is merely incremental, population scaling is absent, or fully charged overhead erases the gain.

Do not select a hypothesis merely because it is mathematically enumerable, elegant, adjacent to the prior experiment, or not yet present in the experiment sequence.

### 3. Select one high-upside core hypothesis

The hypothesis must directly answer the core twelve questions in `RESEARCH_STATE.md` and pass the E0 efficiency triage. Auxiliary work must be labeled auxiliary.

The primary research track must prioritize mechanisms capable in principle of an order-of-magnitude or larger change. Isolated optimizations with a ceiling of a few tens of percent remain auxiliary unless a separately justified composition closes the remaining gap.

### 4. Pre-register the Gate

Commit before interpreting results:

- phase;
- evidence ceiling;
- correctness/error contract;
- future-information policy;
- success thresholds;
- rejection thresholds;
- optimistic target-scale ceiling;
- strongest counterexample;
- cheapest decisive falsification;
- 405B equations;
- unverified assumptions;
- explicit stop rule and prohibited continuation after failure.

### 5. Implement only the cheapest decisive stage

Preferred progression:

```text
resource/information upper bound
-> exact algebraic or causal certificate
-> favorable oracle upper bound
-> pinned small-real-checkpoint measurement
-> minimal operation replacement
-> backend/kernel implementation
-> target hardware
```

Do not implement a model-wide backend, physical kernel, broad parameter sweep, or long rescue workflow before the earlier Gate survives.

When a theorem, lower bound, or favorable oracle already rejects the target thresholds, stop. Do not construct an optimized implementation merely to reconfirm the negative result.

For Phase B, when authorized:

- slow reference;
- minimum optimized candidate;
- independent bound/oracle where possible;
- randomized/property tests;
- boundary and adversarial cases;
- fault injection;
- deterministic replay.

For Phase C, when authorized:

- unmodified pinned real checkpoint;
- real operation replacement;
- disjoint prompts;
- future-information audit;
- exact forward/layer/tile accounting.

### 6. Run only valid phases

Current GitHub environment may run Phase A/B and available small-model Phase C. Phase D is `NOT TESTED` until target hardware exists.

Never emulate a GPU, PCIe measurement, 405B run, TTFT, or tokens/second result with a CPU projection.

### 7. Save evidence

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

### 8. Interpret conservatively and efficiently

- Infrastructure failures are not scientific failures.
- Synthetic success is not real-model success.
- Small-model success is not 405B success.
- Future-token oracle results are not causal execution.
- A probabilistic certificate is not deterministic exactness.
- A fallback-heavy path is not a speedup.
- A best single matrix, prompt, row, head, or synthetic fragment is not population-level promotion.
- Passing storage alone does not rescue failed operations or traffic.
- Reducing bytes while doubling work does not promote a core.
- A useful falsification may close a family without increasing target feasibility.

### 9. Update durable state

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

Also update experiment docs/results, the older chronological ledger/session handoff when still used, PR decision, workflow references, and the efficiency contract when candidate-selection policy changes.

### 10. Commit and validate

Required current-environment commands:

```bash
python -m pytest -q
python scripts/run_validation.py
bash experiments/exp_xxx/run_current_env.sh
```

Inspect actual logs. A status badge alone is insufficient.

For documentation-only governance changes, run the repository validation path and all CI required by branch protection. Do not claim local execution when only GitHub CI was available.

### 11. Decide and stop

Use one explicit decision:

- PROMOTE;
- REVISE;
- REJECT CORE / RETAIN AUXILIARY;
- REJECT;
- INFRASTRUCTURE FAILURE — NO SCIENTIFIC DECISION.

Record why, which assumption changed, whether the result merely closed a family, and why the next candidate has higher expected value.

A rejected family may not continue through parameter sweeps, rank changes, threshold tuning, mode-order variants, or renamed decompositions without a new reopening premise.

## Research portfolio default

Unless evidence justifies another allocation, prioritize approximately:

```text
70% high-upside new execution paradigms capable in principle of 10x-100x change
20% cheap falsification, bounds, certificates, and real-checkpoint screening
10% auxiliary engineering, cleanup, and incremental optimization
```

This is a prioritization rule, not a claim about measured labor time.

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

- prior state/failures/decisions and the efficiency contract were read;
- the candidate passed or was rejected by E0 efficiency triage;
- optimistic target-scale ceiling and cheapest falsification were recorded;
- one hypothesis and thresholds were committed;
- only the implementation stage authorized by the current Gate was built;
- valid current-environment execution was attempted;
- raw logs and checksums were saved, or infrastructure failure documented;
- provenance labels were checked;
- evidence level was assigned;
- failed assumptions and family-closure implications were recorded;
- all durable root documents were updated;
- future GPU script and hardware plan were updated when applicable;
- next session can continue using GitHub alone;
- the next candidate's expected value versus rejected alternatives is stated;
- user-facing wording matches committed evidence.
